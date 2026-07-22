"""
CPB v6 — Brique B0 : auto-découverte, détection de domaine, génération de taxonomie.

Copie autonome de countermeasure_v4/cpb_bootstrap_v4.py (countermeasure_v6
n'importe RIEN des autres dossiers countermeasure*). Logique inchangée :

  0a  Discover PII types from ChromaDB metadata (Presidio on sample)
  0b  Infer corpus domain via nvidia/domain-classifier (repli : Llama zero-shot)
  0c  Generate domain-specific sensitive categories + Presidio hints via Llama
  0d  Seed each category with Llama-generated synthetic phrases, grounded with
      the corpus sentences semantically nearest to those seeds (anonymized)
  0e  Build L2-normalized SBERT centroids (all-MiniLM-L6-v2, local)

Exécuté UNE SEULE FOIS à l'instanciation de CPBNaiveRAGV6.
"""

import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import requests

from config import LLAMA_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

NVIDIA_MODEL_ID = "nvidia/domain-classifier"
NVIDIA_SAMPLE_SIZE = 30
NVIDIA_MAX_CHARS = 2000

SEMANTIC_RELEVANCE_FLOOR = 0.40
D0_SENTENCE_POOL_MAX = 500
D0_SYNTHETIC_SEEDS = 8
D0_REAL_SEEDS = 7


@dataclass
class BootstrapResult:
    domain: str
    domain_confidence: float
    domain_source: str            # "nvidia_domain_classifier" ou "llama_fallback"
    learned_types: set
    dynamic_categories: list
    dynamic_taxonomy: dict        # category -> list[str]
    category_hints: dict          # category -> set[str] (Presidio entity types)
    centroids: dict                # category -> np.ndarray (384,)
    used_fallback: bool = False


class NvidiaDomainClassifier:
    """Détection de domaine via nvidia/domain-classifier (DeBERTa-v3-base, 26
    catégories web fixes). Repli propre sur Llama si torch/transformers
    manquent ou si le téléchargement échoue."""

    _MODEL_ID = NVIDIA_MODEL_ID

    def __init__(self):
        self._torch = None
        self.model = None
        self.tokenizer = None
        self.config = None
        try:
            import torch
            from torch import nn
            from transformers import AutoConfig, AutoModel, AutoTokenizer
            from huggingface_hub import PyTorchModelHubMixin

            class _CustomModel(nn.Module, PyTorchModelHubMixin):
                def __init__(self, config):
                    super().__init__()
                    self.model = AutoModel.from_pretrained(config["base_model"], dtype=torch.float32)
                    self.dropout = nn.Dropout(config["fc_dropout"])
                    self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

                def forward(self, input_ids, attention_mask):
                    features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                    dropped = self.dropout(features)
                    outputs = self.fc(dropped)
                    return torch.softmax(outputs[:, 0, :], dim=1)

            self._torch = torch
            self.config = AutoConfig.from_pretrained(self._MODEL_ID)
            self.tokenizer = AutoTokenizer.from_pretrained(self._MODEL_ID)
            self.model = _CustomModel.from_pretrained(self._MODEL_ID)
            self.model.eval()
        except Exception as exc:
            logger.warning(f"[CPBBootstrapV6 0b] nvidia/domain-classifier unavailable ({exc})")
            self.model = None

    def is_available(self) -> bool:
        return self.model is not None

    def classify(self, texts: list[str]) -> list[tuple[str, float]]:
        inputs = self.tokenizer(texts, return_tensors="pt", padding="longest", truncation=True)
        with self._torch.no_grad():
            outputs = self.model(inputs["input_ids"], inputs["attention_mask"])
        confidences, predicted = outputs.max(dim=1)
        return [
            (self.config.id2label[idx.item()], float(conf.item()))
            for idx, conf in zip(predicted, confidences)
        ]


class CPBBootstrapV6:
    """Brique B0 v6 : one-shot corpus analysis produisant un BootstrapResult.
    Appelée une seule fois dans __init__ de CPBNaiveRAGV6."""

    def __init__(
        self,
        store,
        ollama_host: str = OLLAMA_BASE_URL,
        llama_model: str = LLAMA_MODEL,
        seed: int = 42,
    ):
        self.store = store
        self.ollama_host = ollama_host
        self.llama_model = llama_model
        self.seed = seed
        self._embedder = None
        self._domain_classifier = None

    def run(self) -> BootstrapResult:
        """Exécute les étapes 0a → 0e. Ne lève jamais d'exception."""
        try:
            learned_types = self._step_0a()
            chunks_sample = self._sample_chunks(50)
            domain, domain_confidence, domain_source = self._step_0b(chunks_sample)
            dynamic_categories, category_hints = self._step_0c(domain)
            dynamic_taxonomy = self._step_0d(dynamic_categories, domain, chunks_sample, category_hints)
            centroids = self._step_0e(dynamic_taxonomy)

            result = BootstrapResult(
                domain=domain,
                domain_confidence=domain_confidence,
                domain_source=domain_source,
                learned_types=learned_types,
                dynamic_categories=dynamic_categories,
                dynamic_taxonomy=dynamic_taxonomy,
                category_hints=category_hints,
                centroids=centroids,
                used_fallback=False,
            )
            self._log_mlflow(result)
            logger.info(
                f"[CPBBootstrapV6] domain={domain} ({domain_confidence:.2f}, source={domain_source}), "
                f"categories={dynamic_categories}, types={len(learned_types)}"
            )
            return result
        except Exception as exc:
            logger.warning(f"[CPBBootstrapV6] Failed ({exc!r}), returning empty bootstrap result.")
            return BootstrapResult(
                domain="general",
                domain_confidence=0.0,
                domain_source="none",
                learned_types=set(),
                dynamic_categories=[],
                dynamic_taxonomy={},
                category_hints={},
                centroids={},
                used_fallback=True,
            )

    # ── Step 0a : PII type discovery (Presidio sur documents bruts) ──────────

    def _step_0a(self) -> set[str]:
        try:
            from presidio_analyzer import AnalyzerEngine
            analyzer = AnalyzerEngine()
            result = self.store.collection.get(limit=50, include=["documents"])
            docs = result.get("documents") or []
            types: set[str] = set()
            for doc in docs:
                if not doc:
                    continue
                for finding in analyzer.analyze(text=doc[:2000], language="en"):
                    types.add(finding.entity_type.upper())
            logger.info(f"[CPBBootstrapV6 0a] {len(types)} PII types discovered via Presidio")
            return types
        except Exception as exc:
            logger.warning(f"[CPBBootstrapV6 0a] Presidio discovery failed: {exc}")
            return set()

    # ── Step 0b : Domain inference ────────────────────────────────────────────

    def _step_0b(self, chunks: list[str]) -> tuple[str, float, str]:
        classifier = self._get_domain_classifier()
        if classifier.is_available():
            try:
                domain, confidence = self._step_0b_nvidia(chunks, classifier)
                return domain, confidence, "nvidia_domain_classifier"
            except Exception as exc:
                logger.warning(f"[CPBBootstrapV6 0b] nvidia classifier inference failed ({exc}), falling back to Llama")
        else:
            logger.warning("[CPBBootstrapV6 0b] nvidia/domain-classifier unavailable, falling back to Llama")

        domain, confidence = self._step_0b_llama_fallback(chunks)
        return domain, confidence, "llama_fallback"

    def _step_0b_nvidia(self, chunks: list[str], classifier: NvidiaDomainClassifier) -> tuple[str, float]:
        sample = [c[:NVIDIA_MAX_CHARS] for c in chunks[:NVIDIA_SAMPLE_SIZE] if c and c.strip()]
        if not sample:
            raise ValueError("no chunks available for domain classification")

        predictions = classifier.classify(sample)
        votes: dict[str, list[float]] = {}
        for label, conf in predictions:
            votes.setdefault(label, []).append(conf)

        best_label = max(votes, key=lambda l: len(votes[l]))
        vote_share = len(votes[best_label]) / len(predictions)
        mean_conf = sum(votes[best_label]) / len(votes[best_label])
        confidence = vote_share * mean_conf

        domain = best_label.lower()
        logger.info(
            f"[CPBBootstrapV6 0b] nvidia/domain-classifier: {best_label} "
            f"({len(votes[best_label])}/{len(predictions)} chunks, conf={confidence:.2f})"
        )
        return domain, confidence

    def _step_0b_llama_fallback(self, chunks: list[str]) -> tuple[str, float]:
        excerpt = "\n---\n".join(c[:300] for c in chunks[:10])
        prompt = (
            "You are a corpus analyst. Based on the text excerpts below, "
            "identify the domain of this corpus.\n\n"
            f"Excerpts:\n{excerpt}\n\n"
            "Respond in valid JSON only. "
            'Example: {"domain": "legal", "confidence": 0.9}'
        )
        try:
            raw = self._llama_call(prompt)
            parsed = self._parse_json(raw)
            domain = str(parsed.get("domain", "general")).lower().strip()
            confidence = float(parsed.get("confidence", 0.5))
            return domain, confidence
        except Exception as exc:
            logger.warning(f"[CPBBootstrapV6 0b] Llama fallback failed ({exc}), default=general")
            return "general", 0.0

    def _get_domain_classifier(self) -> NvidiaDomainClassifier:
        if self._domain_classifier is None:
            self._domain_classifier = NvidiaDomainClassifier()
        return self._domain_classifier

    # ── Step 0c : Category generation + Presidio hints ───────────────────────

    def _step_0c(self, domain: str) -> tuple[list[str], dict[str, set[str]]]:
        prompt = (
            f"You are a privacy expert. For a corpus in the '{domain}' domain, "
            "list the categories of SPECIAL sensitive personal attributes — facts that "
            "reveal an intimate or protected aspect of a person when described in natural "
            "language (for example: a health condition, a political opinion, a religious "
            "belief, sexual orientation, ethnic origin, or the domain's equivalent "
            "sensitive facts).\n"
            "Do NOT include ordinary identifiers or contact data such as names, dates, "
            "addresses, phone numbers, emails, employer/job, education, or ID numbers — "
            "those are masked separately by a dedicated PII engine and must be EXCLUDED here.\n"
            "List between 3 and 8 such categories.\n"
            "For each category, also list which Presidio NLP entity types would signal its presence in text.\n"
            "Available Presidio entity types: PERSON, LOCATION, ORGANIZATION, DATE_TIME, NRP, "
            "NATIONALITY, MEDICAL_LICENSE, DISEASE, CHEMICAL, IBAN_CODE, CREDIT_CARD, "
            "US_BANK_NUMBER, EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, US_PASSPORT, IP_ADDRESS\n"
            "Use short uppercase snake_case names for categories.\n"
            "Respond in valid JSON only.\n"
            'Example: {"categories": ['
            '{"name": "HEALTH", "presidio_types": ["MEDICAL_LICENSE", "DISEASE", "CHEMICAL"]}, '
            '{"name": "POLITICAL_OPINION", "presidio_types": ["NRP", "ORGANIZATION"]}'
            ']}'
        )
        try:
            raw = self._llama_call(prompt)
            parsed = self._parse_json(raw)
            raw_cats = parsed.get("categories", [])

            categories: list[str] = []
            hints: dict[str, set[str]] = {}
            for item in raw_cats:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).upper().strip()
                if not re.match(r"^[A-Z][A-Z0-9_]{1,39}$", name):
                    continue
                categories.append(name)
                types = {str(t).upper().strip() for t in item.get("presidio_types", []) if t}
                if types:
                    hints[name] = types

            if 3 <= len(categories) <= 8:
                logger.info(f"[CPBBootstrapV6 0c] {len(categories)} categories with {sum(len(v) for v in hints.values())} hints")
                return categories, hints
        except Exception as exc:
            logger.warning(f"[CPBBootstrapV6 0c] Category generation failed ({exc}), returning empty categories")
        return [], {}

    # ── Step 0d : Anchor phrase enrichment — sélection sémantique ─────────────

    def _step_0d(
        self,
        categories: list[str],
        domain: str,
        chunks: list[str],
        category_hints: dict[str, set[str]],
    ) -> dict[str, list[str]]:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            analyzer = AnalyzerEngine()
            anonymizer = AnonymizerEngine()
        except ImportError:
            analyzer = anonymizer = None

        all_sentences: list[str] = []
        for chunk in chunks:
            for sent in re.split(r"(?<=[.!?])\s+", chunk):
                sent = sent.strip()
                if len(sent) > 20:
                    all_sentences.append(sent)

        anon_sentences = (
            self._anonymize_pool(all_sentences, analyzer, anonymizer)
            if anonymizer is not None else []
        )
        embedder = self._get_embedder()
        corpus_embs = embedder.embed_texts(anon_sentences) if anon_sentences else None

        taxonomy: dict[str, list[str]] = {}
        for category in categories:
            synthetic = self._fill_with_synthetic([], category, domain, target=D0_SYNTHETIC_SEEDS)
            real = (
                self._select_relevant_sentences(synthetic, anon_sentences, corpus_embs, embedder)
                if corpus_embs is not None and synthetic else []
            )
            phrases = list(dict.fromkeys(synthetic + real))
            if len(phrases) < D0_SYNTHETIC_SEEDS:
                phrases = self._fill_with_synthetic(phrases, category, domain, target=15)
            taxonomy[category] = phrases[:15] if phrases else self._fallback_phrases(category)

        return taxonomy

    def _fill_with_synthetic(
        self,
        phrases: list[str],
        category: str,
        domain: str,
        target: int = 15,
        max_attempts: int = 5,
    ) -> list[str]:
        seen = set(phrases)
        attempts = 0
        while len(phrases) < target and attempts < max_attempts:
            attempts += 1
            new_phrases = self._generate_synthetic_phrases(category, domain, exclude=phrases, attempt=attempts)
            if not new_phrases:
                break
            for p in new_phrases:
                if p not in seen:
                    seen.add(p)
                    phrases.append(p)
                    if len(phrases) >= target:
                        break
        return phrases

    def _anonymize_pool(
        self,
        sentences: list[str],
        analyzer,
        anonymizer,
    ) -> list[str]:
        pool: list[str] = []
        for sent in sentences[:D0_SENTENCE_POOL_MAX]:
            try:
                text = sent[:500]
                results = analyzer.analyze(text=text, language="en")
                if results:
                    text = anonymizer.anonymize(text=text, analyzer_results=results).text
                pool.append(text)
            except Exception:
                continue
        return pool

    def _select_relevant_sentences(
        self,
        anchor_phrases: list[str],
        anon_sentences: list[str],
        corpus_embs,
        embedder,
        k: int = D0_REAL_SEEDS,
    ) -> list[str]:
        anchor_embs = embedder.embed_texts(anchor_phrases)
        anchor = anchor_embs.mean(axis=0)
        norm = np.linalg.norm(anchor)
        if norm < 1e-9:
            return []
        anchor = anchor / norm
        sims = corpus_embs @ anchor
        selected: list[str] = []
        for idx in np.argsort(sims)[::-1]:
            if sims[idx] < SEMANTIC_RELEVANCE_FLOOR:
                break
            selected.append(anon_sentences[idx])
            if len(selected) >= k:
                break
        return selected

    def _generate_synthetic_phrases(
        self,
        category: str,
        domain: str,
        exclude: list[str] | None = None,
        attempt: int = 0,
    ) -> list[str]:
        label = category.lower().replace("_", " ")
        avoid_clause = ""
        if exclude:
            avoid_block = "\n".join(f"- {p}" for p in exclude[:15])
            avoid_clause = (
                "Do not repeat or rephrase any of these already-generated sentences:\n"
                f"{avoid_block}\n"
            )
        prompt = (
            f"Generate 3 to 5 short realistic sentences (1-2 sentences each) that describe "
            f"a person's {label} attribute in a {domain or 'legal'} document context. "
            "Do not use real names. "
            f"{avoid_clause}"
            "Respond in valid JSON only.\n"
            'Example: {"phrases": ["The individual has a chronic condition.", "She disclosed her faith."]}'
        )
        try:
            raw = self._llama_call(prompt, seed=self.seed + attempt)
            parsed = self._parse_json(raw)
            phrases = [str(p).strip() for p in parsed.get("phrases", []) if len(str(p).strip()) > 15]
            return phrases[:5]
        except Exception:
            return []

    def _fallback_phrases(self, category: str) -> list[str]:
        label = category.lower().replace("_", " ")
        return [
            f"The individual has a sensitive {label} attribute.",
            f"This information concerns {label} and requires protection.",
        ]

    # ── Step 0e : SBERT centroids ──────────────────────────────────────────────

    def _step_0e(self, taxonomy: dict[str, list[str]]) -> dict[str, np.ndarray]:
        return self._build_centroids(taxonomy)

    def _build_centroids(self, taxonomy: dict[str, list[str]]) -> dict[str, np.ndarray]:
        embedder = self._get_embedder()
        centroids: dict[str, np.ndarray] = {}
        for category, sentences in taxonomy.items():
            if not sentences:
                continue
            embs = embedder.embed_texts(sentences)
            centroid = embs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[category] = centroid / (norm + 1e-9)
        return centroids

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sample_chunks(self, n: int) -> list[str]:
        try:
            result = self.store.collection.get(limit=n, include=["documents"])
            return [d for d in (result.get("documents") or []) if d and d.strip()]
        except Exception:
            return []

    def _llama_call(self, prompt: str, seed: int | None = None) -> str:
        resp = requests.post(
            f"{self.ollama_host}/api/generate",
            json={
                "model": self.llama_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "seed": seed if seed is not None else self.seed},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    @staticmethod
    def _parse_json(raw: str) -> dict:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object in LLM response")
        return json.loads(raw[start:end])

    def _get_embedder(self):
        if self._embedder is None:
            from embeddings.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    # ── MLflow traceability ───────────────────────────────────────────────────

    def _log_mlflow(self, result: BootstrapResult) -> None:
        try:
            import mlflow
            mlflow.log_param("cpb_v6_domain", result.domain)
            mlflow.log_param("cpb_v6_domain_confidence", round(result.domain_confidence, 3))
            mlflow.log_param("cpb_v6_domain_source", result.domain_source)
            mlflow.log_param("cpb_v6_categories", ",".join(result.dynamic_categories))
            mlflow.log_param("cpb_v6_learned_types", ",".join(sorted(result.learned_types)))
            mlflow.log_param("cpb_v6_used_fallback", result.used_fallback)
            mlflow.log_param("cpb_v6_seed", self.seed)

            payload = {
                "domain": result.domain,
                "domain_confidence": result.domain_confidence,
                "domain_source": result.domain_source,
                "categories": result.dynamic_categories,
                "category_hints": {k: list(v) for k, v in result.category_hints.items()},
                "taxonomy": {k: v for k, v in result.dynamic_taxonomy.items()},
                "seed": self.seed,
                "used_fallback": result.used_fallback,
            }
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, artifact_path="cpb_v6_bootstrap")
            os.unlink(tmp_path)
        except Exception:
            pass
