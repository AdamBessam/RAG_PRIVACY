"""
CPB v6 — Brique B6 : SADDetectorV6 (Sensitive Attribute Disclosure).

Fusion autonome de countermeasure/sad_detector.py (base, cascade F1→F2→F3) et
countermeasure_v4/cpb_sad_detector_v4.py (taxonomie/centroïdes dynamiques +
cascade qualité synthesis→mask→block) en une seule classe, pour que
countermeasure_v6 n'importe RIEN des autres dossiers countermeasure*.

Différence de pipeline avec v1-v5 : en v6, B6 s'exécute APRÈS le masquage PII
de la réponse (B3/B4 sur la réponse), pas avant. Le detect() ci-dessous reçoit
donc déjà une réponse partiellement masquée — voir cpb_naive_rag_v6.py.

Cascade :
  F1  regex          : sujet individuel présent dans la réponse (ou la requête) ?
  F2  SBERT centroïde : proximité sémantique aux catégories sensibles du domaine ?
  F3  Phi-3 Mini      : confirme le SAD + décision structurée JSON.

Décision (qualité d'abord, la plus destructive en dernier) :
  1. synthesize : reformulation LLM qui retire le lien sensible
  2. mask       : masquage phrase par phrase (SBERT)
  3. block      : refus complet (seulement si les deux précédents échouent)
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import OLLAMA_BASE_URL
from countermeasure_v6.cpb_models_v6 import SADResult

# Réponse masquée trop agressivement -> inutile de la retourner, direct block.
BLOCK_MASK_FRACTION_LIMIT = 0.70

# Filtre 1 : sujet individuel (placeholders Presidio + mots de rôle courants).
INDIVIDUAL_RE = re.compile(
    r"\[(?:PERSON|APPLICANT|COMPLAINANT|VICTIM|INDIVIDUAL|NRP)_?\d*\]"
    r"|\b(?:applicant|complainant|victim|patient|the individual|the person)\b"
    r"|\b(?:he|she|him|his|her|they|them|their)\b",
    re.IGNORECASE,
)

DEFAULT_SBERT_THRESHOLD = 0.42


class SADDetectorV6:
    """
    CPB v6 Block B6 — détecte une divulgation d'attribut sensible (SAD) en
    utilisant la taxonomie/centroïdes dynamiques de B0 (pas de taxonomie codée
    en dur). Exécuté sur la réponse déjà masquée par B3/B4.
    """

    def __init__(
        self,
        dynamic_taxonomy: dict,          # category -> list[str], depuis BootstrapResult
        centroids: dict,                 # category -> np.ndarray(384,), depuis BootstrapResult
        domain: str = "general",
        sbert_threshold: float = DEFAULT_SBERT_THRESHOLD,
        mask_threshold: float = 0.30,
        phi3_model: str = "phi3:mini",
        ollama_host: str = OLLAMA_BASE_URL,
    ):
        self.dynamic_taxonomy = dynamic_taxonomy
        self.domain = domain
        self.sbert_threshold = sbert_threshold
        self.mask_threshold = mask_threshold
        self.phi3_model = phi3_model
        self.ollama_host = ollama_host
        self._embedder = None
        self._centroids = centroids  # injectés — jamais reconstruits paresseusement

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(
        self,
        query: str,
        chunks: list[dict],
        response: str,
        reask_callback=None,
    ) -> SADResult:
        # F1 — O(n) regex : sujet individuel dans la réponse ou la requête ?
        if not INDIVIDUAL_RE.search(response) and not INDIVIDUAL_RE.search(query):
            return SADResult(
                sad_detected=False,
                attribute_categories=[],
                max_similarity=0.0,
                confidence=0.0,
                decision="pass",
                response=response,
                reasoning="F1: no individual subject found in response or query",
                filter_triggered=1,
            )

        # F2 — proximité SBERT aux centroïdes (gate : pas de F3 si rien ne matche)
        hit_categories, max_sim, category_scores = self._sbert_proximity(response)
        if not hit_categories:
            return SADResult(
                sad_detected=False,
                attribute_categories=[],
                max_similarity=max_sim,
                confidence=0.0,
                decision="pass",
                response=response,
                reasoning=f"F2: no sentence within SBERT threshold of any sensitive centroid (max_sim={max_sim:.2f})",
                filter_triggered=2,
                sbert_category_scores=category_scores,
            )

        # F3 — jugement Phi-3, seulement sur les catégories que SBERT a signalées
        candidate_categories = hit_categories
        phi3 = self._phi3_judge(query, chunks, response, candidate_categories)

        if not phi3["sad_detected"]:
            return SADResult(
                sad_detected=False,
                attribute_categories=candidate_categories,
                max_similarity=max_sim,
                confidence=phi3["confidence"],
                decision="pass",
                response=response,
                reasoning=phi3["reasoning"],
                filter_triggered=3,
                sbert_category_scores=category_scores,
            )

        confirmed_categories = phi3["attribute_categories"] or candidate_categories
        return self._apply_decision(
            query=query,
            chunks=chunks,
            response=response,
            categories=confirmed_categories,
            confidence=phi3["confidence"],
            reasoning=phi3["reasoning"],
            max_similarity=max_sim,
            category_scores=category_scores,
            reask_callback=reask_callback,
        )

    # ── Filtre 2 — SBERT ───────────────────────────────────────────────────────

    def _sbert_proximity(self, text: str) -> tuple[list[str], float, dict[str, float]]:
        embedder = self._get_embedder()
        centroids = self._get_centroids()
        if not centroids:
            return [], 0.0, {}

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]
        if not sentences:
            sentences = [text]

        sent_embs = embedder.embed_texts(sentences)

        hit_categories: list[str] = []
        max_sim = 0.0
        category_scores: dict[str, float] = {}
        for category, centroid in centroids.items():
            best_sim = float((sent_embs @ centroid).max())
            category_scores[category] = best_sim
            if best_sim > max_sim:
                max_sim = best_sim
            if best_sim >= self.sbert_threshold:
                hit_categories.append(category)

        return hit_categories, max_sim, category_scores

    def _get_centroids(self) -> dict[str, np.ndarray]:
        return self._centroids

    # ── Filtre 3 — Phi-3 Mini (prompt enrichi du domaine) ─────────────────────

    def _phi3_judge(
        self,
        query: str,
        chunks: list[dict],
        response: str,
        candidate_categories: list[str],
    ) -> dict:
        context_text = "\n---\n".join(c.get("text", "") for c in chunks[:3])
        categories_str = ", ".join(candidate_categories)

        prompt = (
            "You are a privacy auditor for a RAG system. "
            "Determine whether the response below constitutes a Sensitive Attribute Disclosure (SAD).\n\n"
            "Definition: a SAD occurs when a response links an identifiable individual "
            "(even anonymized, e.g. [PERSON_1]) to a sensitive personal attribute.\n\n"
            "IMPORTANT: A SAD must be detected even if the user explicitly asked for this information. "
            "The user's query intent does NOT determine whether a SAD exists. "
            "A SAD is determined solely by whether the response reveals a sensitive attribute "
            "about an identifiable individual — regardless of the query.\n\n"
            f"Corpus domain: {self.domain}\n"
            f"Sensitive categories relevant to this domain: {categories_str}\n\n"
            f"Query: {query}\n\n"
            f"Retrieved context (anonymized):\n{context_text}\n\n"
            f"Response to audit:\n{response}\n\n"
            "Answer in valid JSON only, no extra text:\n"
            "{\n"
            '  "sad_detected": true or false,\n'
            '  "attribute_categories": ["CAT1"],\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "reasoning": "one sentence"\n'
            "}"
        )

        try:
            import requests
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={"model": self.phi3_model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            raw = resp.json().get("response", "")
            return self._parse_phi3_output(raw, candidate_categories)
        except Exception as exc:
            return {
                "sad_detected": True,
                "attribute_categories": candidate_categories,
                "confidence": 0.55,
                "reasoning": f"Phi-3 unavailable ({exc}), defaulting to SBERT signal",
            }

    @staticmethod
    def _parse_phi3_output(raw: str, fallback_categories: list[str]) -> dict:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            parsed = json.loads(raw[start:end])
            return {
                "sad_detected": bool(parsed.get("sad_detected", False)),
                "attribute_categories": parsed.get("attribute_categories", fallback_categories),
                "confidence": float(parsed.get("confidence", 0.5)),
                "reasoning": str(parsed.get("reasoning", "")),
            }
        except Exception:
            return {
                "sad_detected": True,
                "attribute_categories": fallback_categories,
                "confidence": 0.5,
                "reasoning": "JSON parse error, defaulting to SBERT signal",
            }

    # ── Décision qualité d'abord : synthesis → mask → block ───────────────────

    def _apply_decision(
        self,
        query: str,
        chunks: list[dict],
        response: str,
        categories: list[str],
        confidence: float,
        reasoning: str,
        max_similarity: float,
        category_scores: dict,
        reask_callback,
    ) -> SADResult:
        # 1. Synthesis — reformulation LLM qui retire le lien sensible
        rewritten = self._synthesize_response(response, categories)
        if rewritten and not INDIVIDUAL_RE.search(rewritten):
            verify = self._phi3_judge(query, chunks, rewritten, categories)
            if not verify["sad_detected"]:
                return SADResult(
                    sad_detected=True,
                    attribute_categories=categories,
                    max_similarity=max_similarity,
                    confidence=confidence,
                    decision="synthesize",
                    response=rewritten,
                    reasoning=reasoning + " (resolved via LLM-reformulated synthesis)",
                    filter_triggered=3,
                    sbert_category_scores=category_scores,
                )

        # 2. Masquage phrase par phrase (direct block si ça viderait la réponse)
        masked = self._mask_sensitive(response, categories)
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", response) if s.strip()]
        n_total = len(sentences) or 1
        n_redacted = masked.count("[SENSITIVE_ATTRIBUTE_REDACTED]")
        frac_masked = n_redacted / n_total
        leftover = masked.replace("[SENSITIVE_ATTRIBUTE_REDACTED]", "").strip()

        if frac_masked < BLOCK_MASK_FRACTION_LIMIT and leftover:
            return self._mask_or_block(
                query=query, chunks=chunks, masked=masked, categories=categories,
                confidence=confidence, reasoning=reasoning, max_similarity=max_similarity,
                category_scores=category_scores, note=" (synthesis unavailable/unsafe, sentence-masked)",
            )

        # 3. Dernier recours
        return SADResult(
            sad_detected=True,
            attribute_categories=categories,
            max_similarity=max_similarity,
            confidence=confidence,
            decision="block",
            response=(
                "This information cannot be disclosed as it contains "
                "multiple sensitive personal attributes."
            ),
            reasoning=reasoning,
            filter_triggered=3,
            sbert_category_scores=category_scores,
        )

    def _mask_or_block(
        self,
        query: str,
        chunks: list[dict],
        masked: str,
        categories: list[str],
        confidence: float,
        reasoning: str,
        max_similarity: float,
        category_scores: dict,
        note: str,
    ) -> SADResult:
        """Revérifie le masquage phrase par phrase avec Phi-3 avant de le
        retourner ; sinon escalade en block plutôt que renvoyer une réponse
        sous-masquée."""
        verify = self._phi3_judge(query, chunks, masked, categories)
        if not verify["sad_detected"]:
            return SADResult(
                sad_detected=True,
                attribute_categories=categories,
                max_similarity=max_similarity,
                confidence=confidence,
                decision="mask",
                response=masked,
                reasoning=reasoning + note,
                filter_triggered=3,
                sbert_category_scores=category_scores,
            )
        return SADResult(
            sad_detected=True,
            attribute_categories=categories,
            max_similarity=max_similarity,
            confidence=confidence,
            decision="block",
            response=(
                "This information cannot be disclosed as it contains "
                "multiple sensitive personal attributes."
            ),
            reasoning=reasoning + f" (mask left residual disclosure per Phi-3 recheck: {verify.get('reasoning', '')})",
            filter_triggered=3,
            sbert_category_scores=category_scores,
        )

    def _synthesize_response(self, response: str, categories: list[str]) -> str | None:
        """Demande à Phi-3 de retirer (pas d'atténuer) le lien sensible, en
        gardant le reste de la réponse tel quel."""
        categories_str = ", ".join(categories)
        prompt = (
            "Rewrite the response below so it no longer links any identifiable "
            f"individual to a sensitive personal attribute related to: {categories_str}. "
            "Remove those specific facts entirely rather than rephrasing them in "
            "vaguer terms — do not say things like 'a health condition' or 'a "
            "political belief' either, since that still discloses the link. "
            "Restructure the surrounding sentences so the text still reads "
            "naturally without the removed information. "
            "Keep every other fact in the response exactly as it is. "
            "Do not add disclaimers, do not mention that anything was removed, "
            "do not invent new information.\n\n"
            f"Response to rewrite:\n{response}\n\n"
            "Rewritten response (text only, no preamble):"
        )
        try:
            import requests
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={"model": self.phi3_model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            rewritten = resp.json().get("response", "").strip()
            return rewritten or None
        except Exception:
            return None

    # ── Masquage phrase par phrase (SBERT, vocabulaire-agnostique) ────────────

    def _mask_sensitive(self, response: str, categories: list[str]) -> str:
        embedder = self._get_embedder()
        centroids = self._get_centroids()

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response) if s.strip()]
        if not sentences:
            return response

        sent_embs = embedder.embed_texts(sentences)

        masked = []
        for i, sentence in enumerate(sentences):
            should_mask = any(
                float(sent_embs[i] @ centroids[cat]) >= self.mask_threshold
                for cat in categories
                if cat in centroids
            )
            masked.append("[SENSITIVE_ATTRIBUTE_REDACTED]" if should_mask else sentence)

        return " ".join(masked)

    # ── Lazy loaders ────────────────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            from embeddings.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder
