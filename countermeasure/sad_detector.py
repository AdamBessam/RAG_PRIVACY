"""
CPB Block 6 — Sensitive Attribute Disclosure (SAD) Detector.

A SAD occurs when a response establishes a link between an identifiable
individual (even anonymized, e.g. [PERSON_1]) and a sensitive attribute
from the categories: HEALTH, POLITICS, ETHNIC, SEX, BELIEF.

Three-filter cascade (cheap → expensive):
  F1  regex          : individual subject present in response?
  F2  SBERT centroid : semantic proximity to sensitive categories?
  F3  Phi-3 Mini     : confirm SAD + produce structured JSON decision.

3-tier action (driven by Phi-3 confidence):
  confidence < 0.70              → mask   : replace sensitive sentence(s)
  confidence ≥ 0.70, 1 category  → reask  : constrained regeneration, fallback to mask
  ≥ 2 categories                 → block  : full response suppression
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import OLLAMA_BASE_URL, SPACY_MODEL
from countermeasure.cpb_models import SADResult


# ── Taxonomy ───────────────────────────────────────────────────────────────────
# Representative sentences per sensitive category used to build SBERT centroids.
SENSITIVE_TAXONOMY: dict[str, list[str]] = {
    "HEALTH": [
        "The patient was diagnosed with tuberculosis.",
        "Suspected hepatitis or meningoencephalitis was found.",
        "He suffers from a mental health disorder.",
        "Medical examination revealed a serious illness.",
        "The individual has a chronic condition requiring treatment.",
        "She was hospitalized due to a medical emergency.",
        "The applicant disclosed a disability affecting daily life.",
        "The doctor referred him to the hospital for suspected disease.",
    ],
    "POLITICS": [
        "The individual holds communist political beliefs.",
        "She is an active member of a trade union.",
        "His political opinion led to persecution by the government.",
        "The applicant was a political activist opposing the regime.",
        "Trade union membership caused professional retaliation.",
        "Political affiliation was the basis of the asylum claim.",
        "She holds the title of Chairperson in the local branch of the political party.",
        "The applicant served as a local branch leader of a political party.",
        "He was elected as party representative at the regional level.",
        "She was the Chairperson of the regional party organization.",
        "The individual held an official role within a political party.",
    ],
    "ETHNIC": [
        "The applicant is of Roma ethnic origin.",
        "His nationality and racial background were targeted.",
        "She belongs to a minority ethnic group.",
        "The individual's country of origin led to discrimination.",
        "Ethnic identity was the basis of the complaint.",
        "Racial profiling was applied to the person.",
    ],
    "SEX": [
        "The individual's sexual orientation is homosexual.",
        "She identifies as lesbian and faced discrimination.",
        "Gender identity and transgender status were disclosed.",
        "Sexual orientation was revealed in the proceedings.",
        "He was persecuted because of his sexual preference.",
    ],
    "BELIEF": [
        "The applicant practices Islam as their religion.",
        "Religious belief and faith-based practices were mentioned.",
        "He is a devout Christian whose beliefs caused conflict.",
        "The individual's religious conviction led to an asylum claim.",
        "Her faith and spiritual practice were a source of persecution.",
    ],
}

# ── Filter 1 regex ─────────────────────────────────────────────────────────────
# Matches anonymized placeholders produced by Presidio anonymizer AND common
# legal/medical role words used in European court documents.
INDIVIDUAL_RE = re.compile(
    r"\[(?:PERSON|APPLICANT|COMPLAINANT|VICTIM|INDIVIDUAL|NRP)_?\d*\]"
    r"|\b(?:applicant|complainant|victim|patient|the individual|the person)\b"
    r"|\b(?:he|she|him|his|her|they|them|their)\b",
    re.IGNORECASE,
)

# ── Filter 2 threshold ─────────────────────────────────────────────────────────
DEFAULT_SBERT_THRESHOLD = 0.42

# ── 3-tier thresholds ──────────────────────────────────────────────────────────
MASK_CONFIDENCE_THRESHOLD = 0.70
BLOCK_CATEGORY_COUNT = 2



class SADDetector:
    """
    CPB Block 6 — detects Sensitive Attribute Disclosure in RAG responses.

    All heavy components (spaCy, SBERT, Phi-3) are lazy-loaded on first use.
    """

    def __init__(
        self,
        sbert_threshold: float = DEFAULT_SBERT_THRESHOLD,
        mask_threshold: float = 0.30,
        phi3_model: str = "phi3:mini",
        ollama_host: str = OLLAMA_BASE_URL,
    ):
        self.sbert_threshold = sbert_threshold
        self.mask_threshold = mask_threshold
        self.phi3_model = phi3_model
        self.ollama_host = ollama_host

        self._embedder = None
        self._centroids: dict[str, np.ndarray] | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect(
        self,
        query: str,
        chunks: list[dict],
        response: str,
        reask_callback=None,
    ) -> SADResult:
        """
        Run the 3-filter cascade on a (query, chunks, response) triplet.

        reask_callback: callable(attribute_category: str) -> str
            Called with the confirmed sensitive category name when a constrained
            regeneration is attempted (Tier 2 action).
        """
        # Filter 1 — O(n) regex, essentially free
        # Individual subject can be in the query ("What title did she hold...") even
        # if the response only returns the bare attribute value ("Chairperson (Kreisvorsitzende)").
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

        # Filter 2 — SBERT centroid proximity (informational only, does not block)
        hit_categories, max_sim, category_scores = self._sbert_proximity(response)
        # If SBERT found no close category, pass all categories to Phi-3 so it
        # can still catch implicit disclosures that SBERT missed.
        candidate_categories = hit_categories if hit_categories else list(SENSITIVE_TAXONOMY.keys())

        # Filter 3 — Phi-3 Mini judgment (always runs when F1 passes)
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

    # ── Filter 2 — SBERT ───────────────────────────────────────────────────────

    def _sbert_proximity(self, text: str) -> tuple[list[str], float, dict[str, float]]:
        """
        Sentence-level SBERT with max-pooling.

        A SAD needs only ONE sentence linking an individual to a sensitive
        attribute. Document-level embedding dilutes that signal when the
        sensitive sentence is surrounded by neutral ones. We embed each
        sentence separately and take the maximum similarity per category.

        Returns (hit_categories, max_sim, per_category_scores).
        """
        embedder = self._get_embedder()
        centroids = self._get_centroids(embedder)

        # Split into sentences; fall back to full text if splitting yields nothing
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]
        if not sentences:
            sentences = [text]

        # (N, 384) matrix — all sentence embeddings at once, already normalized
        sent_embs = embedder.embed_texts(sentences)

        hit_categories: list[str] = []
        max_sim = 0.0
        category_scores: dict[str, float] = {}
        for category, centroid in centroids.items():
            # Dot product of each sentence with the category centroid → take best
            best_sim = float((sent_embs @ centroid).max())
            category_scores[category] = best_sim
            if best_sim > max_sim:
                max_sim = best_sim
            if best_sim >= self.sbert_threshold:
                hit_categories.append(category)

        return hit_categories, max_sim, category_scores

    def _get_centroids(self, embedder) -> dict[str, np.ndarray]:
        if self._centroids is None:
            self._centroids = {}
            for category, sentences in SENSITIVE_TAXONOMY.items():
                embs = embedder.embed_texts(sentences)   # (N, 384), already normalized
                centroid = embs.mean(axis=0)
                norm = np.linalg.norm(centroid)
                self._centroids[category] = centroid / (norm + 1e-9)
        return self._centroids

    # ── Filter 3 — Phi-3 Mini ─────────────────────────────────────────────────

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
            "(even anonymized, e.g. [PERSON_1]) to a sensitive attribute "
            "from: HEALTH, POLITICS, ETHNIC, SEX, BELIEF.\n\n"
            f"Query: {query}\n\n"
            f"Retrieved context (anonymized):\n{context_text}\n\n"
            f"Response to audit:\n{response}\n\n"
            f"Candidate sensitive categories signalled by SBERT: {categories_str}\n\n"
            "Answer in valid JSON only, no extra text:\n"
            "{\n"
            '  "sad_detected": true or false,\n'
            '  "attribute_categories": ["HEALTH"],\n'
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
            # Phi-3 unavailable: trust SBERT signal with reduced confidence
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

    # ── 3-tier decision ────────────────────────────────────────────────────────

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

        # Tier 3 — multiple sensitive categories → block immediately
        if len(categories) >= BLOCK_CATEGORY_COUNT:
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

        # Tier 1 — low confidence → simple sentence-level masking
        if confidence < MASK_CONFIDENCE_THRESHOLD:
            return SADResult(
                sad_detected=True,
                attribute_categories=categories,
                max_similarity=max_similarity,
                confidence=confidence,
                decision="mask",
                response=self._mask_sensitive(response, categories),
                reasoning=reasoning,
                filter_triggered=3,
                sbert_category_scores=category_scores,
            )

        # Tier 2 — high confidence, single category → constrained reask
        if reask_callback is not None:
            try:
                new_response = reask_callback(categories[0])
                # Re-check only F1 + F2 on the new response (no recursive F3)
                if not INDIVIDUAL_RE.search(new_response):
                    return SADResult(
                        sad_detected=True,
                        attribute_categories=categories,
                        max_similarity=max_similarity,
                        confidence=confidence,
                        decision="reask",
                        response=new_response,
                        reasoning=f"Constrained reask resolved {categories[0]} disclosure",
                        filter_triggered=3,
                        sbert_category_scores=category_scores,
                    )
                recheck_cats, _, _ = self._sbert_proximity(new_response)
                if not recheck_cats:
                    return SADResult(
                        sad_detected=True,
                        attribute_categories=categories,
                        max_similarity=max_similarity,
                        confidence=confidence,
                        decision="reask",
                        response=new_response,
                        reasoning=f"Constrained reask resolved {categories[0]} disclosure",
                        filter_triggered=3,
                        sbert_category_scores=category_scores,
                    )
            except Exception:
                pass

        # Reask unavailable or failed → fallback to masking
        return SADResult(
            sad_detected=True,
            attribute_categories=categories,
            max_similarity=max_similarity,
            confidence=confidence,
            decision="mask",
            response=self._mask_sensitive(response, categories),
            reasoning=reasoning + " (reask failed, masked)",
            filter_triggered=3,
            sbert_category_scores=category_scores,
        )

    # ── Sentence-level masker ─────────────────────────────────────────────────

    def _mask_sensitive(self, response: str, categories: list[str]) -> str:
        """
        Semantic sentence-level masking: uses the same SBERT centroids as F2.
        Any sentence whose cosine similarity to a detected category centroid
        exceeds sbert_threshold is replaced by [SENSITIVE_ATTRIBUTE_REDACTED].
        This is vocabulary-agnostic — no keyword lists needed.
        """
        embedder = self._get_embedder()
        centroids = self._get_centroids(embedder)

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response) if s.strip()]
        if not sentences:
            return response

        sent_embs = embedder.embed_texts(sentences)  # (N, 384), normalized

        masked = []
        for i, sentence in enumerate(sentences):
            should_mask = any(
                float(sent_embs[i] @ centroids[cat]) >= self.mask_threshold
                for cat in categories
                if cat in centroids
            )
            masked.append("[SENSITIVE_ATTRIBUTE_REDACTED]" if should_mask else sentence)

        return " ".join(masked)

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            from embeddings.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder
