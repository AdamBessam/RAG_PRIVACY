"""
CPB v4 — Brique 6 : SADDetectorV4.

Identique à countermeasure_v3.cpb_sad_detector_v3.SADDetectorV3 — seul le nom
de classe change, pour que countermeasure_v4 reste autonome (n'importe rien de
countermeasure_v3). Aucune différence de logique : le domaine n'est utilisé ici
que comme contexte dans le prompt Phi-3, peu importe qu'il vienne de
nvidia/domain-classifier ou du repli Llama.

  - dynamic_taxonomy and centroids injected from BootstrapV4 (replace SENSITIVE_TAXONOMY)
  - domain passed to Phi-3 prompt for contextually-aware judgment
  - candidate categories default to dynamic_taxonomy.keys() instead of SENSITIVE_TAXONOMY.keys()

The 3-filter cascade (F1 regex → F2 SBERT → F3 Phi-3) and 3-tier decision logic
(block / mask / reask) are unchanged from v1.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import OLLAMA_BASE_URL, SPACY_MODEL
from countermeasure.cpb_models import SADResult
from countermeasure.sad_detector import (
    SADDetector,
    INDIVIDUAL_RE,
    DEFAULT_SBERT_THRESHOLD,
    MASK_CONFIDENCE_THRESHOLD,
    BLOCK_CATEGORY_COUNT,
)


class SADDetectorV4(SADDetector):
    """
    CPB v4 Block 6 — detects Sensitive Attribute Disclosure using dynamic taxonomy.

    Key differences from v1:
      - centroids injected at construction (pre-built by CPBBootstrapV4)
      - candidate categories drawn from dynamic_taxonomy (not SENSITIVE_TAXONOMY)
      - Phi-3 prompt enriched with domain for contextually-aware reasoning
    """

    def __init__(
        self,
        dynamic_taxonomy: dict,          # category -> list[str], from BootstrapResult
        centroids: dict,                 # category -> np.ndarray(384,), from BootstrapResult
        domain: str = "legal",
        sbert_threshold: float = DEFAULT_SBERT_THRESHOLD,
        mask_threshold: float = 0.30,
        phi3_model: str = "phi3:mini",
        ollama_host: str = OLLAMA_BASE_URL,
    ):
        super().__init__(
            sbert_threshold=sbert_threshold,
            mask_threshold=mask_threshold,
            phi3_model=phi3_model,
            ollama_host=ollama_host,
        )
        self.dynamic_taxonomy = dynamic_taxonomy
        self.domain = domain
        # Override the parent's _centroids with pre-built ones
        self._centroids = centroids

    # ── Override: use injected centroids (skip lazy build) ───────────────────

    def _get_centroids(self, embedder=None) -> dict[str, np.ndarray]:
        return self._centroids

    # ── Override: use dynamic_taxonomy categories as candidate set ────────────

    def detect(
        self,
        query: str,
        chunks: list[dict],
        response: str,
        reask_callback=None,
    ) -> SADResult:
        # F1 — O(n) regex: individual subject in response or query?
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

        # F2 — SBERT centroid proximity (sentence-level, max-pooling)
        hit_categories, max_sim, category_scores = self._sbert_proximity(response)
        # If SBERT found nothing, pass all dynamic categories to F3
        candidate_categories = (
            hit_categories if hit_categories else list(self.dynamic_taxonomy.keys())
        )

        # F3 — Phi-3 Mini judgment (always runs when F1 passes)
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

    # ── Override: inject domain into Phi-3 prompt ────────────────────────────

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
