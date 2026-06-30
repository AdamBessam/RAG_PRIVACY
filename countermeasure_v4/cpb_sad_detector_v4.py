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
)

# v4-only override of the parent's BLOCK_CATEGORY_COUNT=2: a full block now
# needs 3+ sensitive categories at once, and even then a masked synthesis is
# tried first instead of discarding the whole response outright.
BLOCK_CATEGORY_COUNT_V4 = 3
BLOCK_MASK_FRACTION_LIMIT = 0.70


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

    # ── Override: 3-category block (not 2) + masked synthesis before refusal ──

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

        # Tier 3 — 3+ sensitive categories: try an LLM reformulation first
        # (generalize the sensitive specifics, keep the rest), then a plain
        # sentence-level mask (Phi-3-verified, see _mask_or_block), and only
        # fall back to a full refusal if both would leave a near-empty /
        # still-unsafe response.
        if len(categories) >= BLOCK_CATEGORY_COUNT_V4:
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
                        reasoning=reasoning + " (block tier downgraded to LLM-reformulated synthesis)",
                        filter_triggered=3,
                        sbert_category_scores=category_scores,
                    )

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
                    category_scores=category_scores,
                    note=f" (block tier downgraded to sentence mask, {frac_masked:.0%} of sentences redacted)",
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
                reasoning=reasoning,
                filter_triggered=3,
                sbert_category_scores=category_scores,
            )

        # Tier 1 — low confidence: mask, then verify with Phi-3 that the
        # masked sentences actually removed the flagged categories (SBERT
        # sentence-level masking can miss phrasings the centroid doesn't
        # score high enough, even when Phi-3 itself flagged the category —
        # escalate to a full block rather than silently under-mask).
        if confidence < MASK_CONFIDENCE_THRESHOLD:
            masked = self._mask_sensitive(response, categories)
            return self._mask_or_block(
                query=query, chunks=chunks, masked=masked, categories=categories,
                confidence=confidence, reasoning=reasoning, max_similarity=max_similarity,
                category_scores=category_scores, note="",
            )

        if reask_callback is not None:
            try:
                new_response = reask_callback(categories[0])
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

        masked = self._mask_sensitive(response, categories)
        return self._mask_or_block(
            query=query, chunks=chunks, masked=masked, categories=categories,
            confidence=confidence, reasoning=reasoning, max_similarity=max_similarity,
            category_scores=category_scores, note=" (reask failed, masked)",
        )

    # ── Verify a masked response actually removed the flagged categories ──────

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
        """
        Sentence-level masking (SBERT, mask_threshold=0.30) can miss a
        sentence that doesn't score high enough against the centroid even
        though Phi-3 flagged the category for the response as a whole (seen
        in practice: a health detail phrased differently from the taxonomy's
        anchor sentences slipped through unmasked). Re-run Phi-3 on the
        MASKED text — the same judge that caught it the first time — before
        trusting it. If it still detects the category, escalate to a full
        block instead of returning an under-masked response.
        """
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

    # ── LLM reformulation: generalize sensitive specifics, keep the rest ──────

    def _synthesize_response(self, response: str, categories: list[str]) -> str | None:
        """
        Asks Phi-3 to rewrite the response so the sensitive specifics flagged
        in `categories` are generalized into non-identifying terms (e.g. "a
        health condition" instead of a named diagnosis) while everything else
        stays as-is. Returns None if the call fails or yields nothing usable
        — callers fall back to sentence-level masking in that case.
        """
        categories_str = ", ".join(categories)
        prompt = (
            "Rewrite the response below so it no longer discloses specific, "
            f"identifiable details related to: {categories_str}. "
            "Generalize each sensitive detail into a broad, non-identifying term "
            "(e.g. 'a health condition' instead of a named diagnosis or procedure, "
            "'a personal belief' instead of a specific religion, "
            "'a political affiliation' instead of a named party). "
            "Keep every other fact in the response exactly as it is. "
            "Do not add disclaimers, do not mention that anything was changed, "
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
