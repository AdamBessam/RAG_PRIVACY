"""
CPB v3 — Brique 1 : QueryRiskScorerV3.

Inherits QueryRiskScorer (v1) with two additions:
  - S5 uses dynamic centroids from BootstrapV3 (no longer hardcoded to SENSITIVE_TAXONOMY)
  - CONTEXT_TARGETS is extended with learned PII type names discovered from the corpus
  - domain is stored for future per-domain weight tuning

Weights and signal families are unchanged (maxima still sum to 1.0).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import SPACY_MODEL
from countermeasure.cpb_query_risk import (
    CONTEXT_TARGETS,
    EXTRACTIVE_VERBS,
    JAILBREAK_PATTERNS,
    NER_ENTITY_INCREMENT,
    NER_MAX_WEIGHT,
    EXTRACTIVE_MAX_WEIGHT,
    EXTRACTIVE_VERB_INCREMENT,
    CONTEXT_TARGET_INCREMENT,
    JAILBREAK_MAX_WEIGHT,
    JAILBREAK_PATTERN_INCREMENT,
    SESSION_FLAG_INCREMENT,
    SESSION_MAX_WEIGHT,
    SEMANTIC_MAX_WEIGHT,
    SEMANTIC_NOISE_FLOOR,
    SEMANTIC_SCALE,
    QueryRiskScorer,
)
from countermeasure.cpb_models import QueryRiskResult


class QueryRiskScorerV3(QueryRiskScorer):
    """
    CPB v3 Block 1: domain-aware query risk scorer.

    Key differences from v1:
      - centroids injected at construction (pre-built by CPBBootstrapV3)
      - context_re augmented with learned PII type names from the corpus
      - domain stored for context (weights unchanged)
    """

    def __init__(
        self,
        centroids: dict,       # category -> np.ndarray(384,), from BootstrapResult
        learned_types: set,    # PII type strings discovered in 0a, from BootstrapResult
        domain: str = "legal",
        spacy_model: str = SPACY_MODEL,
    ):
        super().__init__(spacy_model=spacy_model)
        self.domain = domain
        # Inject pre-built centroids so _get_centroids() is never called
        self._centroids = centroids

        # Extend context targets with lowercase PII type names (e.g. "person", "iban_code")
        extended = set(CONTEXT_TARGETS) | {t.lower() for t in learned_types}
        # Rebuild context_re on top of the augmented set
        self.context_re = re.compile(
            r"\b(" + "|".join(map(re.escape, sorted(extended))) + r")\b",
            re.IGNORECASE,
        )

    # Override S5 to use injected centroids directly (no lazy taxonomy build)
    def _semantic_signal(self, query: str) -> float:
        if not self._centroids:
            return 0.0
        try:
            embedder = self._get_embedder()
            q_emb = embedder.embed_texts([query])   # (1, 384), L2-normalized
            max_sim = max(float(q_emb[0] @ c) for c in self._centroids.values())
            return min(
                SEMANTIC_MAX_WEIGHT,
                max(0.0, max_sim - SEMANTIC_NOISE_FLOOR) * SEMANTIC_SCALE,
            )
        except Exception:
            return 0.0
