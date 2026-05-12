import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import spacy

from config import SPACY_MODEL
from countermeasure.cpb_models import QueryRiskResult


EXTRACTIVE_VERBS = (
    "copy", "print", "output", "return", "show", "reveal", "extract",
    "dump", "list", "repeat", "verbatim", "raw",
)

CONTEXT_TARGETS = (
    "context", "document", "documents", "chunk", "chunks", "database",
    "metadata", "source", "retrieved",
)

JAILBREAK_PATTERNS = (
    r"\bignore\b.{0,60}\b(instruction|instructions|policy|rules|system)\b",
    r"\bbypass\b.{0,60}\b(instruction|instructions|policy|rules|system)\b",
    r"\boverride\b.{0,60}\b(instruction|instructions|policy|rules|system)\b",
    r"\bfrom now on you are\b",
    r"\broleplay\b",
    r"\bjailbreak\b",
    r"\bdeveloper mode\b",
    r"\bdo not refuse\b",
    r"\bdo not summarize\b",
    r"\braw json\b",
)

# Query-risk calibration.
#
# The score is intentionally rule-based and transparent. The four signal
# families are weighted by attack specificity:
# - S3 jailbreak patterns are the strongest signal because they directly encode
#   policy/system override intent.
# - S2 extractive/context-targeting terms are strong because they indicate data
#   exfiltration intent, but they can still appear in benign analytical queries.
# - S1 NER is weaker: mentioning a person/place/date is common in legitimate
#   legal questions.
# - S4 session risk is also moderate: it captures repeated suspicious behavior
#   without making one previous event dominate every future turn.
#
# The maxima sum to 1.0 so the final score is interpretable without relying on
# saturation from clamping.
NER_MAX_WEIGHT = 0.15
EXTRACTIVE_MAX_WEIGHT = 0.25
JAILBREAK_MAX_WEIGHT = 0.35
SESSION_MAX_WEIGHT = 0.10
SEMANTIC_MAX_WEIGHT = 0.15

# S5 linear ramp: sim < SEMANTIC_NOISE_FLOOR → 0, sim >= SEMANTIC_SATURATION → max weight
SEMANTIC_NOISE_FLOOR = 0.25
SEMANTIC_SATURATION = 0.50
SEMANTIC_SCALE = SEMANTIC_MAX_WEIGHT / (SEMANTIC_SATURATION - SEMANTIC_NOISE_FLOOR)

# Per-hit increments are capped by the family maxima above. These values mean:
# 4 NER entities saturate S1; 3 strong extraction terms plus context terms
# saturate S2; 2 jailbreak patterns saturate S3; one direct session flag
# saturates S4.
NER_ENTITY_INCREMENT = 0.04
EXTRACTIVE_VERB_INCREMENT = 0.08
CONTEXT_TARGET_INCREMENT = 0.05
JAILBREAK_PATTERN_INCREMENT = 0.20
SESSION_FLAG_INCREMENT = SESSION_MAX_WEIGHT


class QueryRiskScorer:
    """
    CPB block 1A.

    Signals:
    S1: spaCy NER on the query
    S2: English extractive verbs / context-targeting regex
    S3: jailbreak regex
    S4: multi-turn session risk
    S5: semantic proximity to sensitive attribute categories (SBERT)
    """

    def __init__(self, spacy_model: str = SPACY_MODEL):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model], check=False)
            self.nlp = spacy.load(spacy_model)

        self.extractive_re = re.compile(
            r"\b(" + "|".join(map(re.escape, EXTRACTIVE_VERBS)) + r")\b",
            re.IGNORECASE,
        )
        self.context_re = re.compile(
            r"\b(" + "|".join(map(re.escape, CONTEXT_TARGETS)) + r")\b",
            re.IGNORECASE,
        )
        self.jailbreak_res = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]
        self.session_flags: dict[str, float] = {}
        self._embedder = None
        self._centroids = None

    def score(self, query: str, session_id: str = "default") -> QueryRiskResult:
        query = query or ""
        doc = self.nlp(query)

        ner_entities = [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

        extractive_hits = len(self.extractive_re.findall(query))
        context_hits = len(self.context_re.findall(query))
        jailbreak_hits = sum(1 for pattern in self.jailbreak_res if pattern.search(query))

        signals = {
            "s1_ner": min(NER_MAX_WEIGHT, NER_ENTITY_INCREMENT * len(ner_entities)),
            "s2_extractive": min(
                EXTRACTIVE_MAX_WEIGHT,
                EXTRACTIVE_VERB_INCREMENT * extractive_hits
                + CONTEXT_TARGET_INCREMENT * context_hits,
            ),
            "s3_jailbreak": min(
                JAILBREAK_MAX_WEIGHT,
                JAILBREAK_PATTERN_INCREMENT * jailbreak_hits,
            ),
            "s4_session": min(SESSION_MAX_WEIGHT, self.session_flags.get(session_id, 0.0)),
            "s5_semantic": self._semantic_signal(query),
        }
        score = min(1.0, sum(signals.values()))

        return QueryRiskResult(score=score, signals=signals, ner_entities=ner_entities)

    def _semantic_signal(self, query: str) -> float:
        """
        S5: cosine similarity between the query embedding and each sensitive
        category centroid (HEALTH, POLITICS, ETHNIC, SEX, BELIEF).
        Linear ramp from SEMANTIC_NOISE_FLOOR to SEMANTIC_SATURATION.
        Gracefully returns 0.0 if the embedder is unavailable.
        """
        try:
            embedder = self._get_embedder()
            centroids = self._get_centroids(embedder)
            q_emb = embedder.embed_texts([query])  # (1, 384), normalized
            max_sim = max(float(q_emb[0] @ c) for c in centroids.values())
            return min(
                SEMANTIC_MAX_WEIGHT,
                max(0.0, max_sim - SEMANTIC_NOISE_FLOOR) * SEMANTIC_SCALE,
            )
        except Exception:
            return 0.0

    def _get_embedder(self):
        if self._embedder is None:
            from embeddings.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def _get_centroids(self, embedder) -> dict:
        if self._centroids is None:
            import numpy as np
            from countermeasure.sad_detector import SENSITIVE_TAXONOMY
            self._centroids = {}
            for category, sentences in SENSITIVE_TAXONOMY.items():
                embs = embedder.embed_texts(sentences)
                centroid = embs.mean(axis=0)
                norm = np.linalg.norm(centroid)
                self._centroids[category] = centroid / (norm + 1e-9)
        return self._centroids

    def flag_session(self, session_id: str, amount: float = SESSION_FLAG_INCREMENT) -> None:
        current = self.session_flags.get(session_id, 0.0)
        self.session_flags[session_id] = min(SESSION_MAX_WEIGHT, current + amount)
