"""
CPB v6 — Brique B1 : QueryRiskScorerV6.

Fusion autonome de countermeasure/cpb_query_risk.py (base) et
countermeasure_v4/cpb_query_risk_v4.py (override domain-aware) en une seule
classe, pour que countermeasure_v6 n'importe RIEN des autres dossiers
countermeasure*. Logique et poids strictement identiques à v3/v4 :

  S1: spaCy NER sur la requête
  S2: verbes extractifs / regex de ciblage de contexte
  S3: regex de jailbreak
  S4: risque de session multi-tour
  S5: proximité sémantique aux centroïdes de catégories sensibles (SBERT,
      injectés depuis BootstrapResult — pas de reconstruction paresseuse)

En v6, ce score sert UNIQUEMENT à B2 (BudgetGate) pour décider une suppression
directe — la requête elle-même n'est jamais masquée (contrairement à v1-v5).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import spacy

from config import SPACY_MODEL
from countermeasure_v6.cpb_models_v6 import QueryRiskResult

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

NER_MAX_WEIGHT = 0.15
EXTRACTIVE_MAX_WEIGHT = 0.25
JAILBREAK_MAX_WEIGHT = 0.35
SESSION_MAX_WEIGHT = 0.10
SEMANTIC_MAX_WEIGHT = 0.15

SEMANTIC_NOISE_FLOOR = 0.25
SEMANTIC_SATURATION = 0.50
SEMANTIC_SCALE = SEMANTIC_MAX_WEIGHT / (SEMANTIC_SATURATION - SEMANTIC_NOISE_FLOOR)

NER_ENTITY_INCREMENT = 0.04
EXTRACTIVE_VERB_INCREMENT = 0.08
CONTEXT_TARGET_INCREMENT = 0.05
JAILBREAK_PATTERN_INCREMENT = 0.20
SESSION_FLAG_INCREMENT = SESSION_MAX_WEIGHT


class QueryRiskScorerV6:
    """
    CPB v6 Block B1 : query risk scorer domain-aware.

    Signals: S1 NER, S2 extractif/contexte, S3 jailbreak, S4 session, S5
    proximité sémantique (centroïdes injectés depuis B0, pas de fallback
    codé en dur).
    """

    def __init__(
        self,
        centroids: dict,       # category -> np.ndarray(384,), depuis BootstrapResult
        learned_types: set,    # types PII découverts en 0a, depuis BootstrapResult
        domain: str = "general",
        spacy_model: str = SPACY_MODEL,
    ):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model], check=False)
            self.nlp = spacy.load(spacy_model)

        self.domain = domain
        self._centroids = centroids  # injectés — jamais reconstruits paresseusement
        self._embedder = None

        extractive_re_source = EXTRACTIVE_VERBS
        self.extractive_re = re.compile(
            r"\b(" + "|".join(map(re.escape, extractive_re_source)) + r")\b",
            re.IGNORECASE,
        )
        extended = set(CONTEXT_TARGETS) | {t.lower() for t in learned_types}
        self.context_re = re.compile(
            r"\b(" + "|".join(map(re.escape, sorted(extended))) + r")\b",
            re.IGNORECASE,
        )
        self.jailbreak_res = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]
        self.session_flags: dict[str, float] = {}

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
        if not self._centroids:
            return 0.0
        try:
            embedder = self._get_embedder()
            q_emb = embedder.embed_texts([query])
            max_sim = max(float(q_emb[0] @ c) for c in self._centroids.values())
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

    def flag_session(self, session_id: str, amount: float = SESSION_FLAG_INCREMENT) -> None:
        current = self.session_flags.get(session_id, 0.0)
        self.session_flags[session_id] = min(SESSION_MAX_WEIGHT, current + amount)
