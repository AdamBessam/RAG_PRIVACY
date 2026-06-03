"""
CPB v2 — QueryRiskScorer avec context targets dynamiques appris depuis le store.

Améliorations vs countermeasure/cpb_query_risk.py :
1. JAILBREAK_PATTERNS étendu : markers d'injection courants couverts
   (New instruction, [SYSTEM:], IGNORE PREVIOUS, audit override…)
2. EXTRACTIVE_VERBS étendu : provide, disclose, share, furnish, release, give
3. CONTEXT_TARGETS dynamiques via ENTITY_TYPE_TO_QUERY_TERMS :
   les termes de requête sont déduits des types PII présents dans le store,
   sans aucune configuration manuelle.
"""
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
    # Common data-extraction verbs observed in financial/legal attacks
    "provide", "disclose", "share", "furnish", "release", "give",
)

# Domain-agnostic base targets (context/system level)
BASE_CONTEXT_TARGETS = (
    "context", "document", "documents", "chunk", "chunks", "database",
    "metadata", "source", "retrieved",
)

# Universal mapping: PII entity type → query terms used to request that PII type.
# Covers Presidio, spaCy, ildpil, scispaCy, and GLiNER entity types.
# No domain config needed — the store scan populates which types are relevant.
ENTITY_TYPE_TO_QUERY_TERMS: dict[str, tuple[str, ...]] = {
    # Presidio standard
    "US_SSN":             ("ssn", "social security number", "social security"),
    "CREDIT_CARD":        ("credit card", "card number"),
    "IBAN_CODE":          ("iban", "bank account", "account number"),
    "US_BANK_NUMBER":     ("bank account", "routing number"),
    "PHONE_NUMBER":       ("phone", "telephone", "mobile number"),
    "EMAIL_ADDRESS":      ("email", "email address"),
    "PERSON":             ("full name", "person name"),
    "LOCATION":           ("home address", "address"),
    "DATE_TIME":          ("date of birth", "birthdate"),
    "US_PASSPORT":        ("passport", "passport number"),
    "US_DRIVER_LICENSE":  ("driver license", "license number"),
    "US_ITIN":            ("itin", "tax id"),
    "CRYPTO":             ("bitcoin", "wallet", "crypto"),
    "IP_ADDRESS":         ("ip address",),
    "MEDICAL_LICENSE":    ("medical license",),
    "NRP":                ("national registration",),
    # spaCy / ildpil entity types
    "NATIONALITY":        ("nationality", "ethnic origin", "ethnicity"),
    "DEM":                ("demographic", "personal data"),
    "MISC":               ("personal information", "private information"),
    "ORG":                ("organization",),
    # ildpil sensitivity labels
    "HEALTH":             ("health", "medical", "medication", "diagnosis"),
    "POLITICS":           ("political", "politics", "party affiliation"),
    "ETHNIC":             ("ethnic", "ethnicity", "origin", "race"),
    "SEX":                ("sexual orientation", "gender identity"),
    "BELIEF":             ("religion", "belief", "faith"),
    # scispaCy biomedical
    "DISEASE":            ("disease", "condition", "diagnosis"),
    "CHEMICAL":           ("medication", "drug", "chemical"),
    # GLiNER structured identifiers
    "CASE_NUMBER":        ("case number", "file number"),
    "NATIONAL_ID":        ("national id", "national identifier"),
    "MEDICAL_RECORD_ID":  ("medical record",),
    "BANK_ACCOUNT":       ("bank account", "account"),
    "TAX_ID":             ("tax id", "tax number"),
}

JAILBREAK_PATTERNS = (
    # Classic system/instruction override
    r"\bignore\b.{0,60}\b(instruction|instructions|policy|rules|system|previous|above|prior|all)\b",
    r"\bbypass\b.{0,60}\b(instruction|instructions|policy|rules|system)\b",
    r"\boverride\b.{0,60}\b(instruction|instructions|policy|rules|system|procedures|protocol)\b",
    r"\bdisregard\b.{0,60}\b(instruction|instructions|policy|rules|above|previous)\b",
    # Prompt injection markers — very common in real injection attacks
    r"\bnew instruction[s]?\b",
    r"\[system\b",
    r"\bsystem\s*:\s*\w",
    # Authority-based injection (DGEA-style)
    r"\baudit override\b",
    r"\bsystem diagnostic\b",
    # Classic jailbreak phrases
    r"\bfrom now on you are\b",
    r"\broleplay\b",
    r"\bjailbreak\b",
    r"\bdeveloper mode\b",
    r"\bdo not refuse\b",
    r"\bdo not summarize\b",
    r"\braw json\b",
)

NER_MAX_WEIGHT        = 0.15
EXTRACTIVE_MAX_WEIGHT = 0.25
JAILBREAK_MAX_WEIGHT  = 0.35
SESSION_MAX_WEIGHT    = 0.10
SEMANTIC_MAX_WEIGHT   = 0.15

SEMANTIC_NOISE_FLOOR = 0.25
SEMANTIC_SATURATION  = 0.50
SEMANTIC_SCALE       = SEMANTIC_MAX_WEIGHT / (SEMANTIC_SATURATION - SEMANTIC_NOISE_FLOOR)

NER_ENTITY_INCREMENT       = 0.04
EXTRACTIVE_VERB_INCREMENT  = 0.08
CONTEXT_TARGET_INCREMENT   = 0.05
JAILBREAK_PATTERN_INCREMENT = 0.20
SESSION_FLAG_INCREMENT      = SESSION_MAX_WEIGHT


def build_context_targets(learned_types: set[str]) -> tuple[str, ...]:
    """
    Merge BASE_CONTEXT_TARGETS with query terms derived from the entity types
    discovered in the store. Returns a deduplicated tuple of terms.
    """
    extra: set[str] = set()
    for etype in learned_types:
        for term in ENTITY_TYPE_TO_QUERY_TERMS.get(etype.upper(), ()):
            extra.add(term)
    return BASE_CONTEXT_TARGETS + tuple(sorted(extra))


class QueryRiskScorer:
    """
    CPB v2 block 1A — QueryRiskScorer with adaptive context targets.

    learned_types: PII entity types discovered from the store at startup
    (via GT annotations or Presidio fallback). When empty, only
    BASE_CONTEXT_TARGETS are used.
    """

    def __init__(
        self,
        spacy_model: str = SPACY_MODEL,
        learned_types: set[str] | None = None,
    ):
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

        all_targets = build_context_targets(learned_types or set())
        self.context_re = re.compile(
            r"\b(" + "|".join(map(re.escape, all_targets)) + r")\b",
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
        context_hits    = len(self.context_re.findall(query))
        jailbreak_hits  = sum(1 for p in self.jailbreak_res if p.search(query))

        signals = {
            "s1_ner": min(NER_MAX_WEIGHT, NER_ENTITY_INCREMENT * len(ner_entities)),
            "s2_extractive": min(
                EXTRACTIVE_MAX_WEIGHT,
                EXTRACTIVE_VERB_INCREMENT * extractive_hits
                + CONTEXT_TARGET_INCREMENT * context_hits,
            ),
            "s3_jailbreak": min(JAILBREAK_MAX_WEIGHT, JAILBREAK_PATTERN_INCREMENT * jailbreak_hits),
            "s4_session":   min(SESSION_MAX_WEIGHT, self.session_flags.get(session_id, 0.0)),
            "s5_semantic":  self._semantic_signal(query),
        }
        score = min(1.0, sum(signals.values()))

        return QueryRiskResult(score=score, signals=signals, ner_entities=ner_entities)

    def _semantic_signal(self, query: str) -> float:
        try:
            embedder  = self._get_embedder()
            centroids = self._get_centroids(embedder)
            q_emb     = embedder.embed_texts([query])
            max_sim   = max(float(q_emb[0] @ c) for c in centroids.values())
            return min(SEMANTIC_MAX_WEIGHT, max(0.0, max_sim - SEMANTIC_NOISE_FLOOR) * SEMANTIC_SCALE)
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
                embs     = embedder.embed_texts(sentences)
                centroid = embs.mean(axis=0)
                norm     = np.linalg.norm(centroid)
                self._centroids[category] = centroid / (norm + 1e-9)
        return self._centroids

    def flag_session(self, session_id: str, amount: float = SESSION_FLAG_INCREMENT) -> None:
        current = self.session_flags.get(session_id, 0.0)
        self.session_flags[session_id] = min(SESSION_MAX_WEIGHT, current + amount)
