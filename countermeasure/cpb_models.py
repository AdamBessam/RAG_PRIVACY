from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class QueryRiskResult:
    score: float
    signals: dict[str, float]
    ner_entities: list[dict] = field(default_factory=list)


@dataclass
class PIIFinding:
    entity_type: str
    text: str
    start: int
    end: int
    score: float


@dataclass
class PIISensitivityResult:
    score: float
    findings: list[PIIFinding]


@dataclass
class BudgetDecision:
    chunk_id: str
    pii_score: float
    budget: float
    decision: str
    n_findings: int


@dataclass
class ResponseGuardResult:
    response: str
    leakage_score: float
    decision: str
    n_findings: int
    n_replacements: int
    reason: str = ""


@dataclass
class AuditEntry:
    query_risk: float
    max_pii_score: float
    min_budget: float
    leakage_score: float
    decision: str
    session_id: str
    chunk_decisions: list[BudgetDecision] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SADResult:
    sad_detected: bool
    attribute_categories: list[str]
    max_similarity: float
    confidence: float
    # "pass" | "mask" | "reask" | "block"
    decision: str
    response: str
    reasoning: str = ""
    # which filter stopped the cascade (1, 2, or 3; 0 = not run)
    filter_triggered: int = 0
    # per-category SBERT cosine similarity (populated by F2)
    sbert_category_scores: dict = field(default_factory=dict)

