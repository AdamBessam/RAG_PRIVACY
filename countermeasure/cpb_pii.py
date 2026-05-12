import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import spacy

from config import SPACY_MODEL
from countermeasure.cpb_models import BudgetDecision, PIIFinding, PIISensitivityResult


# PII sensitivity weights.
#
# These weights instantiate the CPB architecture shown in the diagram:
# ID-like direct identifiers receive the maximum weight (1.0), PERSON is high
# but not maximal (0.6), LOCATION is moderate (0.4), and DATE_TIME is lower
# (0.3). Presidio-specific identifier types are mapped to the ID=1.0 family.
# Unknown Presidio types default to 0.3 to remain conservative without treating
# every weak detector hit as a direct identifier.
DEFAULT_ENTITY_WEIGHT = 0.3
ENTITY_WEIGHTS = {
    "CREDIT_CARD": 1.0,
    "CRYPTO": 1.0,
    "EMAIL_ADDRESS": 1.0,
    "IBAN_CODE": 1.0,
    "IP_ADDRESS": 1.0,
    "NRP": 1.0,
    "PHONE_NUMBER": 1.0,
    "US_BANK_NUMBER": 1.0,
    "US_DRIVER_LICENSE": 1.0,
    "US_ITIN": 1.0,
    "US_PASSPORT": 1.0,
    "US_SSN": 1.0,
    "PERSON": 0.6,
    "ORGANIZATION": 0.5,
    "LOCATION": 0.4,
    "DATE_TIME": 0.3,
    "URL": 0.3,
    "MEDICAL_LICENSE": 1.0,
}

# Budget thresholds from the CPB architecture:
# - r > 0.85 means direct query suppression + session flag.
# - b < 0.4 means the chunk is removed.
# - otherwise every PII is still systematically masked before the LLM sees it.
DEFAULT_DIRECT_QUERY_THRESHOLD = 0.85
DEFAULT_CHUNK_SUPPRESS_THRESHOLD = 0.4

SPACY_TO_PRESIDIO_ENTITY = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
}


class PresidioPIIAnalyzer:
    """CPB block 2: Presidio Analyzer + weighted PII sensitivity p."""

    def __init__(self, language: str = "en", use_spacy_fallback: bool = True):
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError as exc:
            raise ImportError(
                "Presidio is required for CPB. Install presidio-analyzer "
                "and presidio-anonymizer."
            ) from exc

        self.language = language
        self.analyzer = AnalyzerEngine()
        self.use_spacy_fallback = use_spacy_fallback
        self.nlp = None
        if self.use_spacy_fallback:
            try:
                self.nlp = spacy.load(SPACY_MODEL)
            except OSError:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", SPACY_MODEL], check=False)
                self.nlp = spacy.load(SPACY_MODEL)

    def analyze(self, text: str) -> PIISensitivityResult:
        text = text or ""
        results = self.analyzer.analyze(text=text, language=self.language)

        findings: list[PIIFinding] = [
            PIIFinding(
                entity_type=r.entity_type,
                text=text[r.start:r.end],
                start=r.start,
                end=r.end,
                score=float(r.score),
            )
            for r in results
            if r.start < r.end
        ]

        if self.use_spacy_fallback and self.nlp is not None and text:
            findings = self._augment_with_spacy(text, findings)

        return PIISensitivityResult(
            score=self._sensitivity_score(findings),
            findings=findings,
        )

    def _augment_with_spacy(self, text: str, findings: list[PIIFinding]) -> list[PIIFinding]:
        """
        Add spaCy entities when Presidio misses them.

        This fallback is conservative:
        - it only maps labels aligned with CPB weighting (PERSON/ORG/LOC/DATE/TIME)
        - it skips spans already covered by Presidio findings
        """
        doc = self.nlp(text)
        augmented = list(findings)
        for ent in doc.ents:
            mapped_type = SPACY_TO_PRESIDIO_ENTITY.get(ent.label_)
            if not mapped_type:
                continue
            if len(ent.text.strip()) < 2:
                continue
            if self._is_already_covered(ent.start_char, ent.end_char, augmented):
                continue
            augmented.append(
                PIIFinding(
                    entity_type=mapped_type,
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=0.75,
                )
            )
        return augmented

    @staticmethod
    def _is_already_covered(start: int, end: int, findings: list[PIIFinding]) -> bool:
        for f in findings:
            overlaps = not (end <= f.start or start >= f.end)
            if overlaps:
                return True
        return False

    @staticmethod
    def _sensitivity_score(findings: list[PIIFinding]) -> float:
        if not findings:
            return 0.0

        numerator = 0.0
        denominator = 0.0
        for finding in findings:
            weight = ENTITY_WEIGHTS.get(finding.entity_type, DEFAULT_ENTITY_WEIGHT)
            numerator += finding.score * weight
            denominator += weight

        return max(0.0, min(1.0, numerator / (denominator or 1.0)))


class BudgetGate:
    """CPB block 3: b = 1 - (r * p)."""

    def __init__(
        self,
        chunk_suppress_threshold: float = DEFAULT_CHUNK_SUPPRESS_THRESHOLD,
        direct_query_threshold: float = DEFAULT_DIRECT_QUERY_THRESHOLD,
    ):
        self.chunk_suppress_threshold = chunk_suppress_threshold
        self.direct_query_threshold = direct_query_threshold

    def direct_suppression(self, query_risk: float) -> bool:
        return query_risk > self.direct_query_threshold

    def jailbreak_suppression(self, signals: dict) -> bool:
        # Any confirmed jailbreak pattern is sufficient for immediate suppression,
        # regardless of the total risk score.
        return signals.get("s3_jailbreak", 0.0) > 0.0

    def decide(self, chunk_id: str, query_risk: float, pii_result: PIISensitivityResult) -> BudgetDecision:
        budget = 1.0 - (query_risk * pii_result.score)
        decision = "suppress" if budget < self.chunk_suppress_threshold else "mask"

        return BudgetDecision(
            chunk_id=chunk_id,
            pii_score=pii_result.score,
            budget=budget,
            decision=decision,
            n_findings=len(pii_result.findings),
        )


class PresidioPIIAnonymizer:
    """
    CPB block 4.

    Presidio provides the detected spans; this class applies stable
    deterministic placeholders: PERSON -> [PERSON_1], LOCATION -> [LOCATION_1].
    """

    def __init__(self):
        try:
            from presidio_anonymizer import AnonymizerEngine
        except ImportError as exc:
            raise ImportError(
                "Presidio Anonymizer is required for CPB. Install presidio-anonymizer."
            ) from exc

        self.engine = AnonymizerEngine()
        self.mapping: dict[tuple[str, str], str] = {}
        self.counters: dict[str, int] = {}

    def anonymize_text(self, text: str, findings: list[PIIFinding]) -> tuple[str, int]:
        text = text or ""
        replacements = 0
        anonymized = text

        for finding in sorted(findings, key=lambda f: (f.start, f.end), reverse=True):
            raw = text[finding.start:finding.end]
            if not raw.strip():
                continue

            placeholder = self._placeholder(finding.entity_type, raw)
            anonymized = anonymized[:finding.start] + placeholder + anonymized[finding.end:]
            replacements += 1

        return anonymized, replacements

    def anonymize_chunk(
        self,
        chunk: dict,
        pii_result: PIISensitivityResult,
        skip_types: set[str] | None = None,
    ) -> dict:
        findings = pii_result.findings
        if skip_types:
            findings = [f for f in findings if f.entity_type not in skip_types]
        masked_text, replacements = self.anonymize_text(chunk.get("text", ""), findings)

        masked = dict(chunk)
        masked["text"] = masked_text
        masked["cpb_masked"] = True
        masked["cpb_n_replacements"] = replacements
        masked["cpb_pii_score"] = pii_result.score
        masked["cpb_findings"] = [
            {
                "entity_type": f.entity_type,
                "text": f.text,
                "start": f.start,
                "end": f.end,
                "score": f.score,
            }
            for f in pii_result.findings
        ]
        return masked

    def _placeholder(self, entity_type: str, raw_text: str) -> str:
        normalized_type = entity_type.upper()
        key = (normalized_type, raw_text.casefold())
        if key not in self.mapping:
            self.counters[normalized_type] = self.counters.get(normalized_type, 0) + 1
            self.mapping[key] = f"[{normalized_type}_{self.counters[normalized_type]}]"
        return self.mapping[key]
