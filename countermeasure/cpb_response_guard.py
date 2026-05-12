import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from countermeasure.cpb_models import ResponseGuardResult
from countermeasure.cpb_pii import PresidioPIIAnalyzer, PresidioPIIAnonymizer


class CPBResponseGuard:
    """
    CPB block 5b: Guardrails AI + Presidio.

    Policy:
      1. Presidio detects PII and computes leakage score f.
      2. If f = 0, response is reliable.
      3. If f > 0, fix by Presidio anonymization.
      4. If PII still remains after fix, optionally reask once.
      5. If PII still remains, block the response.

    Phi-3 indirect inference judging is not included in this class.
    """

    def __init__(self, analyzer: PresidioPIIAnalyzer, anonymizer: PresidioPIIAnonymizer):
        self.analyzer = analyzer
        self.anonymizer = anonymizer
        self.guardrails_available = self._guardrails_available()

    def guard(
        self,
        response: str,
        reask_callback=None,
    ) -> ResponseGuardResult:
        first = self.analyzer.analyze(response)
        if not first.findings:
            return ResponseGuardResult(
                response=response,
                leakage_score=0.0,
                decision="reliable",
                n_findings=0,
                n_replacements=0,
                reason="f=0",
            )

        fixed_response, replacements = self.anonymizer.anonymize_text(response, first.findings)
        after_fix = self.analyzer.analyze(fixed_response)
        if not after_fix.findings:
            return ResponseGuardResult(
                response=fixed_response,
                leakage_score=first.score,
                decision="fix",
                n_findings=len(first.findings),
                n_replacements=replacements,
                reason="Presidio fixed response PII",
            )

        if reask_callback is not None:
            regenerated = reask_callback()
            second = self.analyzer.analyze(regenerated)
            if not second.findings:
                return ResponseGuardResult(
                    response=regenerated,
                    leakage_score=first.score,
                    decision="reask",
                    n_findings=len(first.findings),
                    n_replacements=replacements,
                    reason="Regenerated response without PII",
                )

            fixed_regenerated, regen_replacements = self.anonymizer.anonymize_text(
                regenerated,
                second.findings,
            )
            after_regen_fix = self.analyzer.analyze(fixed_regenerated)
            if not after_regen_fix.findings:
                return ResponseGuardResult(
                    response=fixed_regenerated,
                    leakage_score=max(first.score, second.score),
                    decision="reask_fix",
                    n_findings=len(first.findings) + len(second.findings),
                    n_replacements=replacements + regen_replacements,
                    reason="Regenerated then fixed response PII",
                )

        return ResponseGuardResult(
            response="I cannot provide this answer because it may disclose personal information.",
            leakage_score=first.score,
            decision="exception",
            n_findings=len(first.findings),
            n_replacements=replacements,
            reason="PII remained after fix/reask",
        )

    @staticmethod
    def _guardrails_available() -> bool:
        try:
            import guardrails  # noqa: F401
            return True
        except ImportError:
            return False

