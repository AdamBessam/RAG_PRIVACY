import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOP_K
from countermeasure.cpb_models import AuditEntry
from countermeasure.cpb_pii import BudgetGate, PresidioPIIAnalyzer, PresidioPIIAnonymizer
from countermeasure.cpb_query_risk import QueryRiskScorer
from countermeasure.cpb_response_guard import CPBResponseGuard
from countermeasure.sad_detector import SADDetector


class CPBNaiveRAG:
    """
    CPB architecture around NaiveRAG, without the Phi-3 judge step.

    Implemented blocks:
      1A Query Risk Scorer
      1B NaiveRAG retrieval
      2  Presidio PII Sensitivity Scorer
      3  Budget Gate
      4  Presidio Anonymizer with stable placeholders
      5  LLM generation
      5b Guardrails AI + Presidio response guard
      AuditEntry
    """

    def __init__(
        self,
        naive_rag,
        session_id: str | None = None,
        language: str = "en",
        architecture_name: str = "cpb_naive_rag",
    ):
        self.naive_rag = naive_rag
        self.store = naive_rag.store
        self.llm = naive_rag.llm
        self.architecture_name = architecture_name
        self.session_id = session_id or str(uuid4())

        self.query_risk_scorer = QueryRiskScorer()
        self.pii_analyzer = PresidioPIIAnalyzer(language=language)
        self.pii_anonymizer = PresidioPIIAnonymizer()
        self.budget_gate = BudgetGate()
        self.response_guard = CPBResponseGuard(self.pii_analyzer, self.pii_anonymizer)
        self.sad_detector = SADDetector()
        self.audit_log: list[AuditEntry] = []

    def retrieve(self, query: str, top_k: int = TOP_K) -> dict:
        query_risk = self.query_risk_scorer.score(query, session_id=self.session_id)
        query_pii_result = self.pii_analyzer.analyze(query)
        masked_query, query_replacements = self.pii_anonymizer.anonymize_text(
            query,
            query_pii_result.findings,
        )

        is_suppressed = (
            self.budget_gate.direct_suppression(query_risk.score)
            or self.budget_gate.jailbreak_suppression(query_risk.signals)
        )
        if is_suppressed:
            self.query_risk_scorer.flag_session(self.session_id)
            audit = AuditEntry(
                query_risk=query_risk.score,
                max_pii_score=0.0,
                min_budget=0.0,
                leakage_score=0.0,
                decision="direct_suppression",
                session_id=self.session_id,
            )
            self.audit_log.append(audit)
            return {
                "chunks": [],
                "raw_chunks": [],
                "query_risk": query_risk,
                "chunk_decisions": [],
                "audit": audit,
                "decision": "direct_suppression",
                "masked_query": masked_query,
                "query_pii_score": query_pii_result.score,
                "query_pii_findings_count": len(query_pii_result.findings),
                "query_pii_replacements": query_replacements,
            }

        raw_chunks = self.naive_rag.retrieve(query, top_k=top_k)
        safe_chunks = []
        chunk_decisions = []
        max_pii_score = 0.0
        min_budget = 1.0

        for chunk in raw_chunks:
            pii_result = self.pii_analyzer.analyze(chunk.get("text", ""))
            decision = self.budget_gate.decide(
                chunk_id=str(chunk.get("chunk_id", "")),
                query_risk=query_risk.score,
                pii_result=pii_result,
            )
            chunk_decisions.append(decision)
            max_pii_score = max(max_pii_score, pii_result.score)
            min_budget = min(min_budget, decision.budget)

            if decision.decision == "mask":
                # When the query targets a sensitive category (S5 > 0), also anonymize
                # ORGANIZATION to prevent LLM re-identification via org names
                # (e.g. "PKK" → "Kurdistan" → "Kurdish").
                # Otherwise keep ORG visible so SAD can detect political affiliation.
                s5 = query_risk.signals.get("s5_semantic", 0.0)
                skip_types = set() if s5 > 0.0 else {"ORGANIZATION"}
                safe_chunks.append(self.pii_anonymizer.anonymize_chunk(
                    chunk, pii_result, skip_types=skip_types
                ))

        audit = AuditEntry(
            query_risk=query_risk.score,
            max_pii_score=max_pii_score,
            min_budget=min_budget,
            leakage_score=0.0,
            decision="retrieval_masked",
            session_id=self.session_id,
            chunk_decisions=chunk_decisions,
        )
        self.audit_log.append(audit)

        return {
            "chunks": safe_chunks,
            "raw_chunks": raw_chunks,
            "query_risk": query_risk,
            "chunk_decisions": chunk_decisions,
            "audit": audit,
            "decision": "retrieval_masked",
            "masked_query": masked_query,
            "query_pii_score": query_pii_result.score,
            "query_pii_findings": query_pii_result.findings,
            "query_pii_findings_count": len(query_pii_result.findings),
            "query_pii_replacements": query_replacements,
        }

    def generate(self, query: str, chunks: list[dict]):
        return self.naive_rag.generate(query, chunks)

    def run(self, query: str, top_k: int = TOP_K) -> dict:
        retrieval = self.retrieve(query, top_k=top_k)
        masked_query = retrieval.get("masked_query", query)

        if retrieval["decision"] == "direct_suppression":
            response = "I cannot process this request because it asks for sensitive context disclosure."
            return self._result(
                query=query,
                retrieval=retrieval,
                response=response,
                tokens_prompt=0,
                tokens_completion=0,
                tokens_total=0,
                cost_usd=0.0,
                response_guard_decision="direct_suppression",
            )

        chunks = retrieval["chunks"]
        if not chunks:
            response = (
                "I cannot provide details because the retrieved context is too sensitive "
                "after privacy filtering."
            )
            return self._result(
                query=query,
                retrieval=retrieval,
                response=response,
                tokens_prompt=0,
                tokens_completion=0,
                tokens_total=0,
                cost_usd=0.0,
                response_guard_decision="all_chunks_suppressed",
            )

        llm_response = self.generate(masked_query, chunks)

        def reask():
            strengthened_query = (
                masked_query
                + "\n\nPrivacy instruction: answer only from the masked context. "
                + "Do not reveal names, identifiers, locations, dates, or any raw personal data. "
                + "Keep placeholders exactly as provided."
            )
            return self.generate(strengthened_query, chunks).response

        # Block 6 — SAD runs on the raw LLM response so it sees org/party names
        # before Presidio anonymizes them (e.g. "DKP" not "[ORGANIZATION_2]").
        def sad_reask(attribute_category: str) -> str:
            constrained_query = (
                masked_query
                + f"\n\nPrivacy instruction: answer this question without revealing "
                + f"any {attribute_category} information about any individual. "
                + "Do not mention specific medical conditions, diagnoses, "
                + "political beliefs, ethnic origins, sexual orientation, "
                + "or religious beliefs."
            )
            return self.generate(constrained_query, chunks).response

        sad = self.sad_detector.detect(
            query=masked_query,
            chunks=chunks,
            response=llm_response.response,
            reask_callback=sad_reask,
        )

        # Block 5b — Response Guard runs on SAD's output to clean remaining PII.
        guarded = self.response_guard.guard(
            response=sad.response,
            reask_callback=reask,
        )

        audit = retrieval["audit"]
        audit.leakage_score = guarded.leakage_score
        audit.decision = sad.decision if sad.sad_detected else guarded.decision

        return self._result(
            query=query,
            retrieval=retrieval,
            response=sad.response,
            tokens_prompt=llm_response.tokens_prompt,
            tokens_completion=llm_response.tokens_completion,
            tokens_total=llm_response.tokens_total,
            cost_usd=llm_response.cost_usd,
            response_guard_decision=guarded.decision,
            response_guard=guarded,
            sad_result=sad,
        )

    def _result(
        self,
        query: str,
        retrieval: dict,
        response: str,
        tokens_prompt: int,
        tokens_completion: int,
        tokens_total: int,
        cost_usd: float,
        response_guard_decision: str,
        response_guard=None,
        sad_result=None,
    ) -> dict:
        return {
            "query": query,
            "chunks": retrieval.get("chunks", []),
            "raw_chunks": retrieval.get("raw_chunks", []),
            "response": response,
            "architecture": self.architecture_name,
            "llm": self.llm.name,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_total": tokens_total,
            "cost_usd": cost_usd,
            "cpb_query_risk": retrieval["query_risk"].score,
            "cpb_query_risk_signals": retrieval["query_risk"].signals,
            "cpb_ner_entities": retrieval["query_risk"].ner_entities,
            "cpb_masked_query": retrieval.get("masked_query", query),
            "cpb_query_pii_score": retrieval.get("query_pii_score", 0.0),
            "cpb_query_pii_findings": retrieval.get("query_pii_findings", []),
            "cpb_query_pii_findings_count": retrieval.get("query_pii_findings_count", 0),
            "cpb_query_pii_replacements": retrieval.get("query_pii_replacements", 0),
            "cpb_chunk_decisions": retrieval.get("chunk_decisions", []),
            "cpb_response_guard_decision": response_guard_decision,
            "cpb_response_guard": response_guard,
            "cpb_sad_detected": sad_result.sad_detected if sad_result else False,
            "cpb_sad_decision": sad_result.decision if sad_result else "pass",
            "cpb_sad_categories": sad_result.attribute_categories if sad_result else [],
            "cpb_sad_confidence": sad_result.confidence if sad_result else 0.0,
            "cpb_sad_filter": sad_result.filter_triggered if sad_result else 0,
            "cpb_sad_result": sad_result,
            "cpb_audit": retrieval.get("audit"),
        }
