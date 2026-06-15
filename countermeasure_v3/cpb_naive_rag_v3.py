"""
CPB v3 — Orchestrateur CPBNaiveRAGV3.

Pipeline (briques numérotées selon la spec):
  B0  CPBBootstrapV3          — auto-découverte domaine + taxonomie (once in __init__)
  B1  QueryRiskScorerV3       — scoring domain-aware + centroids dynamiques
  B2  BudgetGate              — suppression directe si r > 0.85 ou jailbreak (v1)
  B3  PresidioPIIAnalyzer     — v1, couches activées selon le domaine inféré
  B4  PresidioPIIAnonymizer   — placeholders stables (v1)
  B5  LLM.generate            — masked_query + safe_chunks
  B6  SADDetectorV3           — cascade F1→F2→F3 domain-aware + centroïdes dynamiques
  B7  CPBResponseGuard        — cascade 5 étapes (v1)

Imports v1 réutilisés sans modification :
  countermeasure.cpb_models   : AuditEntry
  countermeasure.cpb_pii      : BudgetGate, PresidioPIIAnalyzer, PresidioPIIAnonymizer
  countermeasure.cpb_response_guard : CPBResponseGuard
"""

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOP_K
from countermeasure.cpb_models import AuditEntry
from countermeasure.cpb_pii import BudgetGate, PresidioPIIAnalyzer, PresidioPIIAnonymizer
from countermeasure.cpb_response_guard import CPBResponseGuard
from countermeasure_v3.cpb_bootstrap_v3 import CPBBootstrapV3
from countermeasure_v3.cpb_query_risk_v3 import QueryRiskScorerV3
from countermeasure_v3.cpb_sad_detector_v3 import SADDetectorV3


class CPBNaiveRAGV3:
    """
    CPB v3 pipeline around NaiveRAG.

    The only difference at instantiation time vs v1 is the bootstrap step (B0)
    which auto-discovers the corpus domain and generates a domain-specific
    taxonomy. All subsequent per-query logic mirrors cpb_naive_rag.py exactly,
    except that B1 (QueryRiskScorerV3) and B6 (SADDetectorV3) use the dynamic
    centroids and categories produced by B0.
    """

    def __init__(
        self,
        naive_rag,
        session_id: str | None = None,
        language: str = "en",
        architecture_name: str = "cpb_naive_rag_v3",
    ):
        self.naive_rag = naive_rag
        self.store = naive_rag.store
        self.llm = naive_rag.llm
        self.architecture_name = architecture_name
        self.session_id = session_id or str(uuid4())

        # ── B0 Bootstrap (once) ───────────────────────────────────────────────
        bootstrap = CPBBootstrapV3(store=self.store)
        self.bootstrap_result = bootstrap.run()
        domain = self.bootstrap_result.domain

        # ── B1 QueryRiskScorerV3 ──────────────────────────────────────────────
        self.query_risk_scorer = QueryRiskScorerV3(
            centroids=self.bootstrap_result.centroids,
            learned_types=self.bootstrap_result.learned_types,
            domain=domain,
        )

        # ── B3 PresidioPIIAnalyzer — conditional layer activation ─────────────
        # scispaCy:  only for medical domain (or general)
        # GLiNER:    for legal, financial, general domains
        self.pii_analyzer = PresidioPIIAnalyzer(language=language)
        use_scispacy = domain in {"medical", "general"}
        use_gliner = domain in {"legal", "financial", "general"}
        if not use_scispacy:
            self.pii_analyzer.medical_recognizer.nlp = None
        if not use_gliner:
            self.pii_analyzer.gliner_recognizer.model = None

        # ── B4 PresidioPIIAnonymizer (v1) ─────────────────────────────────────
        self.pii_anonymizer = PresidioPIIAnonymizer()

        # ── B2 BudgetGate (v1) ────────────────────────────────────────────────
        self.budget_gate = BudgetGate()

        # ── B7 CPBResponseGuard (v1) ──────────────────────────────────────────
        self.response_guard = CPBResponseGuard(self.pii_analyzer, self.pii_anonymizer)

        # ── B6 SADDetectorV3 ──────────────────────────────────────────────────
        self.sad_detector = SADDetectorV3(
            dynamic_taxonomy=self.bootstrap_result.dynamic_taxonomy,
            centroids=self.bootstrap_result.centroids,
            domain=domain,
        )

        self.audit_log: list[AuditEntry] = []

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> dict:
        query_risk = self.query_risk_scorer.score(query, session_id=self.session_id)

        is_suppressed = (
            self.budget_gate.direct_suppression(query_risk.score)
            or self.budget_gate.jailbreak_suppression(query_risk.signals)
        )
        if is_suppressed:
            self.query_risk_scorer.flag_session(self.session_id)
            query_pii_result = self.pii_analyzer.analyze(query)
            masked_query, query_replacements = self.pii_anonymizer.anonymize_text(
                query, query_pii_result.findings,
            )
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

        query_pii_result = self.pii_analyzer.analyze(query)
        masked_query, query_replacements = self.pii_anonymizer.anonymize_text(
            query, query_pii_result.findings,
        )

        safe_chunks = []
        max_pii_score = 0.0

        for chunk in raw_chunks:
            pii_result = self.pii_analyzer.analyze(chunk.get("text", ""))
            max_pii_score = max(max_pii_score, pii_result.score)
            safe_chunks.append(self.pii_anonymizer.anonymize_chunk(
                chunk, pii_result, skip_types=set(),
            ))

        audit = AuditEntry(
            query_risk=query_risk.score,
            max_pii_score=max_pii_score,
            min_budget=1.0,
            leakage_score=0.0,
            decision="retrieval_masked",
            session_id=self.session_id,
            chunk_decisions=[],
        )
        self.audit_log.append(audit)

        return {
            "chunks": safe_chunks,
            "raw_chunks": raw_chunks,
            "query_risk": query_risk,
            "chunk_decisions": [],
            "audit": audit,
            "decision": "retrieval_masked",
            "masked_query": masked_query,
            "query_pii_score": query_pii_result.score,
            "query_pii_findings": query_pii_result.findings,
            "query_pii_findings_count": len(query_pii_result.findings),
            "query_pii_replacements": query_replacements,
        }

    # ── Generate ──────────────────────────────────────────────────────────────

    def generate(self, query: str, chunks: list[dict]):
        return self.naive_rag.generate(query, chunks)

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, query: str, top_k: int = TOP_K) -> dict:
        retrieval = self.retrieve(query, top_k=top_k)
        masked_query = retrieval.get("masked_query", query)

        if retrieval["decision"] == "direct_suppression":
            return self._result(
                query=query,
                retrieval=retrieval,
                response="I cannot process this request because it asks for sensitive context disclosure.",
                tokens_prompt=0,
                tokens_completion=0,
                tokens_total=0,
                cost_usd=0.0,
                response_guard_decision="direct_suppression",
            )

        chunks = retrieval["chunks"]
        llm_response = self.generate(masked_query, chunks)

        def reask():
            strengthened = (
                masked_query
                + "\n\nPrivacy instruction: answer only from the masked context. "
                + "Do not reveal names, identifiers, locations, dates, or any raw personal data. "
                + "Keep placeholders exactly as provided."
            )
            return self.generate(strengthened, chunks).response

        def sad_reask(attribute_category: str) -> str:
            constrained = (
                masked_query
                + f"\n\nPrivacy instruction: answer this question without revealing "
                + f"any {attribute_category} information about any individual. "
                + "Do not mention specific medical conditions, diagnoses, "
                + "political beliefs, ethnic origins, sexual orientation, "
                + "or religious beliefs."
            )
            return self.generate(constrained, chunks).response

        # B6 — SADDetectorV3 on raw LLM response (sees org/party names before Presidio)
        sad = self.sad_detector.detect(
            query=masked_query,
            chunks=chunks,
            response=llm_response.response,
            reask_callback=sad_reask,
        )

        # B7 — ResponseGuard cleans residual PII from SAD output
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

    # ── Result builder ────────────────────────────────────────────────────────

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
        br = self.bootstrap_result
        return {
            # ── Core
            "query":                    query,
            "chunks":                   retrieval.get("chunks", []),
            "raw_chunks":               retrieval.get("raw_chunks", []),
            "response":                 response,
            "architecture":             self.architecture_name,
            "llm":                      self.llm.name,
            # ── Tokens / cost
            "tokens_prompt":            tokens_prompt,
            "tokens_completion":        tokens_completion,
            "tokens_total":             tokens_total,
            "cost_usd":                 cost_usd,
            # ── B0 Bootstrap
            "cpb_v3_domain":            br.domain,
            "cpb_v3_domain_confidence": br.domain_confidence,
            "cpb_v3_categories":        br.dynamic_categories,
            "cpb_v3_used_fallback":     br.used_fallback,
            # ── B1 QueryRisk
            "cpb_query_risk":           retrieval["query_risk"].score,
            "cpb_query_risk_signals":   retrieval["query_risk"].signals,
            "cpb_ner_entities":         retrieval["query_risk"].ner_entities,
            "cpb_masked_query":         retrieval.get("masked_query", query),
            "cpb_query_pii_score":      retrieval.get("query_pii_score", 0.0),
            "cpb_query_pii_findings":   retrieval.get("query_pii_findings", []),
            "cpb_query_pii_findings_count": retrieval.get("query_pii_findings_count", 0),
            "cpb_query_pii_replacements":   retrieval.get("query_pii_replacements", 0),
            # ── B2/B3 Chunks
            "cpb_chunk_decisions":      retrieval.get("chunk_decisions", []),
            # ── B6 SAD
            "cpb_sad_detected":         sad_result.sad_detected if sad_result else False,
            "cpb_sad_decision":         sad_result.decision if sad_result else "pass",
            "cpb_sad_categories":       sad_result.attribute_categories if sad_result else [],
            "cpb_sad_confidence":       sad_result.confidence if sad_result else 0.0,
            "cpb_sad_filter":           sad_result.filter_triggered if sad_result else 0,
            "cpb_sad_result":           sad_result,
            # ── B7 ResponseGuard
            "cpb_response_guard_decision": response_guard_decision,
            "cpb_response_guard":          response_guard,
            # ── Audit
            "cpb_audit":                retrieval.get("audit"),
        }
