"""
CPB v2 — CPBNaiveRAG avec découverte automatique des types PII du store.

Améliorations vs countermeasure/cpb_naive_rag.py :
- _discover_pii_types() scanne ChromaDB au démarrage :
    Path A : lit les annotations GT (pii_entities dans les métadonnées)
    Path B : Presidio sur un échantillon de chunks (fallback sans annotations)
- QueryRiskScorer initialisé avec les types découverts → context targets adaptatifs
- Tous les autres blocs CPB sont identiques (importés depuis countermeasure/)
"""
import json
import re
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOP_K
from countermeasure.cpb_models import AuditEntry
from countermeasure.cpb_pii import BudgetGate, PresidioPIIAnalyzer, PresidioPIIAnonymizer
from countermeasure.cpb_response_guard import CPBResponseGuard
from contre_mesure_nv.cpb_query_risk import QueryRiskScorer
from contre_mesure_nv.sad_detector_nv import SADDetectorNV


# Mapping : type d'entité PII → catégorie SAD (pour enrichir la taxonomie SBERT)
ENTITY_TYPE_TO_SAD_CATEGORY: dict[str, str] = {
    # Financier
    "IBAN_CODE":         "FINANCIAL",
    "CREDIT_CARD":       "FINANCIAL",
    "US_BANK_NUMBER":    "FINANCIAL",
    "BANK_ACCOUNT":      "FINANCIAL",
    "TAX_ID":            "FINANCIAL",
    "US_ITIN":           "FINANCIAL",
    "CRYPTO":            "FINANCIAL",
    # Identité
    "US_SSN":            "IDENTITY",
    "US_PASSPORT":       "IDENTITY",
    "US_DRIVER_LICENSE": "IDENTITY",
    "NATIONAL_ID":       "IDENTITY",
    "NRP":               "IDENTITY",
    "PERSON":            "IDENTITY",
    # Contact
    "EMAIL_ADDRESS":     "CONTACT",
    "PHONE_NUMBER":      "CONTACT",
    "LOCATION":          "CONTACT",
    # Santé
    "DISEASE":           "HEALTH",
    "CHEMICAL":          "HEALTH",
    "MEDICAL_LICENSE":   "HEALTH",
    "MEDICAL_RECORD_ID": "HEALTH",
    # Technique
    "IP_ADDRESS":        "TECHNICAL",
}


class CPBNaiveRAG:
    """
    CPB v2 — identique à countermeasure.CPBNaiveRAG avec QueryRiskScorer adaptatif.

    Le QueryRiskScorer est configuré automatiquement selon les types PII
    présents dans le store (GT ou Presidio), sans paramètre domaine.
    """

    def __init__(
        self,
        naive_rag,
        session_id: str | None = None,
        language: str = "en",
        architecture_name: str = "cpb_naive_rag_nv",
    ):
        self.naive_rag = naive_rag
        self.store     = naive_rag.store
        self.llm       = naive_rag.llm
        self.architecture_name = architecture_name
        self.session_id = session_id or str(uuid4())

        self.pii_analyzer   = PresidioPIIAnalyzer(language=language)
        self.pii_anonymizer = PresidioPIIAnonymizer()
        self.budget_gate    = BudgetGate()
        self.response_guard = CPBResponseGuard(self.pii_analyzer, self.pii_anonymizer)

        learned_types = self._discover_pii_types()
        self.query_risk_scorer = QueryRiskScorer(learned_types=learned_types)

        sad_taxonomy      = self._build_sad_taxonomy()
        self.sad_detector = SADDetectorNV(extra_taxonomy=sad_taxonomy)

        self.audit_log: list[AuditEntry] = []

    def _discover_pii_types(self) -> set[str]:
        """
        Discover PII entity types present in the store.

        Path A — GT annotations: reads pii_entities from ChromaDB metadata.
            Works for any annotated dataset (financial, ildpil, etc.)
            without any domain-specific configuration.

        Path B — Presidio fallback: if no GT annotations found, runs
            Presidio on a sample of chunks to infer present PII types.
            Works for unannotated datasets.

        Returns an empty set if neither path yields results; QueryRiskScorer
        then uses BASE_CONTEXT_TARGETS only (universal core).
        """
        entity_types: set[str] = set()

        # Path A: GT annotations in ChromaDB metadata
        try:
            results = self.store.collection.get(limit=200, include=["metadatas"])
            for meta in results.get("metadatas", []):
                pii_raw = meta.get("pii_entities", "[]")
                entities = json.loads(pii_raw) if isinstance(pii_raw, str) else (pii_raw or [])
                for ent in entities:
                    t = ent.get("entity_type") or ent.get("label")
                    if t:
                        entity_types.add(t.upper())
        except Exception:
            pass

        # Path B: Presidio on a sample (fallback for unannotated datasets)
        if not entity_types:
            try:
                results = self.store.collection.get(limit=50, include=["documents"])
                for doc in results.get("documents", []):
                    if doc:
                        result = self.pii_analyzer.analyze(doc)
                        for finding in result.findings:
                            entity_types.add(finding.entity_type)
            except Exception:
                pass

        if entity_types:
            print(f"[CPB-NV] Types PII découverts dans le store : {sorted(entity_types)}")
        else:
            print("[CPB-NV] Aucun type PII découvert — core universel uniquement")

        return entity_types

    def _build_sad_taxonomy(self) -> dict[str, list[str]]:
        """
        Construit des phrases de taxonomie SAD depuis les chunks du store.

        Scan un échantillon de chunks, détecte les PII avec Presidio,
        extrait les phrases qui contiennent ces PII, les anonymise, puis
        les regroupe par catégorie SAD (FINANCIAL, IDENTITY, CONTACT...).

        Fonctionne que le dataset soit annoté (GT) ou non — Presidio est
        toujours utilisé pour localiser les spans exacts dans les phrases.
        """
        taxonomy: dict[str, list[str]] = {}
        MAX_SENTENCES_PER_CATEGORY = 15

        def add(entity_type: str, sentence: str) -> None:
            category = ENTITY_TYPE_TO_SAD_CATEGORY.get(entity_type.upper(), entity_type.upper())
            bucket = taxonomy.setdefault(category, [])
            if sentence not in bucket and len(bucket) < MAX_SENTENCES_PER_CATEGORY:
                bucket.append(sentence)

        try:
            results = self.store.collection.get(limit=100, include=["documents"])
            for doc in results.get("documents", []):
                if not doc:
                    continue
                sentences = [
                    s.strip()
                    for s in re.split(r"(?<=[.!?])\s+", doc)
                    if len(s.strip()) > 20
                ]
                for sent in sentences:
                    pii_result = self.pii_analyzer.analyze(sent)
                    if not pii_result.findings:
                        continue
                    anon_sent, _ = self.pii_anonymizer.anonymize_text(sent, pii_result.findings)
                    for finding in pii_result.findings:
                        add(finding.entity_type, anon_sent)
        except Exception:
            pass

        if taxonomy:
            n = sum(len(v) for v in taxonomy.values())
            print(f"[CPB-NV] Taxonomie SAD enrichie : {sorted(taxonomy.keys())} ({n} phrases)")
        else:
            print("[CPB-NV] Taxonomie SAD : base uniquement (aucune phrase extraite du store)")

        return taxonomy

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
                "chunks":                    [],
                "raw_chunks":                [],
                "query_risk":                query_risk,
                "chunk_decisions":           [],
                "audit":                     audit,
                "decision":                  "direct_suppression",
                "masked_query":              masked_query,
                "query_pii_score":           query_pii_result.score,
                "query_pii_findings_count":  len(query_pii_result.findings),
                "query_pii_replacements":    query_replacements,
            }

        raw_chunks = self.naive_rag.retrieve(query, top_k=top_k)

        query_pii_result = self.pii_analyzer.analyze(query)
        masked_query, query_replacements = self.pii_anonymizer.anonymize_text(
            query, query_pii_result.findings,
        )
        safe_chunks     = []
        chunk_decisions = []
        max_pii_score   = 0.0
        min_budget      = 1.0

        for chunk in raw_chunks:
            pii_result = self.pii_analyzer.analyze(chunk.get("text", ""))
            decision   = self.budget_gate.decide(
                chunk_id=str(chunk.get("chunk_id", "")),
                query_risk=query_risk.score,
                pii_result=pii_result,
            )
            chunk_decisions.append(decision)
            max_pii_score = max(max_pii_score, pii_result.score)
            min_budget    = min(min_budget, decision.budget)

            if decision.decision == "mask":
                s5 = query_risk.signals.get("s5_semantic", 0.0)
                skip_types = set() if s5 > 0.0 else {"ORGANIZATION"}
                safe_chunks.append(self.pii_anonymizer.anonymize_chunk(
                    chunk, pii_result, skip_types=skip_types,
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
            "chunks":                    safe_chunks,
            "raw_chunks":                raw_chunks,
            "query_risk":                query_risk,
            "chunk_decisions":           chunk_decisions,
            "audit":                     audit,
            "decision":                  "retrieval_masked",
            "masked_query":              masked_query,
            "query_pii_score":           query_pii_result.score,
            "query_pii_findings":        query_pii_result.findings,
            "query_pii_findings_count":  len(query_pii_result.findings),
            "query_pii_replacements":    query_replacements,
        }

    def generate(self, query: str, chunks: list[dict]):
        return self.naive_rag.generate(query, chunks)

    def run(self, query: str, top_k: int = TOP_K) -> dict:
        retrieval    = self.retrieve(query, top_k=top_k)
        masked_query = retrieval.get("masked_query", query)

        if retrieval["decision"] == "direct_suppression":
            return self._result(
                query=query,
                retrieval=retrieval,
                response="I cannot process this request because it asks for sensitive context disclosure.",
                tokens_prompt=0, tokens_completion=0, tokens_total=0, cost_usd=0.0,
                response_guard_decision="direct_suppression",
            )

        chunks = retrieval["chunks"]
        if not chunks:
            return self._result(
                query=query,
                retrieval=retrieval,
                response=(
                    "I cannot provide details because the retrieved context is too sensitive "
                    "after privacy filtering."
                ),
                tokens_prompt=0, tokens_completion=0, tokens_total=0, cost_usd=0.0,
                response_guard_decision="all_chunks_suppressed",
            )

        llm_response = self.generate(masked_query, chunks)

        def reask():
            return self.generate(
                masked_query
                + "\n\nPrivacy instruction: answer only from the masked context. "
                + "Do not reveal names, identifiers, locations, dates, or any raw personal data. "
                + "Keep placeholders exactly as provided.",
                chunks,
            ).response

        def sad_reask(attribute_category: str) -> str:
            return self.generate(
                masked_query
                + f"\n\nPrivacy instruction: answer this question without revealing "
                + f"any {attribute_category} information about any individual. "
                + "Do not mention specific medical conditions, diagnoses, "
                + "political beliefs, ethnic origins, sexual orientation, "
                + "or religious beliefs.",
                chunks,
            ).response

        sad     = self.sad_detector.detect(
            query=masked_query,
            chunks=chunks,
            response=llm_response.response,
            reask_callback=sad_reask,
        )
        guarded = self.response_guard.guard(response=sad.response, reask_callback=reask)

        audit = retrieval["audit"]
        audit.leakage_score = guarded.leakage_score
        audit.decision      = sad.decision if sad.sad_detected else guarded.decision

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
            "query":                        query,
            "chunks":                       retrieval.get("chunks", []),
            "raw_chunks":                   retrieval.get("raw_chunks", []),
            "response":                     response,
            "architecture":                 self.architecture_name,
            "llm":                          self.llm.name,
            "tokens_prompt":                tokens_prompt,
            "tokens_completion":            tokens_completion,
            "tokens_total":                 tokens_total,
            "cost_usd":                     cost_usd,
            "cpb_query_risk":               retrieval["query_risk"].score,
            "cpb_query_risk_signals":       retrieval["query_risk"].signals,
            "cpb_ner_entities":             retrieval["query_risk"].ner_entities,
            "cpb_masked_query":             retrieval.get("masked_query", query),
            "cpb_query_pii_score":          retrieval.get("query_pii_score", 0.0),
            "cpb_query_pii_findings":       retrieval.get("query_pii_findings", []),
            "cpb_query_pii_findings_count": retrieval.get("query_pii_findings_count", 0),
            "cpb_query_pii_replacements":   retrieval.get("query_pii_replacements", 0),
            "cpb_chunk_decisions":          retrieval.get("chunk_decisions", []),
            "cpb_response_guard_decision":  response_guard_decision,
            "cpb_response_guard":           response_guard,
            "cpb_sad_detected":             sad_result.sad_detected if sad_result else False,
            "cpb_sad_decision":             sad_result.decision if sad_result else "pass",
            "cpb_sad_categories":           sad_result.attribute_categories if sad_result else [],
            "cpb_sad_confidence":           sad_result.confidence if sad_result else 0.0,
            "cpb_sad_filter":               sad_result.filter_triggered if sad_result else 0,
            "cpb_sad_result":               sad_result,
            "cpb_audit":                    retrieval.get("audit"),
        }
