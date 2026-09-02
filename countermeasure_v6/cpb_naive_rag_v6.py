"""
CPB v6 — Orchestrateur CPBNaiveRAGV6.

Différence architecturale majeure avec v1-v5 : le masquage PII ne s'applique
JAMAIS à la requête ni aux chunks avant génération. Le LLM voit toujours la
requête et les chunks BRUTS (meilleure utilité) ; seule la RÉPONSE générée est
ensuite masquée. B6 (SAD detector) s'exécute APRÈS ce masquage PII de la
réponse. Il n'y a PAS de brique B7 (ResponseGuard) — B6 est le dernier filet.

Pipeline (briques numérotées) :
  B0  CPBBootstrapV6        — domaine + taxonomie dynamique (once, __init__)
  B1  QueryRiskScorerV6     — score de risque de la requête (jamais masquée)
  B2  BudgetGate            — suppression directe si risque trop élevé / jailbreak
      (si suppression : refus direct, PAS de retrieval, PAS de génération)
  --  retrieve + generate   — requête et chunks BRUTS, aucun masquage
  B3  PresidioPIIAnalyzer   — analyse PII de la RÉPONSE générée (pas des chunks)
  B4  PresidioPIIAnonymizer — masquage SÉLECTIF de la réponse (poids + hints de
      domaine B0 + combinaisons ré-identifiantes, cf. countermeasure_v5) :
      une entité n'est masquée que si elle est jugée sensible, isolément ou en
      combinaison ; les identifiants forts sont toujours masqués (filet).
  B6  SADDetectorV6         — cascade F1→F2→F3 sur la réponse DÉJÀ masquée par
      B3/B4 (synthesis → mask → block si une divulgation d'attribut persiste)

Module 100% autonome : countermeasure_v6 n'importe RIEN de countermeasure/,
countermeasure_v3/, countermeasure_v4/ ni countermeasure_v5/. Toutes les
briques nécessaires (modèles, PII, bootstrap, query risk, SAD) sont dupliquées
localement dans countermeasure_v6/.
"""

import json
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOP_K
from countermeasure_v6.cpb_ablation_v6 import AblationConfigV6
from countermeasure_v6.cpb_bootstrap_v6 import BootstrapResult, CPBBootstrapV6
from countermeasure_v6.cpb_models_v6 import (
    AuditEntry,
    PIISensitivityResult,
    QueryRiskResult,
    SADResult,
)
from countermeasure_v6.cpb_pii_v6 import (
    DEFAULT_ENTITY_WEIGHT,
    ENTITY_WEIGHTS,
    BudgetGate,
    PresidioPIIAnalyzer,
    PresidioPIIAnonymizer,
)
from countermeasure_v6.cpb_query_risk_v6 import QueryRiskScorerV6
from countermeasure_v6.cpb_sad_detector_v6 import SADDetectorV6

# Table d'activation des couches PII optionnelles selon le domaine détecté par
# B0 (vocabulaire fixe de nvidia/domain-classifier, ou repli Llama non couvert
# par cette table -> les deux couches sont activées par défaut, sûr pour la
# détection PII).
DOMAIN_LAYER_HINTS: dict[str, set[str]] = {
    "health": {"scispacy"},
    "science": {"scispacy"},
    "law_and_government": {"gliner"},
    "finance": {"gliner"},
    "business_and_industrial": {"gliner"},
}

# Types Presidio proposés au LLM pour composer les combinaisons ré-identifiantes.
_AVAILABLE_TYPES = [
    "PERSON", "LOCATION", "ORGANIZATION", "DATE_TIME", "NATIONALITY", "NRP", "URL",
    "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE",
    "CREDIT_CARD", "IBAN_CODE", "US_BANK_NUMBER", "IP_ADDRESS", "MEDICAL_LICENSE",
    "CASE_NUMBER", "FILE_NUMBER", "NATIONAL_ID", "MEDICAL_RECORD_ID", "BANK_ACCOUNT",
    "TAX_ID", "OPAQUE_ID",
]


class CPBNaiveRAGV6:
    """
    CPB v6 pipeline around NaiveRAG/HybridRAG.

    La requête et les chunks ne sont JAMAIS masqués. Seule la réponse générée
    est masquée (sélectivement, par poids + hints de domaine + combinaisons
    ré-identifiantes), puis vérifiée par B6. Pas de B7.
    """

    def __init__(
        self,
        naive_rag,
        session_id: str | None = None,
        language: str = "en",
        architecture_name: str = "cpb_naive_rag_v6",
        ablation: AblationConfigV6 | None = None,
        mask_min_weight: float = 0.5,          # Signal 1 : masque si poids(type) >= seuil
        use_domain_hints: bool = True,         # Signal 2 : ajoute les types sensibles du domaine (B0)
        use_llm_combos: bool = True,           # Signal 3 : combinaisons ré-identifiantes (Llama, post-B0)
        always_mask_min_weight: float = 0.9,   # filet de sécurité : identifiants forts toujours masqués
    ):
        self.naive_rag = naive_rag
        self.store = naive_rag.store
        self.llm = naive_rag.llm
        self.architecture_name = architecture_name
        self.session_id = session_id or str(uuid4())
        self.ablation = ablation or AblationConfigV6()

        self.mask_min_weight = mask_min_weight
        self.use_domain_hints = use_domain_hints
        self.always_mask_min_weight = always_mask_min_weight
        # Bascule à chaud : baseline "anonymise TOUT" pour comparaison (aucune
        # entité épargnée dans la réponse), sans reconstruire la contre-mesure.
        self.mask_all = False

        # ── B0 Bootstrap (once) ───────────────────────────────────────────────
        if self.ablation.b0_bootstrap:
            bootstrap = CPBBootstrapV6(store=self.store)
            self.bootstrap_result = bootstrap.run()
        else:
            self.bootstrap_result = BootstrapResult(
                domain="general",
                domain_confidence=0.0,
                domain_source="none",
                learned_types=set(),
                dynamic_categories=[],
                dynamic_taxonomy={},
                category_hints={},
                centroids={},
                used_fallback=True,
            )
        domain = self.bootstrap_result.domain

        # ── B1 QueryRiskScorerV6 ────────────────────────────────────────────────
        self.query_risk_scorer = QueryRiskScorerV6(
            centroids=self.bootstrap_result.centroids,
            learned_types=self.bootstrap_result.learned_types,
            domain=domain,
        )

        # ── B2 BudgetGate ────────────────────────────────────────────────────────
        self.budget_gate = BudgetGate()

        # ── B3 PresidioPIIAnalyzer — couches conditionnelles selon le domaine ───
        self.pii_analyzer = PresidioPIIAnalyzer(language=language)
        layers = DOMAIN_LAYER_HINTS.get(domain, {"scispacy", "gliner"})
        if "scispacy" not in layers:
            self.pii_analyzer.medical_recognizer.nlp = None
        if "gliner" not in layers:
            self.pii_analyzer.gliner_recognizer.model = None

        # ── B4 PresidioPIIAnonymizer ─────────────────────────────────────────────
        self.pii_anonymizer = PresidioPIIAnonymizer()

        # Signal 2 : ensemble des types Presidio que B0 a jugés sensibles POUR
        # CE domaine (union des category_hints). Vide si B0 off/fallback -> on
        # retombe proprement sur le seul Signal 1 (poids générique).
        hints = self.bootstrap_result.category_hints or {}
        self.domain_sensitive_types: set[str] = set()
        for types in hints.values():
            self.domain_sensitive_types |= {str(t).upper() for t in types}

        # Signal 3 : combinaisons de types ré-identifiantes propres au domaine
        # (générées par Llama une fois, après B0).
        self.risky_combos: list[frozenset[str]] = (
            self._discover_risky_combinations() if use_llm_combos else []
        )

        # ── B6 SADDetectorV6 (s'exécute APRÈS le masquage B3/B4) ───────────────
        self.sad_detector = SADDetectorV6(
            dynamic_taxonomy=self.bootstrap_result.dynamic_taxonomy,
            centroids=self.bootstrap_result.centroids,
            domain=domain,
        )

        self.audit_log: list[AuditEntry] = []

        print(
            f"CPB v6: domain={domain} (fallback={self.bootstrap_result.used_fallback}), "
            f"mask_min_weight={self.mask_min_weight}, "
            f"domain_sensitive_types={sorted(self.domain_sensitive_types) or '(none)'}, "
            f"risky_combinations={[sorted(c) for c in self.risky_combos] or '(none)'}"
        )

    # ── Découverte des combinaisons risquées (LLM local, post-bootstrap) ──────

    def _discover_risky_combinations(self) -> list[frozenset[str]]:
        domain = self.bootstrap_result.domain or "general"
        prompt = (
            "You are a privacy / re-identification expert. A document corpus is in "
            f"the '{domain}' domain.\n"
            "An individual can be RE-IDENTIFIED when a COMBINATION of entity types "
            "appears together in the same passage, even when each type alone is not "
            "identifying. Which entity types are re-identifying TOGETHER depends on "
            "the domain.\n"
            "From the available entity types below, list the COMBINATIONS of types "
            "that, appearing TOGETHER in one passage, would single out or re-identify "
            "a specific individual in THIS domain. Also list any single type that "
            "identifies a person on its own (as a combination of one).\n"
            f"Available entity types: {', '.join(_AVAILABLE_TYPES)}\n"
            "Give between 3 and 10 combinations. Use ONLY the type names above.\n"
            "Respond in valid JSON only.\n"
            'Example: {"risky_combinations": [["PERSON","CASE_NUMBER"], '
            '["PERSON","LOCATION","DATE_TIME"], ["US_SSN"]]}'
        )
        try:
            raw = self._llama_json(prompt)
            parsed = self._parse_json(raw)
            allowed = set(_AVAILABLE_TYPES)
            combos: list[frozenset[str]] = []
            for item in parsed.get("risky_combinations", []):
                if not isinstance(item, (list, tuple)):
                    continue
                types = {str(t).upper().strip() for t in item if t}
                types &= allowed
                if types:
                    combos.append(frozenset(types))
            seen, unique = set(), []
            for c in combos:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)
            if unique:
                return unique
            print("CPB v6: LLM n'a produit aucune combinaison valide -> Signal 3 désactivé.")
        except Exception as exc:
            print(f"CPB v6: génération des combos échouée ({exc!r}) -> Signal 3 désactivé.")
        return []

    def _llama_json(self, prompt: str) -> str:
        """Appelle Llama en forçant une sortie JSON (comme B0)."""
        client = getattr(self.llm, "client", None)
        if client is not None:
            resp = client.chat(
                model=self.llm.name,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0, "num_predict": 512},
            )
            if hasattr(resp, "message"):
                return resp.message.content or ""
            if isinstance(resp, dict):
                return resp.get("message", {}).get("content", "")
            return str(resp)
        return self.llm.generate(prompt).response

    @staticmethod
    def _parse_json(raw: str) -> dict:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object in LLM response")
        return json.loads(raw[start:end])

    # ── Décision de sensibilité d'une entité (masquage de la RÉPONSE) ─────────

    def _is_sensitive(self, entity_type: str) -> bool:
        etype = (entity_type or "").upper()
        weight = ENTITY_WEIGHTS.get(etype, DEFAULT_ENTITY_WEIGHT)
        if weight >= self.mask_min_weight:
            return True                                    # Signal 1
        if self.use_domain_hints and etype in self.domain_sensitive_types:
            return True                                    # Signal 2
        return False

    def _always_mask(self, entity_type: str) -> bool:
        w = ENTITY_WEIGHTS.get((entity_type or "").upper(), DEFAULT_ENTITY_WEIGHT)
        return w >= self.always_mask_min_weight

    def _anonymize_response(self, text: str, pii_result: PIISensitivityResult) -> tuple[str, int]:
        """B4 appliqué à la réponse : masque seulement les entités jugées
        sensibles (seuil, hints de domaine, ou combinaison ré-identifiante
        entièrement présente), + identifiants forts en filet de sécurité."""
        findings = pii_result.findings
        if not findings:
            return text, 0

        if self.mask_all:
            findings_to_mask = findings  # baseline "anonymise tout", pour comparaison
        else:
            present = {(f.entity_type or "").upper() for f in findings}
            to_mask: set[str] = {t for t in present if self._always_mask(t)}
            to_mask |= {t for t in present if self._is_sensitive(t)}
            for combo in self.risky_combos:
                if combo <= present:                       # tous les membres présents
                    to_mask |= set(combo)                   # on casse toute la combinaison
            findings_to_mask = [f for f in findings if (f.entity_type or "").upper() in to_mask]

        if not findings_to_mask:
            return text, 0
        return self.pii_anonymizer.anonymize_text(text, findings_to_mask)

    # ── Retrieve (requête + chunks BRUTS, aucun masquage) ─────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> dict:
        if self.ablation.b1_query_risk:
            query_risk = self.query_risk_scorer.score(query, session_id=self.session_id)
        else:
            query_risk = QueryRiskResult(score=0.0, signals={}, ner_entities=[])

        is_suppressed = self.ablation.b2_budget_gate and (
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
                "raw_chunks": [],
                "query_risk": query_risk,
                "audit": audit,
                "decision": "direct_suppression",
            }

        raw_chunks = self.naive_rag.retrieve(query, top_k=top_k)
        audit = AuditEntry(
            query_risk=query_risk.score,
            max_pii_score=0.0,
            min_budget=1.0,
            leakage_score=0.0,
            decision="retrieved",
            session_id=self.session_id,
        )
        self.audit_log.append(audit)
        return {
            "raw_chunks": raw_chunks,
            "query_risk": query_risk,
            "audit": audit,
            "decision": "retrieved",
        }

    # ── Generate (requête + chunks BRUTS) ──────────────────────────────────────

    def generate(self, query: str, chunks: list[dict]):
        return self.naive_rag.generate(query, chunks)

    # ── Run ─────────────────────────────────────────────────────────────────────

    def run(self, query: str, top_k: int = TOP_K) -> dict:
        retrieval = self.retrieve(query, top_k=top_k)

        if retrieval["decision"] == "direct_suppression":
            return self._result(
                query=query,
                retrieval=retrieval,
                response="I cannot process this request because it asks for sensitive context disclosure.",
                response_before_masking=None,
                tokens_prompt=0,
                tokens_completion=0,
                tokens_total=0,
                cost_usd=0.0,
                pii_result=PIISensitivityResult(score=0.0, findings=[]),
                n_replacements=0,
            )

        raw_chunks = retrieval["raw_chunks"]
        llm_response = self.generate(query, raw_chunks)   # requête + chunks BRUTS
        raw_response = llm_response.response

        # B3 — analyse PII de la réponse générée
        if self.ablation.b3_pii_analyzer:
            pii_result = self.pii_analyzer.analyze(raw_response)
        else:
            pii_result = PIISensitivityResult(score=0.0, findings=[])

        # B4 — masquage sélectif de la réponse
        if self.ablation.b4_pii_anonymizer:
            masked_response, n_replacements = self._anonymize_response(raw_response, pii_result)
        else:
            masked_response, n_replacements = raw_response, 0

        # B6 — SAD detector APRÈS le masquage B3/B4
        def reask():
            strengthened = (
                query
                + "\n\nPrivacy instruction: answer only from the retrieved context. "
                + "Do not reveal names, identifiers, locations, dates, or any raw personal data."
            )
            return self.generate(strengthened, raw_chunks).response

        if self.ablation.b6_sad_detector:
            sad = self.sad_detector.detect(
                query=query,
                chunks=raw_chunks,
                response=masked_response,
                reask_callback=reask,
            )
        else:
            sad = SADResult(
                sad_detected=False,
                attribute_categories=[],
                max_similarity=0.0,
                confidence=0.0,
                decision="pass",
                response=masked_response,
                reasoning="B6 disabled (ablation)",
                filter_triggered=0,
            )

        audit = retrieval["audit"]
        audit.max_pii_score = pii_result.score
        audit.decision = sad.decision if sad.sad_detected else "pass"

        return self._result(
            query=query,
            retrieval=retrieval,
            response=sad.response,
            response_before_masking=raw_response,
            tokens_prompt=llm_response.tokens_prompt,
            tokens_completion=llm_response.tokens_completion,
            tokens_total=llm_response.tokens_total,
            cost_usd=llm_response.cost_usd,
            pii_result=pii_result,
            n_replacements=n_replacements,
            sad_result=sad,
        )

    # ── Result builder ───────────────────────────────────────────────────────────

    def _result(
        self,
        query: str,
        retrieval: dict,
        response: str,
        response_before_masking: str | None,
        tokens_prompt: int,
        tokens_completion: int,
        tokens_total: int,
        cost_usd: float,
        pii_result: PIISensitivityResult,
        n_replacements: int,
        sad_result: SADResult | None = None,
    ) -> dict:
        br = self.bootstrap_result
        return {
            # ── Core
            "query":                    query,
            "chunks":                   retrieval.get("raw_chunks", []),   # jamais masqués en v6
            "raw_chunks":               retrieval.get("raw_chunks", []),
            "response":                 response,
            "response_before_masking":  response_before_masking,
            "architecture":             self.architecture_name,
            "llm":                      self.llm.name,
            "ablation":                 self.ablation.name,
            # ── Tokens / cost
            "tokens_prompt":            tokens_prompt,
            "tokens_completion":        tokens_completion,
            "tokens_total":             tokens_total,
            "cost_usd":                 cost_usd,
            # ── B0 Bootstrap
            "cpb_v6_domain":            br.domain,
            "cpb_v6_domain_confidence": br.domain_confidence,
            "cpb_v6_domain_source":     br.domain_source,
            "cpb_v6_categories":        br.dynamic_categories,
            "cpb_v6_category_hints":    {k: sorted(v) for k, v in (br.category_hints or {}).items()},
            "cpb_v6_used_fallback":     br.used_fallback,
            "cpb_v6_risky_combinations": [sorted(c) for c in self.risky_combos],
            # ── B1 QueryRisk (requête jamais masquée)
            "cpb_query_risk":           retrieval["query_risk"].score,
            "cpb_query_risk_signals":   retrieval["query_risk"].signals,
            "cpb_ner_entities":         retrieval["query_risk"].ner_entities,
            "cpb_masked_query":         query,   # inchangée par design en v6
            # ── B3/B4 PII de la réponse
            "cpb_response_pii_score":       pii_result.score,
            "cpb_response_pii_findings_count": len(pii_result.findings),
            "cpb_response_n_replacements":  n_replacements,
            # ── B6 SAD (post-masquage)
            "cpb_sad_detected":         sad_result.sad_detected if sad_result else False,
            "cpb_sad_decision":         sad_result.decision if sad_result else "pass",
            "cpb_sad_categories":       sad_result.attribute_categories if sad_result else [],
            "cpb_sad_confidence":       sad_result.confidence if sad_result else 0.0,
            "cpb_sad_filter":           sad_result.filter_triggered if sad_result else 0,
            "cpb_sad_result":           sad_result,
            # ── Audit
            "cpb_audit":                retrieval.get("audit"),
        }
