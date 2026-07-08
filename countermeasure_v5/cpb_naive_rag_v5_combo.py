"""
CPB v5 — masquage par COMBINAISONS RISQUÉES, domain-aware (isolé, rollback-safe).

Idée : une entité seule est souvent inoffensive ; c'est l'ASSEMBLAGE de plusieurs
types d'entités dans un même passage qui ré-identifie une personne — et QUELLE
combinaison est risquée DÉPEND DU DOMAINE.
  • domaine juridique/CEDH : {PERSON, CASE_NUMBER} ou {PERSON, LOCATION, DATE_TIME}
  • domaine médical        : {PERSON, DISEASE, DATE_TIME}
  • domaine financier      : {PERSON, ORGANIZATION, BANK_ACCOUNT} ...

Fonctionnement :
  1. Après le bootstrap B0 (qui a DÉJÀ détecté le domaine), on demande au LLM local
     la liste des combinaisons de types ré-identifiantes POUR CE DOMAINE
     (_discover_risky_combinations). B0 n'est PAS modifié : on lit juste son
     bootstrap_result.domain et on génère les combos ici.
  2. Par chunk : pour chaque combinaison risquée dont TOUS les membres sont présents,
     on masque TOUS ses membres (« casser » la combinaison). Les entités qui ne
     participent à aucune combinaison présente sont GARDÉES (utilité préservée).
  3. Filet de sécurité : les identifiants forts (poids >= ALWAYS_MASK_MIN_WEIGHT :
     SSN, e-mail, passeport, n° de dossier...) sont TOUJOURS masqués, même seuls.

Isolation / rollback : cette classe vit uniquement dans countermeasure_v5/, hérite
de CPBNaiveRAGV5 et n'override que _anonymize_chunk (+ un helper de génération de
combos). Rien dans v4, v5 ou B0 n'est modifié. En cas de mauvais résultats :
revenir à CPBNaiveRAGV5 (seuil) ou CPBNaiveRAGV4 (tout masqué). Si la génération de
combos échoue (LLM off), on retombe automatiquement sur le comportement v5.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from countermeasure_v4.cpb_naive_rag_v4 import CPBNaiveRAGV4
from countermeasure_v5.cpb_naive_rag_v5 import CPBNaiveRAGV5

# Types Presidio proposés au LLM pour composer les combinaisons.
_AVAILABLE_TYPES = [
    "PERSON", "LOCATION", "ORGANIZATION", "DATE_TIME", "NATIONALITY", "NRP", "URL",
    "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE",
    "CREDIT_CARD", "IBAN_CODE", "US_BANK_NUMBER", "IP_ADDRESS", "MEDICAL_LICENSE",
    "CASE_NUMBER", "FILE_NUMBER", "NATIONAL_ID", "MEDICAL_RECORD_ID", "BANK_ACCOUNT",
    "TAX_ID", "OPAQUE_ID",
]


class CPBNaiveRAGV5Combo(CPBNaiveRAGV5):
    """v5 + masquage par combinaisons ré-identifiantes propres au domaine."""

    def __init__(
        self,
        naive_rag,
        use_llm_combos: bool = True,      # False → pas d'appel LLM, fallback v5
        always_mask_min_weight: float = 0.9,  # identifiants forts toujours masqués
        **kwargs,
    ):
        super().__init__(naive_rag, **kwargs)  # exécute B0, remplit bootstrap_result
        self.always_mask_min_weight = always_mask_min_weight
        self.risky_combos: list[frozenset[str]] = (
            self._discover_risky_combinations() if use_llm_combos else []
        )
        pretty = [sorted(c) for c in self.risky_combos] or "(none → fallback v5)"
        print(
            f"CPB v5 combo masking: domain="
            f"{getattr(self.bootstrap_result, 'domain', '?')}, "
            f"risky_combinations={pretty}"
        )

    # ── Découverte des combinaisons risquées (LLM local, post-bootstrap) ──────
    def _discover_risky_combinations(self) -> list[frozenset[str]]:
        domain = getattr(self.bootstrap_result, "domain", "general") or "general"
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
            raw = self.llm.generate(prompt).response
            parsed = self._parse_json(raw)
            allowed = set(_AVAILABLE_TYPES)
            combos: list[frozenset[str]] = []
            for item in parsed.get("risky_combinations", []):
                if not isinstance(item, (list, tuple)):
                    continue
                types = {str(t).upper().strip() for t in item if t}
                types &= allowed  # écarte tout type inventé hors référentiel
                if types:
                    combos.append(frozenset(types))
            # dé-duplique en conservant l'ordre
            seen, unique = set(), []
            for c in combos:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)
            if unique:
                return unique
            print("CPB v5 combo: LLM n'a produit aucune combinaison valide → fallback v5.")
        except Exception as exc:
            print(f"CPB v5 combo: génération des combos échouée ({exc!r}) → fallback v5.")
        return []

    @staticmethod
    def _parse_json(raw: str) -> dict:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object in LLM response")
        return json.loads(raw[start:end])

    def _always_mask(self, entity_type: str) -> bool:
        from countermeasure.cpb_pii import DEFAULT_ENTITY_WEIGHT, ENTITY_WEIGHTS
        w = ENTITY_WEIGHTS.get((entity_type or "").upper(), DEFAULT_ENTITY_WEIGHT)
        return w >= self.always_mask_min_weight

    # ── Override : masquer pour casser les combinaisons risquées présentes ────
    def _anonymize_chunk(self, chunk: dict, pii_result):
        # Ablation B4 off → mask-all baseline v4.
        if not self.ablation.b4_pii_anonymizer:
            return super(CPBNaiveRAGV5, self)._anonymize_chunk(chunk, pii_result)
        # Pas de combos (LLM off/échec) → comportement v5 standard.
        if not self.risky_combos:
            return super()._anonymize_chunk(chunk, pii_result)

        findings = pii_result.findings
        present = {(f.entity_type or "").upper() for f in findings}

        # Types à masquer = membres de toute combinaison entièrement présente
        # + identifiants forts (filet de sécurité), toujours masqués.
        to_mask: set[str] = {t for t in present if self._always_mask(t)}
        for combo in self.risky_combos:
            if combo <= present:          # tous les membres du combo présents
                to_mask |= set(combo)     # on casse toute la combinaison

        skip_types = {
            f.entity_type
            for f in findings
            if (f.entity_type or "").upper() not in to_mask
        }
        return self.pii_anonymizer.anonymize_chunk(
            chunk, pii_result, skip_types=skip_types,
        )
