"""
CPB v5 — Masquage sélectif domain-aware (optimisation de B3/B4).

Motivation (étude d'ablation CEDH) : B3+B4 (masquage PII) est le bloc le plus
coûteux en utilité (ΔQS +0.10) pour un gain de privacy marginal. En masquant
TOUTES les entités détectées, il sur-caviarde et abîme la réponse.

CPB v5 hérite intégralement de CPBNaiveRAGV4 et ne change QUE la décision de
masquage des chunks (_anonymize_chunk) : au lieu de masquer toutes les entités,
il ne masque une entité que si elle est jugée "sensible" selon deux signaux
GÉNÉRIQUES (les mêmes dans tous les domaines) :

  Signal 1 — poids de sensibilité du TYPE (ENTITY_WEIGHTS, cpb_pii.py) :
             on masque si poids(type) >= mask_min_weight (un seul curseur).
  Signal 2 — pertinence DOMAINE (category_hints découverts par B0) :
             on masque aussi si le type appartient à l'ensemble sensible que
             B0 a généré pour CE corpus. C'est ce qui rend la décision
             domain-aware SANS règle codée en dur : la liste des types
             sensibles change automatiquement selon le domaine détecté.

Ce qui change par domaine n'est donc pas le code ni les curseurs, mais les
`category_hints` fournis par B0 → même formule, décisions différentes.

Tout le reste (B0/B1/B2/B3/B6/B7, retrieve/generate/run, ablation) est
strictement identique à cpb_naive_rag_v4.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from countermeasure_v4.cpb_naive_rag_v4 import CPBNaiveRAGV4


class CPBNaiveRAGV5(CPBNaiveRAGV4):
    """CPB v4 + masquage sélectif domain-aware sur les chunks (B4)."""

    def __init__(
        self,
        naive_rag,
        mask_min_weight: float = 0.5,   # curseur générique : masque si poids(type) >= seuil
        use_domain_hints: bool = True,  # ajoute les types sensibles du domaine (B0)
        **kwargs,
    ):
        super().__init__(naive_rag, **kwargs)
        self.mask_min_weight = mask_min_weight
        self.use_domain_hints = use_domain_hints

        # Signal 2 : ensemble des types Presidio que B0 a jugés sensibles POUR CE
        # domaine (union des category_hints). Vide si B0 off / fallback -> on
        # retombe proprement sur le seul Signal 1 (poids générique).
        hints = getattr(self.bootstrap_result, "category_hints", None) or {}
        self.domain_sensitive_types: set[str] = set()
        for types in hints.values():
            self.domain_sensitive_types |= {str(t).upper() for t in types}

        print(
            f"CPB v5 selective masking: mask_min_weight={self.mask_min_weight}, "
            f"domain_sensitive_types={sorted(self.domain_sensitive_types) or '(none)'}"
        )

    # ── Décision de sensibilité d'une entité ─────────────────────────────────
    def _is_sensitive(self, entity_type: str) -> bool:
        from countermeasure.cpb_pii import DEFAULT_ENTITY_WEIGHT, ENTITY_WEIGHTS

        etype = (entity_type or "").upper()
        weight = ENTITY_WEIGHTS.get(etype, DEFAULT_ENTITY_WEIGHT)
        if weight >= self.mask_min_weight:
            return True                                   # Signal 1
        if self.use_domain_hints and etype in self.domain_sensitive_types:
            return True                                   # Signal 2
        return False

    # ── Override : masquer sélectivement au lieu de tout masquer ─────────────
    def _anonymize_chunk(self, chunk: dict, pii_result):
        # Respecte l'ablation B4 (si désactivée, comportement v4 inchangé).
        if not self.ablation.b4_pii_anonymizer:
            return super()._anonymize_chunk(chunk, pii_result)

        # skip_types = tous les types détectés jugés NON sensibles -> non masqués.
        skip_types = {
            f.entity_type
            for f in pii_result.findings
            if not self._is_sensitive(f.entity_type)
        }
        return self.pii_anonymizer.anonymize_chunk(
            chunk, pii_result, skip_types=skip_types,
        )
