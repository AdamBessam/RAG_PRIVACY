"""
CPB v6 — Switches d'ablation leave-one-out.

Adapté de countermeasure_v4/cpb_ablation.py à l'architecture v6 : pas de
brique B7 (supprimée), et B3/B4 s'appliquent à la RÉPONSE (post-génération)
au lieu de la requête/des chunks.

  B0  bootstrap (domaine + taxonomie, once)
  B1  query risk scorer
  B2  budget gate (suppression directe si risque trop élevé / jailbreak)
  B3  PII analyzer   — sur la réponse générée
  B4  PII anonymizer — masquage sélectif (poids + hints domaine + combos) de la réponse
  B6  SAD detector   — sur la réponse déjà masquée par B3/B4

  Comme en v4 : si B0 est désactivé, B6 dégénère (plus de taxonomie/centroïdes
  dynamiques) → __post_init__ force aussi B6 off pour éviter un état ambigu.
  B1+B2 et B3+B4 restent chacun un seul "bloc" testé ensemble (mêmes raisons
  qu'en v4 : rien d'autre ne consomme leur sortie intermédiaire).
"""

from dataclasses import dataclass


@dataclass
class AblationConfigV6:
    name: str = "full_pipeline_v6"
    b0_bootstrap: bool = True
    b1_query_risk: bool = True
    b2_budget_gate: bool = True
    b3_pii_analyzer: bool = True
    b4_pii_anonymizer: bool = True
    b6_sad_detector: bool = True

    def __post_init__(self):
        if not self.b0_bootstrap:
            self.b6_sad_detector = False


# ── Variantes leave-one-out ────────────────────────────────────────────────────
VARIANTS_V6: list[AblationConfigV6] = [
    AblationConfigV6(name="b0_off",    b0_bootstrap=False),                          # force aussi b6 off
    AblationConfigV6(name="b1_b2_off", b1_query_risk=False, b2_budget_gate=False),
    AblationConfigV6(name="b3_b4_off", b3_pii_analyzer=False, b4_pii_anonymizer=False),
    AblationConfigV6(name="b6_off",    b6_sad_detector=False),
]
