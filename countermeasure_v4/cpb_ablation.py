"""
CPB v4 — Leave-one-out ablation switches.

Each AblationConfig flags one or more layers "off" while leaving the rest of
the pipeline (cpb_naive_rag_v4.CPBNaiveRAGV4) intact, so each block's
contribution to the privacy/utility metrics can be isolated by rerunning the
same attack queries through each variant.

B5 (LLM.generate) is never ablated: it is the core RAG step.

Dependency notes (traced from cpb_naive_rag_v4.py — identical control flow to
cpb_naive_rag_v3.py, only B0's domain source differs):
  - B1's score/signals are consumed only by B2's suppression checks; B3's
    findings are consumed only by B4's anonymizer calls. Disabling either
    member of either pair alone produces byte-identical pipeline output to
    disabling both (nothing else reads query_risk.score or pii_result before
    it reaches B2/B4), so B1+B2 and B3+B4 are always toggled together below
    — testing them separately would waste a full 300-query run for no new
    information.
  - B6 still runs with no domain taxonomy, but degrades to F1 regex + a
    Phi-3 prompt with no candidate categories (F2 SBERT proximity always
    empty). To avoid that ambiguous half-working state, "B0 off" forces B6
    off too (enforced in __post_init__). "B6 off" alone (B0 left on) stays a
    separate, clean variant that isolates B6's own contribution.
"""

from dataclasses import dataclass


@dataclass
class AblationConfig:
    name: str = "full_pipeline"
    b0_bootstrap: bool = True
    b1_query_risk: bool = True
    b2_budget_gate: bool = True
    b3_pii_analyzer: bool = True
    b4_pii_anonymizer: bool = True
    b6_sad_detector: bool = True
    b7_response_guard: bool = True

    def __post_init__(self):
        if not self.b0_bootstrap:
            self.b6_sad_detector = False


# ── The 5 leave-one-out variants used by the ablation study ───────────────────
VARIANTS: list[AblationConfig] = [
    AblationConfig(name="b0_off",    b0_bootstrap=False),                          # forces b6 off too
    AblationConfig(name="b1_b2_off", b1_query_risk=False, b2_budget_gate=False),
    AblationConfig(name="b3_b4_off", b3_pii_analyzer=False, b4_pii_anonymizer=False),
    AblationConfig(name="b6_off",    b6_sad_detector=False),
    AblationConfig(name="b7_off",    b7_response_guard=False),
]


# ── The 4 cumulative ("knock-out from B0 outward") variants ────────────────────
# Instead of leave-one-out, each variant disables one MORE layer than the
# previous one, in pipeline order, starting at B0:
#     cum_b0            : B0
#     cum_b0_b1b2       : B0 + B1B2
#     cum_b0_b1b2_b3b4  : B0 + B1B2 + B3B4
#     cum_all           : B0 + B1B2 + B3B4 + B7  (== naive RAG, all CPB off)
# B1+B2 and B3+B4 are each a single layer (same reasoning as above). B6 has no
# dedicated cumulative step: disabling B0 already forces b6 off via
# __post_init__, so B6 stays off across every variant here (nothing to isolate).
# B5 (LLM.generate) is never ablated.
CUMULATIVE_VARIANTS: list[AblationConfig] = [
    AblationConfig(
        name="cum_b0",
        b0_bootstrap=False,                                                        # -> b6 forced off
    ),
    AblationConfig(
        name="cum_b0_b1b2",
        b0_bootstrap=False,
        b1_query_risk=False, b2_budget_gate=False,
    ),
    AblationConfig(
        name="cum_b0_b1b2_b3b4",
        b0_bootstrap=False,
        b1_query_risk=False, b2_budget_gate=False,
        b3_pii_analyzer=False, b4_pii_anonymizer=False,
    ),
    AblationConfig(
        name="cum_all",
        b0_bootstrap=False,
        b1_query_risk=False, b2_budget_gate=False,
        b3_pii_analyzer=False, b4_pii_anonymizer=False,
        b7_response_guard=False,                                                   # b6 already off
    ),
]
