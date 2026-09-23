"""Constant optical efficiency eta = a0 — weakest rung.

The crudest model assumes the collector runs at a fixed efficiency
equal to its optical (zero-loss) efficiency a0, ignoring all thermal
losses:

    eta = a0.

This is the y-intercept of the Hottel-Whillier-Bliss efficiency curve —
the efficiency a perfectly insulated collector would reach when the
inlet temperature equals ambient (zero reduced temperature).  Real
collectors lose heat as the inlet fluid gets hotter than ambient, so a
constant over-/under-predicts depending on the operating point and
captures none of the operating-condition dependence.

a0 is pre-fit on the v2 train as the mean efficiency (NOT the optical
intercept, which the loss-bearing rungs recover separately).  Test
rmse ~ 0.025; r^2 ~ 0 (no better than predicting the mean) — the
ladder's lower bound.

LAW_CONSTANTS — frozen, pre-fit on v2 train
-------------------------------------------
- A0 = 0.5280   (mean gross-area thermal efficiency)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["G_W_m2", "T_in_C", "T_amb_C"]
PAPER_REF = "summary_solar_collector.md"
EQUATION_LOC = (
    "Constant-efficiency baseline eta = a0 (the optical-efficiency "
    "intercept of the Hottel-Whillier-Bliss curve, ISO 9806; here a0 = "
    "mean efficiency on v2 train).  Ignores all thermal losses."
)

LAW_CONSTANTS = {
    "A0": 0.5280,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A0: float = 0.5280) -> np.ndarray:
    """eta = A0 (constant)."""
    G = np.asarray(X[:, 0], dtype=float)
    return np.full_like(G, A0)
