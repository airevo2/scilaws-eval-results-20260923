"""Empirical power law (Aubry 2023 form), coefficients fit on train — middle rung.

Aubry et al. (2023, GRL 50:e2022GL102633) Eq. 1 establishes that the
eruption-column top height scales with the mass eruption rate as a
two-coefficient power law:

    H_top = A * MER^B          (MER in kg/s, H_top in km above vent).

Aubry's published fit on the full 130-event IVESPA corpus gives
A = 0.345, B = 0.226.  Because the v2 split is MER-range OOD and those
published constants were fit on data INCLUDING the high-MER test
events, this baseline does NOT use the paper constants (that would leak
the test set).  Instead it RE-FITS A, B on the v2 OOD train (the 98
lowest-MER events) by log-log OLS, giving A = 0.668, B = 0.175 — a
clean held-out Type I baseline that "saw" only small/moderate
eruptions.

The fitted exponent (0.175) is shallower than both Aubry's full-corpus
value (0.226) and the Morton-Taylor-Turner theoretical 1/4: the
small/moderate-eruption subsample alone constrains the slope only
weakly against the 53% log-scatter.  Extrapolated to the largest
eruptions (test), this shallow empirical slope is matched, but not
beaten, by the theory-fixed-exponent rung — illustrating that a
data-fit slope from a restricted range can extrapolate worse than a
first-principles exponent.  Test log_mae ~ 0.134.

LAW_CONSTANTS — frozen, pre-fit on v2 MER-OOD train
---------------------------------------------------
- A = 0.6681   (prefactor)
- B = 0.1753   (power-law exponent)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["MER_kg_per_s"]
PAPER_REF = "summary_formula_aubry_2023.md"
EQUATION_LOC = (
    "Aubry 2023 GRL Eq. 1 power-law form H_top = A*MER^B (PDF p. 3); "
    "A, B RE-fit on v2 MER-OOD train by log-log OLS (NOT the paper's "
    "0.345/0.226, which were fit on data including the OOD test)."
)

LAW_CONSTANTS = {
    "A": 0.6681,
    "B": 0.1753,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A: float = 0.6681, B: float = 0.1753) -> np.ndarray:
    """H_top = A * MER^B; empirical power law (Aubry form), train-refit."""
    MER = np.asarray(X[:, 0], dtype=float)
    return A * MER ** B
