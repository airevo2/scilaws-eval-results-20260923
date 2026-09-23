"""Demand-limit (potential-ET) water balance — weakest rung.

The long-term catchment water balance is Q = P - E_actual, where
E_actual is the actual evapotranspiration.  The simplest closed-form
assumption is that the catchment evaporates at its full atmospheric
demand, i.e. E_actual = PET (potential evapotranspiration).  This is the
"energy / demand limit" — one of the two physical bounds of the Budyko
framework (the other is the water limit E <= P).  It gives

    Q = max(P - PET, 0)

(clamped at zero because runoff cannot be negative).

This assumption is reasonable in strongly water-limited deserts and in
very humid catchments where P >> PET, but it is systematically wrong in
the intermediate and arid regimes the Budyko curve was built to
describe: real catchments evaporate LESS than the potential rate when
water is scarce (E_actual < PET as aridity rises), so the demand limit
under-predicts runoff in arid catchments (and predicts exactly zero
whenever PET > P, which is most of the arid test set).  It captures no
curvature of the Budyko relation — it is the straight-line upper bound,
not the curve.

On the v2 aridity range-OOD test (arid catchments, PET/P > 1.1) this
rung is the weakest: test rmse ~ 0.67 mm/day, versus ~0.41 for the
Budyko-curve rungs.

LAW_CONSTANTS / OTHER_CONSTANTS / LOCAL_FITTABLE
-----------------------------------------------
None — the demand limit is parameter-free (Type I).
"""

import numpy as np

USED_INPUTS = ["p_mean", "pet_mean"]
PAPER_REF = "summary_formula_choudhury_1999.md"
EQUATION_LOC = (
    "Budyko-framework energy/demand limit E_actual = PET, so "
    "Q = max(P - PET, 0).  The straight-line upper bound discussed as "
    "the limiting case in Choudhury 1999 §1 (Budyko/Schreiber/Ol'dekop "
    "tradition); parameter-free."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray) -> np.ndarray:
    """Q = max(P - PET, 0); demand-limit water balance."""
    P = np.asarray(X[:, 0], dtype=float)
    PET = np.asarray(X[:, 1], dtype=float)
    return np.maximum(P - PET, 0.0)
