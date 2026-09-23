"""Pike (1964) parameter-free Budyko runoff formula — middle rung.

Pike, J. G. (1964), "The estimation of annual run-off from
meteorological data in a tropical climate", Journal of Hydrology
2:116-123, DOI 10.1016/0022-1694(64)90022-8.  Pike adapted Turc's
(1954) closed-form evapotranspiration equation by replacing the
temperature-based energy proxy L with Penman's open-water evaporation
estimate E_o (here the potential evapotranspiration PET), and set the
outer constant to unity, giving the actual evapotranspiration (PDF p. 3,
Eq. 5):

    E_actual = P / sqrt(1 + (PET/P)^2)  =  P·PET / sqrt(P^2 + PET^2).

By water balance Q = P - E_actual:

    Q = P - P·PET / sqrt(P^2 + PET^2)  =  P·(1 - PET / sqrt(P^2 + PET^2)).

This is a genuine Budyko-type curve: it interpolates smoothly between
the water limit (Q -> 0 as PET/P -> infinity, the arid asymptote) and
the energy limit (Q -> P - PET as PET/P -> 0, the humid asymptote),
capturing the curvature the demand-limit rung misses.  It is
PARAMETER-FREE — the functional form has no free constant (the "1" and
"2" are structural), so nothing is fit on train.

Pike's curve is one specific member of the Budyko family; the
single-parameter Choudhury-Yang generalisation (next rung) can bend the
curve to fit a particular dataset better.  On the v2 aridity range-OOD
test, Pike achieves rmse ~ 0.44 mm/day — better than the demand limit
(~0.67) but slightly worse than the fitted Choudhury-Yang form (~0.41).

LAW_CONSTANTS / OTHER_CONSTANTS / LOCAL_FITTABLE
-----------------------------------------------
None — Pike's form is parameter-free (Type I).
"""

import numpy as np

USED_INPUTS = ["p_mean", "pet_mean"]
PAPER_REF = "summary_formula_pike_1964.md"
EQUATION_LOC = (
    "Pike 1964 J. Hydrol. 2:116, Eq. 5 (PDF p. 3): "
    "Q = P·(1 - PET/sqrt(P^2 + PET^2)).  Parameter-free Budyko-type "
    "curve (Turc 1954 form with Penman E_o)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray) -> np.ndarray:
    """Q = P·(1 - PET/sqrt(P^2 + PET^2)); Pike 1964 parameter-free Budyko."""
    P = np.asarray(X[:, 0], dtype=float)
    PET = np.asarray(X[:, 1], dtype=float)
    return P - P * PET / np.sqrt(P * P + PET * PET)
