"""Choudhury-Yang (1999) one-parameter Budyko runoff formula — best rung.

Choudhury, B. J. (1999), "Evaluation of an empirical equation for annual
evaporation using field observations and results from a biophysical
model", Journal of Hydrology 216:99-110,
DOI 10.1016/S0022-1694(98)00293-5.  Choudhury generalised the
Budyko/Pike/Turc family with a single shape exponent n (often written α
or, in Yang et al. 2008's identical re-derivation, n), giving actual
evapotranspiration (PDF p. 3, Eq. 3):

    E_actual = P / (1 + (P/PET)^n)^(1/n).

By water balance Q = P - E_actual:

    Q = P · (1 - 1 / (1 + (P/PET)^n)^(1/n)).

The exponent n controls how sharply the curve bends between the water
limit (Q -> 0 as PET/P -> infinity) and the energy limit
(Q -> P - PET as PET/P -> 0).  This is the canonical modern Budyko
curve; it reduces to the parameter-free forms for particular n and is
mathematically identical to the Fu (1981) / Yang et al. (2008) equation.

n is a single LAW_CONSTANT, pre-fit on the v2 train (humid / sub-humid
catchments) and frozen.  Fitting gives n ~ 1.69, consistent with the
literature range n ~ 1.5-2.6 (Choudhury found α ~ 1.8-2.6 across his
basins).  On the v2 aridity range-OOD test this rung achieves the lowest
error of the three (rmse ~ 0.41 mm/day) — though only marginally below
the parameter-free Pike curve, because the Budyko relation is robust and
a humid-fitted n transfers imperfectly into the arid regime.

LAW_CONSTANTS — frozen, pre-fit on v2 train (469 humid/sub-humid catchments)
---------------------------------------------------------------------------
- N = 1.6872   Budyko shape exponent

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["p_mean", "pet_mean"]
PAPER_REF = "summary_formula_choudhury_1999.md"
EQUATION_LOC = (
    "Choudhury 1999 J. Hydrol. 216:99, Eq. 3 (PDF p. 3): "
    "E = P/(1+(P/PET)^n)^(1/n), so Q = P·(1 - 1/(1+(P/PET)^n)^(1/n)).  "
    "Exponent N pre-fit on v2 train."
)

LAW_CONSTANTS = {
    "N": 1.6872,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, N: float = 1.6872) -> np.ndarray:
    """Q = P·(1 - 1/(1+(P/PET)^N)^(1/N)); Choudhury-Yang one-parameter Budyko."""
    P = np.asarray(X[:, 0], dtype=float)
    PET = np.asarray(X[:, 1], dtype=float)
    return P - P / (1.0 + (P / PET) ** N) ** (1.0 / N)
