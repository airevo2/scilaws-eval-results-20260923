"""Rousset (1997) 2-D IBD linear regression (Type I, frozen LAW_CONSTANTS).

Rousset (1997), Genetics 145:1219-1228 (Eq. 8, p. 1221; "Implications
for Data Analysis", pp. 1223-1224), derives that for populations in a
two-dimensional continuous habitat the Rousset M-statistic
F_ST / (1 - F_ST) is asymptotically linear in the logarithm of the
geographic distance:

    M_stat = A + B * ln(d),

with B = 1 / (4·π·D·σ²), where D is reproductive density per unit
area and σ² is the second moment of the parental axial dispersal
distance.  Rousset 1997 does not tabulate a universal numerical (A, B)
pair for Drosophila — his two numerical examples involve human
populations and snails.  The v1 sister task's build_pairwise.py
provenance documented a self-fit on the DEST mainland-Europe subset
("M = -0.017 + 0.0091 * ln(d)").  This v2 Type I task freezes (A, B)
from the analogous OLS fit on the v2 TRAIN distance band (848 km -
2575 km, 828 pairs), which is the literature-equivalent for a
"calibration band" frozen-constant evaluation:

    A_train = -0.08523
    B_train =  0.01892    (1 / ln(km))

These differ from the v1 mainland-only numbers because (a) the v2
train band excludes very-short-distance within-country pairs and
very-long-distance trans-continental pairs (the test set), and (b)
the v2 set includes all 53 populations, not only mainland Europe.

This is a TYPE I baseline: LOCAL_FITTABLE is empty, no fit() function,
predict() is invoked once on the full test set with LAW_CONSTANTS
only.  The benchmark question is whether the train-band-calibrated
linear-in-ln(d) IBD form extrapolates to the test extremes — very
short distances (where ln(d) → 0 makes M predicted ≈ A, potentially
negative and unphysical) and very long distances (>2575 km, where
trans-continental Guadeloupe ↔ Europe pairs dominate).

LAW_CONSTANTS — frozen, pre-fit on v2 train data band
-----------------------------------------------------
- A = -0.08523  (intercept, dimensionless)
- B =  0.01892  (slope, 1 / ln(km))

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["log_distance_km"]
PAPER_REF = "rousset_1997.pdf"
EQUATION_LOC = (
    "Rousset (1997) Eq. 8, p. 1221 (2-D form M_stat = A + B·ln(d) with "
    "B = 1/(4πD·σ²)); (A, B) frozen from v2 train-band OLS calibration."
)

LAW_CONSTANTS = {
    "A": -0.08523,
    "B":  0.01892,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A: float = -0.08523, B: float = 0.01892) -> np.ndarray:
    """M_stat = A + B * log_distance_km."""
    log_d = np.asarray(X[:, 0], dtype=float)
    return A + B * log_d
