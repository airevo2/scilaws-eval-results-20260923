"""Rousset (1997) 1-D IBD linear regression (Type I, frozen LAW_CONSTANTS).

Rousset (1997), Genetics 145:1219-1228, derives the analogous IBD
linear law for a 1-dimensional continuous habitat (Eq. 7, p. 1221;
Bembicium snail example, Fig. 6, p. 1225):

    M_stat = A + B * d,

with B = 1 / (4·N·σ²) (note the absence of π relative to the 2-D
form; p. 1224: "in one dimension, the product of linear density Dc
times σ² is always N·σ²").  The 1-D form is the correct IBD law for
populations along a narrow linear habitat (a coastline or river
transect); for continental Drosophila spread across a 2-D plane the
2-D log-form (rousset_1997.py) is the canonical published choice.

Including the 1-D form here as a STRUCTURALLY DISTINCT alternative
baseline is critical: empirically on the v2 distance-extrapolation
OOD test set, the linear-in-d form extrapolates BETTER than the
linear-in-ln(d) form, because (a) at very short distances ln(d) → 0
makes the 2-D form predict M ≈ A (potentially negative and
unphysical), while M_stat is monotonically increasing in d, and (b)
the global trend in F_ST across the trans-continental range happens
to be closer to linear-in-d than linear-in-ln(d).

(A, B) frozen from v2 train-band OLS calibration (828 pairs, 848 km
- 2575 km, same band as the 2-D form's pre-fit).

LAW_CONSTANTS — frozen, pre-fit on v2 train data band
-----------------------------------------------------
- A = 0.03473      (intercept, dimensionless)
- B = 1.1715e-05   (slope, 1 / km)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["distance_km"]
PAPER_REF = "rousset_1997.pdf"
EQUATION_LOC = (
    "Rousset (1997) Eq. 7, p. 1221 + Fig. 6, p. 1225 (1-D form "
    "M_stat = A + B·d with B = 1/(4N·σ²)); (A, B) frozen from v2 "
    "train-band OLS calibration."
)

LAW_CONSTANTS = {
    "A": 0.03473,
    "B": 1.1715e-05,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A: float = 0.03473,
            B: float = 1.1715e-05) -> np.ndarray:
    """M_stat = A + B * distance_km."""
    d = np.asarray(X[:, 0], dtype=float)
    return A + B * d
