"""Free exponential size-frequency distribution — middle rung.

The cumulative NEO size-frequency distribution is a power law in size,
equivalently an EXPONENTIAL in absolute magnitude H:

    N(<H) = A * 10^(alpha * H),

i.e. log10 N grows linearly with H (each magnitude fainter, ~10^alpha
times more objects).  Here BOTH the amplitude A and the magnitude slope
alpha are fit on the v2 train.

This captures the SFD's exponential growth (test log_mae ~ 0.044, far
below the constant rung's ~0.45).  The fitted slope alpha ~ 0.48 is
close to — but, being constrained only by the restricted-range training
sample against the catalog scatter, slightly below — the
collisional-equilibrium value 0.5 that the next rung fixes from theory.

LAW_CONSTANTS — frozen, pre-fit on v2 train
-------------------------------------------
- LOG_A = -5.4989   (log10 amplitude; A = 10^LOG_A)
- ALPHA =  0.4787   (magnitude slope, fit)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["H_upper"]
PAPER_REF = "summary_formula_bottke_2002.md"
EQUATION_LOC = (
    "Exponential cumulative SFD N(<H) = A*10^(alpha*H) (Bottke 2002; "
    "summary_formula_bottke_2002.md).  log10 A and alpha both pre-fit on "
    "v2 train by log-linear OLS."
)

LAW_CONSTANTS = {
    "LOG_A": -5.4989,
    "ALPHA":  0.4787,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, LOG_A: float = -5.4989, ALPHA: float = 0.4787) -> np.ndarray:
    """N(<H) = 10^(LOG_A + ALPHA*H); free exponential SFD."""
    H = np.asarray(X[:, 0], dtype=float)
    return 10.0 ** (LOG_A + ALPHA * H)
