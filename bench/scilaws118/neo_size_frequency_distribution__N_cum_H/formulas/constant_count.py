"""Constant cumulative count N_cum = A0 — weakest rung.

The crudest baseline ignores the magnitude dependence and predicts a
single constant cumulative count (the train mean).  It captures none of
the size-frequency distribution's exponential growth with magnitude.
Test log_mae ~ 0.45 — the ladder's lower bound.

LAW_CONSTANTS — frozen, pre-fit on v2 train
-------------------------------------------
- A0 = 266.56   (mean cumulative count in the H in [12,18] window)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["H_upper"]
PAPER_REF = "summary_formula_bottke_2002.md"
EQUATION_LOC = (
    "Constant baseline N_cum = A0 (train mean); no magnitude dependence. "
    "Ladder lower bound for the exponential SFD forms."
)

LAW_CONSTANTS = {
    "A0": 266.56,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A0: float = 266.56) -> np.ndarray:
    """N_cum = A0 (constant)."""
    H = np.asarray(X[:, 0], dtype=float)
    return np.full_like(H, A0)
