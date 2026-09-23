"""Constant cross section log10(sigma) = a0 — weakest rung.

The crudest baseline ignores the energy dependence entirely and predicts
a single constant log10 cross section (the train mean).  It captures
neither the 1/v thermal decline nor the 4.9 eV resonance peak.  Test
rmse ~ 1.07 (log10 barn), r^2 ~ 0 — the ladder's lower bound.

LAW_CONSTANTS — frozen, pre-fit on v2 train
-------------------------------------------
- A0 = 0.9678   (mean log10 cross section in the E < 50 eV window)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["E_eV"]
PAPER_REF = "summary_exfor_au197.md"
EQUATION_LOC = (
    "Constant baseline log10(sigma) = a0 (train-mean); no energy "
    "dependence.  Ladder lower bound for the 1/v + Breit-Wigner forms."
)

LAW_CONSTANTS = {
    "A0": 0.9678,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A0: float = 0.9678) -> np.ndarray:
    """log10(sigma) = A0 (constant)."""
    E = np.asarray(X[:, 0], dtype=float)
    return np.full_like(E, A0)
