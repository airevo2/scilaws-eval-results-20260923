"""Linear H-MER model — wrong functional form, weakest rung.

A naive linear fit H_top = A*MER + B.  This is the WRONG functional
form for the column-height / mass-eruption-rate relationship: MER spans
seven orders of magnitude (1e1 - 1e8 kg/s) while H_top spans only ~two
(0.5 - 36 km).  A linear model is dominated by the few highest-MER
events and grossly mispredicts the vast majority of small/moderate
eruptions; it cannot capture the power-law (concave-in-log) growth that
buoyant-plume theory requires.

Included as the ladder's lower bound — any genuine power-law form must
beat it.  Coefficients pre-fit on the v2 MER-range-OOD train (98
lowest-MER events).  Test log_mae ~ 0.37 (vs ~0.13 for the power-law
rungs).

LAW_CONSTANTS — frozen, pre-fit on v2 train
-------------------------------------------
- A = 1.3206e-06   (km per kg/s)
- B = 5.2679       (km, intercept)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["MER_kg_per_s"]
PAPER_REF = "summary_formula_aubry_2023.md"
EQUATION_LOC = (
    "Naive linear baseline H_top = A*MER + B (wrong form; the Aubry 2023 "
    "relation is a power law).  A, B pre-fit on v2 MER-OOD train by OLS."
)

LAW_CONSTANTS = {
    "A": 1.3206e-06,
    "B": 5.2679,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A: float = 1.3206e-06, B: float = 5.2679) -> np.ndarray:
    """H_top = A*MER + B (linear; clipped at 0.01 km to stay positive)."""
    MER = np.asarray(X[:, 0], dtype=float)
    return np.maximum(A * MER + B, 0.01)
