"""Darc et al. (2025) PyOperon SR-derived disk-mass expression.

Darc 2025 Eq. (3) (PDF p. 4) -- the authors' recommended primary
SR alternative:

    M_disk = a0 - a1 * sin(sin(a2 * C_1))

with SR-fitted coefficients on the Krueger 2020 56-simulation calibration
set:

    a0 = 0.118824,
    a1 = 0.142985,
    a2 = 40.896317.

Convention: Darc 2025 follows Krueger & Foucart's M_1 <= M_2 labelling
(PDF p. 2 footnote), so the paper's `C_1` maps directly to the released
`C1` column.

Output is in solar masses. The double-sine form has no explicit floor;
the paper notes (PDF p. 5) that for very high C_1 the second sine can
fluctuate, suggesting a clip at M_disk >= 0 in production use. This
implementation does NOT clip -- the SR baseline is reported verbatim.

Calibration domain: C_1 in approx [0.13, 0.22], q = M_1/M_2 in [0.77, 1.0]
(Krueger 2020 calibration set).

All three coefficients are SR-discovered universal constants;
LOCAL_FITTABLE is empty for this Setting 1 Type I task.
"""

import numpy as np

USED_INPUTS = ["C1"]
PAPER_REF = "summary_formula_darc_2025.md"
EQUATION_LOC = "Eq. 3, p. 4"
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {"a0": 0.118824, "a1": 0.142985, "a2": 40.896317}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X, a0=LAW_CONSTANTS["a0"], a1=LAW_CONSTANTS["a1"], a2=LAW_CONSTANTS["a2"]):
    C1 = np.asarray(X[:, 0], dtype=float)
    return a0 - a1 * np.sin(np.sin(a2 * C1))
