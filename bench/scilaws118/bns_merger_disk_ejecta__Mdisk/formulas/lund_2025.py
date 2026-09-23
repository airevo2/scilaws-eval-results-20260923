"""Lund et al. (2025) BNS remnant disk-mass log-tanh fit.

Lund 2025 Eq. (2) (PDF p. 3):

    log10(m_disk) = alpha * tanh(beta * C_light + gamma) + delta

with best-fit coefficients on a 112-simulation NR catalogue (PDF p. 3):

    alpha = -1.21, beta = 72.62, gamma = -12.48, delta = -1.93.

C_light is the compactness of the lighter NS, defined in Lund 2025 Eq. (1)
as C_light = G M_light / (R_light c^2). The released CSV uses the M_1 <= M_2
labelling convention, so the paper's `C_light` maps directly to the released
`C1` column.

Output is in solar masses (the formula gives log10(m_disk / M_sun);
exponentiate). The log-tanh form has no explicit floor, but the saturation
of tanh at high compactness drives m_disk towards a finite asymptote
(approx 10^(delta - alpha) approximately 10^(-0.72) M_sun for low C_1, and
approx 10^(delta + alpha) approximately 10^(-3.14) M_sun for high C_1).

Calibration domain: C_1 in [0.12, 0.20] over 112 NR simulations from 11
sources, average mass ratio q = 0.91. The fit is global (no per-cluster
parameter); all four coefficients are universal (PDF p. 3, "best-fit
parameters alpha = -1.21, beta = 72.62, gamma = -12.48, and delta = -1.93").

Type / setting: setting1_typeI; LOCAL_FITTABLE is empty.
"""

import numpy as np

USED_INPUTS = ["C1"]
PAPER_REF = "summary_formula_lund_2025.md"
EQUATION_LOC = "Eq. 2, p. 3"
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {"alpha": -1.21, "beta": 72.62, "gamma": -12.48, "delta": -1.93}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X, alpha=LAW_CONSTANTS["alpha"], beta=LAW_CONSTANTS["beta"],
            gamma=LAW_CONSTANTS["gamma"], delta=LAW_CONSTANTS["delta"]):
    C1 = np.asarray(X[:, 0], dtype=float)
    log10_m = alpha * np.tanh(beta * C1 + gamma) + delta
    return 10.0 ** log10_m
