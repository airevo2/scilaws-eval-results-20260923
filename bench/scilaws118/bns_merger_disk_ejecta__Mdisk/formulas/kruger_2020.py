"""Krueger & Foucart (2020) BNS remnant disk-mass fit.

Krueger & Foucart 2020 Eq. (4) (PDF p. 3):

    M_disk = M_1 * max(5e-4, (a * C_1 + c) ** d)

with least-squares coefficients on a 57-simulation NR catalogue
(Radice et al. 2018 + Kiuchi et al. 2019):

    a = -8.1324, c = 1.4820, d = 1.7784.

Convention: M_1 is the lighter neutron star (M_1 <= M_2), C_1 = G M_1 / (R_1 c^2)
its compactness. The released CSV header uses the same M_1 <= M_2 convention,
so column names map directly:

    paper M_1 -> released M1
    paper C_1 -> released C1

Output is in solar masses. The 5e-4 floor is a structural numerical-error
floor that prevents negative disk-mass predictions in the high-compactness
regime (paper PDF p. 3 footnote on Eq. 3 / Eq. 4).

Calibration domain: C_1 in [0.135, 0.219], q = M_1/M_2 in [0.775, 1.0],
zero NS spin in all training simulations.

All three coefficients (a, c, d) are universal across the dataset (PDF p. 3,
"least squares fit using (4) yields the coefficients a = -8.1324, c = 1.4820,
and d = 1.7784"). They are stored in LAW_CONSTANTS as the frozen, paper-published
scientific claim — evaluated as published on the test split, NOT refit on the
released training data. LOCAL_FITTABLE is empty for this Setting 1 Type I task.
"""

import numpy as np

USED_INPUTS = ["M1", "C1"]
PAPER_REF = "summary_formula_kruger_2020.md"
EQUATION_LOC = "Eq. 4, p. 3"
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {"a": -8.1324, "c": 1.4820, "d": 1.7784}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X, a=LAW_CONSTANTS["a"], c=LAW_CONSTANTS["c"], d=LAW_CONSTANTS["d"]):
    M1 = np.asarray(X[:, 0], dtype=float)
    C1 = np.asarray(X[:, 1], dtype=float)
    inner = a * C1 + c
    # Real-valued power: clip the base to >= 0 before exponentiation so a
    # non-integer exponent does not produce NaN. Negative inner values would
    # be replaced by the 5e-4 floor anyway.
    base = np.maximum(inner, 0.0)
    powered = base ** d
    return M1 * np.maximum(5.0e-4, powered)
