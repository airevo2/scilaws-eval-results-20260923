"""Nedora et al. (2020) physics-motivated ejecta-velocity fit (Eq. 9).

Nedora et al. (2020) Eq. 9 (PDF p. 6) is a recalibration of the physics-
motivated single-formula expression for the mass-averaged terminal ejecta
velocity <v_inf>:

    vej = alpha * (M1/M2) * (1 + gamma*C1)
        + alpha * (M2/M1) * (1 + gamma*C2)
        + beta.

The functional form is the one introduced in Kruger and Foucart (2020) /
Dietrich and Ujevic (2017) but with constants refit on Nedora 2020's
combined microphysical-EOS dataset MORefSet + M0/M1Set (Table V, PDF p. 14):

    alpha = -0.5631, beta = 1.109, gamma = -1.186,

with reduced chi-squared 2.3. The three constants are universal across
the calibration set (stored in LAW_CONSTANTS).

Symbol map: paper M_A, M_B (gravitational masses, M_sun) -> CSV M1, M2
(M1 = heavier, M2 = lighter, by convention M_A >= M_B); paper C_A, C_B
(compactness GM/(R c**2)) -> CSV C1, C2; output in units of c.

Validity caveat: the formula is symmetric under (M1, C1) <-> (M2, C2) by
construction, so the M1 >= M2 ordering is not load-bearing. Calibration
domain: q in [1.0, 2.06], Lambda_tilde in [50, 3196], M_tot in
[2.4, 4.0] M_sun (PDF p. 2, 12).
"""

import numpy as np

USED_INPUTS = ["M1", "M2", "C1", "C2"]
PAPER_REF = "summary_formula_dataset_nedora_2020.md"
EQUATION_LOC = "Eq. 9, p. 6 (coefficients Table V row 'MORefSet & M0/M1Set', p. 14)"
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {"alpha": -0.5631, "beta": 1.109, "gamma": -1.186}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(
    X,
    alpha=LAW_CONSTANTS["alpha"],
    beta=LAW_CONSTANTS["beta"],
    gamma=LAW_CONSTANTS["gamma"],
):
    X = np.asarray(X, dtype=float)
    M1 = X[:, 0]
    M2 = X[:, 1]
    C1 = X[:, 2]
    C2 = X[:, 3]
    r12 = M1 / M2
    r21 = M2 / M1
    return alpha * r12 * (1.0 + gamma * C1) + alpha * r21 * (1.0 + gamma * C2) + beta
