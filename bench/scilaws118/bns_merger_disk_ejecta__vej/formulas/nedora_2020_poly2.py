"""Nedora et al. (2020) recommended P_2^2(q, Lambda_tilde) polynomial.

Nedora et al. (2020) introduces a second-order polynomial fit P_2^2 in the
binary mass ratio q and reduced tidal deformability Lambda_tilde for the
mass-averaged terminal ejecta velocity <v_inf>:

    vej = b0 + b1*q + b2*Lambda_tilde + b3*q**2
          + b4*q*Lambda_tilde + b5*Lambda_tilde**2.

This is the recommended (highlighted) row of Table IV (PDF p. 14) at the
MORefSet + M0/M1Set calibration:

    b0 = 5.94e-1, b1 = -1.48e-1, b2 = -8.62e-4,
    b3 = -5.02e-2, b4 = 3.25e-4,  b5 = 3.16e-7,

with reduced chi-squared 1.6. The functional form is identical to the
Nedora 2021 polynomial (Eq. 10) but the constants differ because the
calibration dataset and physics input differ.

Symbol map: paper q -> CSV q (>= 1); paper Lambda_tilde -> CSV
Lambda_tilde; output in units of c.

Validity caveat: calibration domain q in [1.0, 2.06], Lambda_tilde in
[50, 3196] (PDF p. 2). All six coefficients are universal across the
calibration set (stored in LAW_CONSTANTS).
"""

import numpy as np

USED_INPUTS = ["q", "Lambda_tilde"]
PAPER_REF = "summary_formula_dataset_nedora_2020.md"
EQUATION_LOC = "P_2^2 polynomial (Eq. 8 form), Table IV row 'MORefSet & M0/M1Set', p. 14"
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "b0": 5.94e-1,
    "b1": -1.48e-1,
    "b2": -8.62e-4,
    "b3": -5.02e-2,
    "b4": 3.25e-4,
    "b5": 3.16e-7,
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(
    X,
    b0=LAW_CONSTANTS["b0"],
    b1=LAW_CONSTANTS["b1"],
    b2=LAW_CONSTANTS["b2"],
    b3=LAW_CONSTANTS["b3"],
    b4=LAW_CONSTANTS["b4"],
    b5=LAW_CONSTANTS["b5"],
):
    X = np.asarray(X, dtype=float)
    q = X[:, 0]
    Lt = X[:, 1]
    return b0 + b1 * q + b2 * Lt + b3 * q * q + b4 * q * Lt + b5 * Lt * Lt
