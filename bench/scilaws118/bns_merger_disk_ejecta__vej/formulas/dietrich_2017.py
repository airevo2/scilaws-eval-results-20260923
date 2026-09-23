"""Dietrich and Ujevic (2017) BNS dynamical-ejecta velocity fit.

Dietrich and Ujevic (2017) Eq. 5-9 (PDF p. 10-11) decompose the mass-averaged
terminal ejecta velocity vej into in-plane (v_rho) and out-of-plane (v_z)
components, each following the same functional form

    v_x = a*(M1/M2)*(1 + c*C1) + a*(M2/M1)*(1 + c*C2) + b,

with separately fitted (a, b, c) triples for the two components, and
combined as

    vej = sqrt(v_rho**2 + v_z**2).

The published constants are calibrated on 66 NR BNS simulations from six
independent groups (PDF p. 10, Section 3.3) and are universal across the
calibration set. They are stored in LAW_CONSTANTS: the harness
re-fits them once on train.csv, with the published values as the initial point.

Symbol map: paper M_1, M_2 (gravitational masses, M_sun) -> CSV M1, M2;
paper C_1, C_2 (compactness GM/(R c**2)) -> CSV C1, C2; output in units of c.

Validity caveat: the velocity fit is calibrated on irrotational simulations;
spinning binaries can drift by O(10%) (PDF p. 14, Section 3.4.3). The fit
slightly underestimates vej across the calibration set (PDF p. 11).
"""

import numpy as np

USED_INPUTS = ["M1", "M2", "C1", "C2"]
PAPER_REF = "summary_formula+dataset_dietrich_2017.md"
EQUATION_LOC = "Eq. 5-9, p. 10-11 (constants Eq. 6 and Eq. 8)"
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a_rho": -0.219479,
    "b_rho": 0.444836,
    "c_rho": -2.67385,
    "a_z": -0.315585,
    "b_z": 0.63808,
    "c_z": -1.00757,
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(
    X,
    a_rho=LAW_CONSTANTS["a_rho"],
    b_rho=LAW_CONSTANTS["b_rho"],
    c_rho=LAW_CONSTANTS["c_rho"],
    a_z=LAW_CONSTANTS["a_z"],
    b_z=LAW_CONSTANTS["b_z"],
    c_z=LAW_CONSTANTS["c_z"],
):
    X = np.asarray(X, dtype=float)
    M1 = X[:, 0]
    M2 = X[:, 1]
    C1 = X[:, 2]
    C2 = X[:, 3]
    r12 = M1 / M2
    r21 = M2 / M1
    v_rho = a_rho * r12 * (1.0 + c_rho * C1) + a_rho * r21 * (1.0 + c_rho * C2) + b_rho
    v_z = a_z * r12 * (1.0 + c_z * C1) + a_z * r21 * (1.0 + c_z * C2) + b_z
    return np.sqrt(v_rho * v_rho + v_z * v_z)
