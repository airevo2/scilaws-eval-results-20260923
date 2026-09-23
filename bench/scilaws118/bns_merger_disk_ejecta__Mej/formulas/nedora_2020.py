"""Nedora et al. (2020) two-parameter polynomial fit for BNS Mej.

Nedora et al. (2020) Eq. (6) (PDF p. 4) gives a second-order polynomial
fit for log10(Mej / M_sun) in (q, Lambda_tilde):

    log10(Mej / M_sun)
        = b0 + b1*qp + b2*Lt + b3*qp^2 + b4*qp*Lt + b5*Lt^2

Recommended calibration: P^2_2(q, Lambda_tilde) on M0/M1Set (Table IV,
PDF p. 14):

    b0 = -1.32
    b1 = -3.82e-1
    b2 = -4.47e-3
    b3 = -3.39e-1
    b4 =  3.21e-3
    b5 =  4.31e-7

The paper convention is qp = M_A / M_B with M_A >= M_B (qp >= 1). The
released-CSV q column is q = M2/M1 with M1 >= M2 (q in (0, 1]); the two
are reciprocals, so this `predict` evaluates the polynomial at
qp = 1/q.

Symbol mapping to released-CSV columns:

    q             -> 1 / q (released CSV q is the reciprocal of paper q)
    Lambda_tilde  -> Lambda_tilde

Output: Mej in M_sun (formula returns log10(Mej/M_sun); exponentiated).

Validity domain (M0/M1Set): qp in [1.0, 1.30], Lt in [340, 1437].
Predictions outside this domain extrapolate the polynomial.

Setting / Type: setting1_typeI. All six coefficients are universal; no
per-cluster secondaries.
"""

import numpy as np

USED_INPUTS = ["q", "Lambda_tilde"]
PAPER_REF = "summary_formula+dataset_nedora_2020.md"
EQUATION_LOC = "Eq. 6, p. 4 (coefficients M0/M1Set row, Table IV, p. 14)"
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "b0": -1.32,
    "b1": -3.82e-1,
    "b2": -4.47e-3,
    "b3": -3.39e-1,
    "b4":  3.21e-3,
    "b5":  4.31e-7,
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X,
            b0=LAW_CONSTANTS["b0"], b1=LAW_CONSTANTS["b1"], b2=LAW_CONSTANTS["b2"],
            b3=LAW_CONSTANTS["b3"], b4=LAW_CONSTANTS["b4"], b5=LAW_CONSTANTS["b5"]):
    X = np.asarray(X, dtype=float)
    q_csv = X[:, 0]
    Lt = X[:, 1]
    qp = 1.0 / q_csv  # paper convention: qp = M_heavy / M_light >= 1
    log10_mej = (b0
                 + b1 * qp
                 + b2 * Lt
                 + b3 * qp * qp
                 + b4 * qp * Lt
                 + b5 * Lt * Lt)
    return np.power(10.0, log10_mej)
