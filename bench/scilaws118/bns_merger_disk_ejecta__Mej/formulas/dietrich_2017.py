"""Dietrich and Ujevic (2017) BNS dynamical-ejecta-mass fit.

Dietrich and Ujevic (2017) Eq. (1) (PDF p. 8) gives the dynamical ejecta
mass of a binary-neutron-star (BNS) merger as

    Mej_fit / (1e-3 M_sun)
        = [ a * (M2/M1)^(1/3) * (1 - 2*C1) / C1
            + b * (M2/M1)^n
            + c * (1 - M1/Mb1) ] * Mb1
          + (1 <-> 2)
          + d

where the bracketed expression is symmetrised under the (1 <-> 2)
relabelling (M2/M1 -> M1/M2, C1 -> C2, Mb1 -> Mb2). The fitted
coefficients (Eq. 2, PDF p. 8) are universal across all 172 BNS NR
simulations in the calibration set:

    a = -1.35695, b = 6.11252, c = -49.43355, d = 16.1144, n = -2.5484.

Symbol mapping to released-CSV columns:

    M1, M2  -> M1, M2  (gravitational masses, M_sun)
    M1*, M2* -> Mb1, Mb2 (baryonic masses, M_sun)
    C1, C2  -> C1, C2  (dimensionless compactnesses)

Output: Mej in M_sun (the formula's intrinsic output is in 1e-3 M_sun;
multiplied by 1e-3 here to match the released target unit).

Negative-prediction handling: the paper does not impose a clip; physical
ejecta mass is non-negative, so a max(0, ...) clip is applied near the
equal-mass / high-compactness corner where the fit can dip negative.

Setting / Type: setting1_typeI. All five coefficients (a, b, c, d, n)
are universal; no per-cluster secondaries.
"""

import numpy as np

USED_INPUTS = ["M1", "M2", "Mb1", "Mb2", "C1", "C2"]
PAPER_REF = "summary_formula_dataset_dietrich_2017.md"
EQUATION_LOC = "Eq. 1, p. 8 (coefficients in Eq. 2, p. 8)"
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a": -1.35695,
    "b": 6.11252,
    "c": -49.43355,
    "d": 16.1144,
    "n": -2.5484,
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def _half_term(M1, M2, Mb1, C1, a, b, c, n):
    # Single half of the symmetric (1 <-> 2) sum, in units of 1e-3 M_sun.
    ratio = M2 / M1
    term1 = a * ratio ** (1.0 / 3.0) * (1.0 - 2.0 * C1) / C1
    term2 = b * ratio ** n
    term3 = c * (1.0 - M1 / Mb1)
    return (term1 + term2 + term3) * Mb1


def predict(X,
            a=LAW_CONSTANTS["a"], b=LAW_CONSTANTS["b"], c=LAW_CONSTANTS["c"],
            d=LAW_CONSTANTS["d"], n=LAW_CONSTANTS["n"]):
    X = np.asarray(X, dtype=float)
    M1 = X[:, 0]
    M2 = X[:, 1]
    Mb1 = X[:, 2]
    Mb2 = X[:, 3]
    C1 = X[:, 4]
    C2 = X[:, 5]
    half_a = _half_term(M1, M2, Mb1, C1, a, b, c, n)
    half_b = _half_term(M2, M1, Mb2, C2, a, b, c, n)
    mej_in_1e_minus_3 = half_a + half_b + d
    return np.maximum(mej_in_1e_minus_3, 0.0) * 1e-3
