"""Krueger and Foucart (2020) BNS dynamical-ejecta-mass fit.

Krueger and Foucart (2020) Eq. (6) (PDF p. 5) gives a compact alternative
to the Dietrich and Ujevic (2017) form that uses only gravitational
masses and compactnesses (no baryonic masses required), and that has the
physically correct sign of dM_dyn/dC:

    Mdyn / (1e-3 M_sun)
        = ( a / C1 + b * (M2/M1)^n + c * C1 ) * M1
          + (1 <-> 2)

The bracketed expression is symmetrised under the (1 <-> 2) relabelling.
Fitted coefficients (200 NR simulations; PDF p. 5):

    a = -9.3335, b = 114.17, c = -337.56, n = 1.5465.

Negative predictions are interpreted as Mdyn = 0 (PDF p. 5).

Symbol mapping to released-CSV columns:

    M1, M2 -> M1, M2 (gravitational masses, M_sun)
    C1, C2 -> C1, C2 (dimensionless compactnesses)

Output: Mej in M_sun (formula returns Mdyn in 1e-3 M_sun; multiplied by
1e-3 here to match the released target unit, with a max(0, .) clip).

Setting / Type: setting1_typeI. All four coefficients (a, b, c, n) are
universal; no per-cluster secondaries.
"""

import numpy as np

USED_INPUTS = ["M1", "M2", "C1", "C2"]
PAPER_REF = "summary_formula_kruger_2020.md"
EQUATION_LOC = "Eq. 6, p. 5"
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a": -9.3335,
    "b": 114.17,
    "c": -337.56,
    "n": 1.5465,
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def _half_term(M1, M2, C1, a, b, c, n):
    return (a / C1 + b * (M2 / M1) ** n + c * C1) * M1


def predict(X,
            a=LAW_CONSTANTS["a"], b=LAW_CONSTANTS["b"], c=LAW_CONSTANTS["c"], n=LAW_CONSTANTS["n"]):
    X = np.asarray(X, dtype=float)
    M1 = X[:, 0]
    M2 = X[:, 1]
    C1 = X[:, 2]
    C2 = X[:, 3]
    mdyn_in_1e_minus_3 = (_half_term(M1, M2, C1, a, b, c, n)
                          + _half_term(M2, M1, C2, a, b, c, n))
    return np.maximum(mdyn_in_1e_minus_3, 0.0) * 1e-3
