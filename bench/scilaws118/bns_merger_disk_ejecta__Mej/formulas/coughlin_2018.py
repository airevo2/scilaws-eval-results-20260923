"""Coughlin et al. (2018) BNS NR dynamical-ejecta-mass fit.

Coughlin et al. (2018) Eq. (E8) (PDF p. 15) refits the Dietrich and
Ujevic (2017) BNS NR catalogue, fitting log10(Mej^NR) instead of Mej:

    log10(Mej^NR / M_sun)
        = [ a * (1 - 2*C1) * M1 / C1
            + b * M2 * (M1/M2)^n
            + d/2 ]
          + (1 <-> 2)

where the bracketed expression is repeated with all subscripts 1 and 2
swapped and added to the first. The fitted coefficients (PDF p. 15) are
universal across the calibration set:

    a = -0.0812, b = 0.2288, d = -2.16, n = -2.51.

Symbol mapping to released-CSV columns:

    M1, M2 -> M1, M2 (gravitational masses, M_sun)
    C1, C2 -> C1, C2 (dimensionless compactnesses)

Output: Mej in M_sun (formula returns log10(Mej^NR / M_sun); exponentiated
here). The paper introduces a per-event scale factor A > 1 (Eq. 1, p. 6)
that converts Mej^NR to total kilonova ejecta mass, but A is a per-event
inferred quantity, not part of the closed-form NR predictor; the released
target Mej is calibrated against simulation Mej^NR, so A = 1 is the
appropriate baseline.

Setting / Type: setting1_typeI. All four coefficients (a, b, d, n) are
universal; no per-cluster secondaries.
"""

import numpy as np

USED_INPUTS = ["M1", "M2", "C1", "C2"]
PAPER_REF = "summary_formula_coughlin_2018.md"
EQUATION_LOC = "Eq. E8, p. 15"
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a": -0.0812,
    "b": 0.2288,
    "d": -2.16,
    "n": -2.51,
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def _half_term(M1, M2, C1, a, b, d, n):
    # Single half of the symmetric (1 <-> 2) sum.
    return (a * (1.0 - 2.0 * C1) * M1 / C1
            + b * M2 * (M1 / M2) ** n
            + d / 2.0)


def predict(X,
            a=LAW_CONSTANTS["a"], b=LAW_CONSTANTS["b"], d=LAW_CONSTANTS["d"], n=LAW_CONSTANTS["n"]):
    X = np.asarray(X, dtype=float)
    M1 = X[:, 0]
    M2 = X[:, 1]
    C1 = X[:, 2]
    C2 = X[:, 3]
    log10_mej = (_half_term(M1, M2, C1, a, b, d, n)
                 + _half_term(M2, M1, C2, a, b, d, n))
    return np.power(10.0, log10_mej)
