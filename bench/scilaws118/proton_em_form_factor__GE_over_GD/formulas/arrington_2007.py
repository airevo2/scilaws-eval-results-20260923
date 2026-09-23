"""Arrington-Melnitchouk-Tjon (2007) global fit of G_E^p / G_D.

Citation: Arrington, Melnitchouk & Tjon, Phys. Rev. C 76, 035205 (2007),
arXiv:0707.1861. Primary formula: Eq. (11), PDF p. 6 (Sec. III.A).
Coefficients: Table I, PDF p. 6.

Formula
-------
    G_E(Q^2) = (1 + a1*tau + a2*tau^2 + a3*tau^3)
               / (1 + b1*tau + b2*tau^2 + b3*tau^3 + b4*tau^4 + b5*tau^5)

    tau = Q^2 / (4 * M_p^2)

    G_D(Q^2) = (1 + Q^2 / Lambda2)^(-2)

    Target: GE_over_GD(Q^2) = G_E(Q^2) / G_D(Q^2)

n = 3 yields 8 fit parameters; denominator coefficients constrained positive
(PDF p. 7) to prevent narrow divergences. Valid up to Q^2 = 6 GeV^2 for G_E
(PDF p. 6). Fit to 569 cross-section + 54 polarisation-transfer points after
explicit two-photon-exchange (TPE) corrections (PDF p. 6).

LAW_CONSTANTS — Table I, PDF p. 6 (G_E column)
-----------------------------------------------
    a1 =   3.439
    a2 =  -1.602
    a3 =   0.068
    b1 =  15.055
    b2 =  48.061
    b3 =  99.304
    b4 =   0.012
    b5 =   8.650

OTHER_CONSTANTS — universal / structural
-----------------------------------------
    M_p      = 0.938 GeV  — proton mass; tau definition (PDF p. 2, Eq. 2 context;
                            stated in passing in Sec. III.A, PDF p. 6).
    Lambda2  = 0.71 GeV^2 — canonical dipole mass-squared; "G_D = [1 + Q^2/(0.71
                            GeV^2)]^{-2} is the dipole form factor" (PDF p. 7,
                            sentence below Table I).

Type designation: Type I. All eight coefficients are globally fitted once to
the full world-data set; no per-experiment or per-cluster refit. LOCAL_FITTABLE
is empty. No fit() function.

Column mapping: Q2_GeV2 (col 1 in released CSV) -> Q^2 in GeV^2 as used in
the paper's Eq. (11) variable tau.
"""

import numpy as np

USED_INPUTS = ["Q2_GeV2"]
PAPER_REF = "summary_formula+dataset_arrington_2007.md"
EQUATION_LOC = "Arrington 2007 Eq. (11), PDF p. 6; coefficients Table I, PDF p. 6"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a1":  3.439,    # Table I, PDF p. 6 — G_E column
    "a2": -1.602,    # Table I, PDF p. 6
    "a3":  0.068,    # Table I, PDF p. 6
    "b1": 15.055,    # Table I, PDF p. 6
    "b2": 48.061,    # Table I, PDF p. 6
    "b3": 99.304,    # Table I, PDF p. 6
    "b4":  0.012,    # Table I, PDF p. 6
    "b5":  8.650,    # Table I, PDF p. 6
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "M_p":     0.938,   # GeV — proton mass (tau definition, PDF p. 6 Sec. III.A)
    "Lambda2": 0.71,    # GeV^2 — dipole scale (PDF p. 7, below Table I)
}

LOCAL_FITTABLE = {}     # Type I — no per-cluster parameters


def predict(
    X: np.ndarray,
    a1: float,
    a2: float,
    a3: float,
    b1: float,
    b2: float,
    b3: float,
    b4: float,
    b5: float,
) -> np.ndarray:
    """Predicted GE_over_GD via Arrington 2007 Eq. (11).

    X: (n, 1) — column Q2_GeV2.
    params: LAW_CONSTANTS keys (a1, a2, a3, b1, b2, b3, b4, b5).
    OTHER_CONSTANTS (M_p, Lambda2) read from module namespace.
    """
    M_p     = OTHER_CONSTANTS["M_p"]
    Lambda2 = OTHER_CONSTANTS["Lambda2"]

    Q2  = np.asarray(X[:, 0], dtype=float)
    tau = Q2 / (4.0 * M_p * M_p)

    num = 1.0 + a1 * tau + a2 * tau**2 + a3 * tau**3
    den = (1.0
           + b1 * tau
           + b2 * tau**2
           + b3 * tau**3
           + b4 * tau**4
           + b5 * tau**5)
    G_E = num / den
    G_D = (1.0 + Q2 / Lambda2) ** (-2.0)
    return G_E / G_D
