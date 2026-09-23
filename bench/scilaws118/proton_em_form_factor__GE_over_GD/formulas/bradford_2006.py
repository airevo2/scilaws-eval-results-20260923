"""Bradford-Bodek-Budd-Arrington (2006) BBBA05 parameterisation of G_E^p / G_D.

Citation: Bradford, Bodek, Budd & Arrington, Nucl. Phys. B (Proc. Suppl.) 159,
127-132 (2006), arXiv:hep-ex/0602017. Primary formula: Eq. (6), PDF p. 4.
Coefficients: Table 1, PDF p. 4 (preferred fit: G_en > 0, d/u = 0.2).

Formula (BBBA05, Eq. 6, PDF p. 4)
------------------------------------
    G_Ep(Q^2) = (a0 + a1 * tau) / (1 + b1*tau + b2*tau^2 + b3*tau^3)

    tau = Q^2 / (4 * M_p^2)

    G_D(Q^2) = (1 + Q^2 / Lambda2)^(-2)       [Eq. (2), PDF p. 2]

    Target: GE_over_GD(Q^2) = G_Ep(Q^2) / G_D(Q^2)

a0 = 1 is fixed to enforce the correct low-Q^2 limit and was not varied during
the fits (Table 1 caption, PDF p. 4). Fit excludes Rosenbluth G_Ep above
Q^2 > 1 GeV^2 due to two-photon exchange contamination (Sec. 5, PDF p. 3).
Parameterisation claimed valid up to Q^2 ~ 18 GeV^2 (PDF p. 6, Sec. 6).

LAW_CONSTANTS — Table 1, PDF p. 4 (G_Ep row, preferred fit Gen > 0, d/u = 0.2)
---------------------------------------------------------------------------------
    a1 = -0.0578
    b1 =  11.1
    b2 =  13.6
    b3 =  33.0

OTHER_CONSTANTS — fixed structural / universal values
------------------------------------------------------
    a0     = 1.0   — Table 1 caption PDF p. 4: "a0 was used to ensure the correct
                     low Q^2 limit and was not varied during the fits". Structural
                     normalisation constant; not a LAW scientific claim.
    M_p    = 0.938  GeV  — nucleon mass (tau definition, PDF p. 1 Eq. (1)).
    Lambda2 = 0.71  GeV^2 — dipole scale; "Λ^2 = 0.71 GeV^2" in Eq. (2), PDF p. 2.

Type designation: Type I. Four BBBA05 coefficients (a1, b1, b2, b3) are
globally fitted once to the world data; a0 is frozen by construction. No
per-experiment refit. LOCAL_FITTABLE is empty. No fit() function.

Column mapping: Q2_GeV2 (released CSV col 1) -> Q^2 in GeV^2 (paper Eq. 6).

Caveat: Bradford 2006 excludes Rosenbluth G_Ep data above Q^2 > 1 GeV^2 to
avoid two-photon exchange contamination. The released test set extends to
Q^2 = 5.85 GeV^2 (Arrington 2007 Table II, post-Phase-2 data swap), so this
baseline is extrapolating beyond its fit domain above Q^2 ~ 1 GeV^2.
"""

import numpy as np

USED_INPUTS = ["Q2_GeV2"]
PAPER_REF = "summary_formula_bradford_2006.md"
EQUATION_LOC = "Bradford 2006 Eq. (6), PDF p. 4; G_Ep coefficients Table 1, PDF p. 4"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a1": -0.0578,   # Table 1, PDF p. 4 — G_Ep row
    "b1":  11.1,     # Table 1, PDF p. 4
    "b2":  13.6,     # Table 1, PDF p. 4
    "b3":  33.0,     # Table 1, PDF p. 4
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "a0":     1.0,    # fixed normalisation (Table 1 caption, PDF p. 4); not scored
    "M_p":    0.938,  # GeV — nucleon mass (PDF p. 1)
    "Lambda2": 0.71,  # GeV^2 — dipole scale (Eq. 2, PDF p. 2)
}

LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters


def predict(
    X: np.ndarray,
    a1: float,
    b1: float,
    b2: float,
    b3: float,
) -> np.ndarray:
    """Predicted GE_over_GD via Bradford 2006 BBBA05 Eq. (6).

    X: (n, 1) — column Q2_GeV2.
    params: LAW_CONSTANTS keys (a1, b1, b2, b3).
    OTHER_CONSTANTS (a0, M_p, Lambda2) read from module namespace.
    """
    a0      = OTHER_CONSTANTS["a0"]
    M_p     = OTHER_CONSTANTS["M_p"]
    Lambda2 = OTHER_CONSTANTS["Lambda2"]

    Q2  = np.asarray(X[:, 0], dtype=float)
    tau = Q2 / (4.0 * M_p * M_p)

    num  = a0 + a1 * tau
    den  = 1.0 + b1 * tau + b2 * tau**2 + b3 * tau**3
    G_Ep = num / den
    G_D  = (1.0 + Q2 / Lambda2) ** (-2.0)
    return G_Ep / G_D
