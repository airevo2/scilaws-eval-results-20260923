"""Weizsäcker (1935) semi-empirical mass formula (SEMF, liquid-drop model).

Weizsäcker, Z. Phys. 96 (1935) 431-458, Eq. (51), PDF p. 24. The total
binding energy is the sum of a hyperbolic volume + asymmetry term, a
surface-tension reduction, and a Coulomb term:

    E(Z, N) = (-sqrt(alpha**2 + beta**2)
               + sqrt(alpha**2 + beta**2 * (Z-N)**2 / (Z+N)**2))
              * ((Z + N - 1) - gamma * (Z + N - 1)**(2/3))
            + (3 * e2 / (r0 * (Z + N)**(1/3)))
              * (1.0 - delta * abs(Z - N) / (Z + N))
              * (Z**2 / 5.0 - (Z / 2.0)**(4.0 / 3.0))

The benchmark target is BE/A = E(Z, N) / A.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
The five Method-II coefficients of Weizsäcker (1935) Tabelle 3, PDF p. 24
("Methode II"):

    alpha = 2.6,  beta = 18.4,  gamma = 1.07,  delta = 1.1,  r0 = 0.42

These are the paper's own calibrated values — used as the scientific
claim, not refit on AME2020.

OTHER_CONSTANTS — universal physics factors
-------------------------------------------
- e2 = 1.4399764 MeV*fm : the Coulomb coupling e^2 (CODATA fine-structure
  value). A universal constant, not a discovery target.

Type designation: Type I — each nucleus is an independent row, no cluster
structure, LOCAL_FITTABLE empty, no fit().

Mapping to the released CSV columns: Z -> col `Z`, N -> col `N`,
A -> col `A`. Eq. (51) contains no explicit pairing term (Weizsäcker §4
handles even-odd staggering by interpolation), so the `pair` column is
not consumed.

Sign convention: Weizsäcker (footnote 5, PDF p. 1) takes binding energy
as positive; the volume-surface contribution is negated to match the
benchmark's BE/A > 0 convention.
"""

import numpy as np

USED_INPUTS = ["Z", "N", "A"]
PAPER_REF = "summary_formula_weizsacker_1935.md"
EQUATION_LOC = "Weizsäcker 1935 Eq. 51, PDF p. 24 (Tabelle 3 Methode II coefficients)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "alpha": 2.6,      # MeV-equiv; Tabelle 3 Methode II
    "beta":  18.4,     # MeV-equiv
    "gamma": 1.07,
    "delta": 1.1,
    "r0":    0.42,     # fm
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "e2": 1.4399764,   # MeV*fm — Coulomb coupling e^2 (CODATA)
}
LOCAL_FITTABLE = {}    # Type I — no per-cluster parameters


def predict(X: np.ndarray, alpha: float, beta: float, gamma: float,
            delta: float, r0: float) -> np.ndarray:
    """BE/A (MeV) for each nucleus under the Weizsäcker 1935 SEMF.

    X: (n, 3) — columns Z, N, A.
    """
    e2 = OTHER_CONSTANTS["e2"]
    Z = np.asarray(X[:, 0], dtype=float)
    N = np.asarray(X[:, 1], dtype=float)
    A = np.asarray(X[:, 2], dtype=float)

    NZ_diff = Z - N
    asym = NZ_diff / A
    eps_sym = np.sqrt(alpha * alpha + beta * beta * asym * asym)
    eps_iso = np.sqrt(alpha * alpha + beta * beta)
    Am1 = A - 1.0
    surface_factor = Am1 - gamma * np.power(Am1, 2.0 / 3.0)
    E_VO = (-eps_iso + eps_sym) * surface_factor

    coul_pref = 3.0 * e2 / (r0 * np.power(A, 1.0 / 3.0))
    coul_iso = 1.0 - delta * np.abs(NZ_diff) / A
    coul_z = Z * Z / 5.0 - np.power(Z / 2.0, 4.0 / 3.0)
    E_C = coul_pref * coul_iso * coul_z

    return -(E_VO + E_C) / A
