"""Duflo-Zuker (1995) asymptotic / liquid-drop limit (Table III, 6p column).

Duflo & Zuker, Phys. Rev. C 52 (1995) R23-R27. The full DZ formula is a
shell-model-inspired 28-parameter algorithm; this module implements the
analytically tractable asymptotic-LD limit the paper uses as its own
comparison baseline — the "6p" column of Table III (PDF p. 5):

    E(Z, N) = a_vol  * A**(4/3) / R
            + a_surf * A**(4/3) / R**2
            + a_sym  * T2 / (A**(2/3) * R)
            + a_ssy  * T2 / (A**(2/3) * R**2)
            + a_pair * V_p
            + a_coul * V_c

with the operators (Eq. 4, PDF p. 2 and Table I, p. 3):

    T   = |N - Z| / 2 ;   T2 = 4 T (T + 1)
    R_c = A**(1/3) * (1 - (T/A)**2)
    R   = R_c**2 / A**(1/3)
    V_p = -mod(N,2) - mod(Z,2)               (= pair - 1, from the `pair` column)
    V_c = (-Z(Z-1) + 0.76 * (Z(Z-1))**(2/3)) / R_c

The benchmark target is BE/A = E(Z, N) / A.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
The six 6p coefficients of Duflo-Zuker (1995) Table III, PDF p. 5
(fitted to AME1993; the 6p model has 2499 keV rms vs. measured masses):

    a_vol  =  15.42    a_surf = -17.56    a_sym  = -33.65
    a_ssy  =  50.91    a_pair =   5.18    a_coul =   0.699

These are the paper's own fitted values. AME1993 → AME2020 transfer is
expected to be good (nuclear masses are stable; AME2020 only adds more
measured nuclei).

OTHER_CONSTANTS — structural operator-definition constants
----------------------------------------------------------
- vc_coef = 0.76 : fixed coefficient inside the V_c operator
  (Duflo-Zuker Table I). Not one of the six fitted model coefficients
  and not a discovery target — declared so every constant the formula
  consumes is countable.

Type designation: Type I — LOCAL_FITTABLE empty, no fit().

Mapping paper notation -> released CSV columns: Z, N, A direct;
NZ_diff -> T = |N-Z|/2; pair (-1/0/+1) -> V_p = pair - 1.
"""

import numpy as np

USED_INPUTS = ["Z", "N", "A", "NZ_diff", "pair"]
PAPER_REF = "summary_formula_duflo_1995.md"
EQUATION_LOC = "Duflo-Zuker 1995 Table III 6p column, PDF p. 5 (operators: Eq. 4 + Table I)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a_vol":   15.42,
    "a_surf": -17.56,
    "a_sym":  -33.65,
    "a_ssy":   50.91,
    "a_pair":   5.18,
    "a_coul":   0.699,
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "vc_coef": 0.76,   # fixed coefficient in the V_c operator (Duflo-Zuker Table I)
}
LOCAL_FITTABLE = {}    # Type I


def predict(X: np.ndarray, a_vol: float, a_surf: float, a_sym: float,
            a_ssy: float, a_pair: float, a_coul: float) -> np.ndarray:
    """BE/A (MeV) under the Duflo-Zuker 6p asymptotic model.

    X: (n, 5) — columns Z, N, A, NZ_diff, pair.
    """
    Z = np.asarray(X[:, 0], dtype=float)
    A = np.asarray(X[:, 2], dtype=float)
    NZ_diff = np.asarray(X[:, 3], dtype=float)
    pair = np.asarray(X[:, 4], dtype=float)

    T = 0.5 * np.abs(NZ_diff)
    T2 = 4.0 * T * (T + 1.0)
    A13 = np.power(A, 1.0 / 3.0)
    Rc = A13 * (1.0 - (T / A) ** 2)
    R = Rc * Rc / A13
    A43 = np.power(A, 4.0 / 3.0)
    A23 = np.power(A, 2.0 / 3.0)

    vc_coef = OTHER_CONSTANTS["vc_coef"]
    Vp = pair - 1.0
    z_zm1 = Z * (Z - 1.0)
    Vc = (-z_zm1 + vc_coef * np.power(z_zm1, 2.0 / 3.0)) / Rc

    E = (a_vol * A43 / R
         + a_surf * A43 / (R * R)
         + a_sym * T2 / (A23 * R)
         + a_ssy * T2 / (A23 * R * R)
         + a_pair * Vp
         + a_coul * Vc)
    return E / A
