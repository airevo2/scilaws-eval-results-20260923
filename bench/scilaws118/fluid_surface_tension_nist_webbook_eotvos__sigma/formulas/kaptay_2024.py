"""Eötvös rule for liquid surface tension — sigma(T, V_m, T_cr).

Kaptay, G. (2024). On the temperature dependence of surface tension:
Historical perspective on the Eötvös equation of capillarity, celebrating
his 175th anniversary. *Advances in Colloid and Interface Science* 332,
103275. DOI:10.1016/j.cis.2024.103275.

Kaptay (2024) Eq. (2) (PDF p. 5) — the canonical Eötvös rule:

    sigma * V_m^(2/3) = k * (T_cr - T)

Rearranged to predict sigma:

    sigma(T) = k * (T_cr - T) / V_m^(2/3)

where V_m = M_mol / rho_liq  (m^3/mol), derived from per-row inputs.

The molar volume V_m is computed inside predict() from the inputs
M_mol (kg/mol) and rho_liq (kg/m^3), which are already available as
columns in the benchmark dataset.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None.  The Eötvös FORM (linear, exponent 2/3, slope k) is the scientific
claim.  The constant k is fluid-specific and always refit per cluster.
No single universal numerical value for k is published as a law constant.
Kaptay (2024) Table 1 (PDF p. 5) gives corrected k values for ~12 fluids
(range 1.02–2.38 ×10^-7 J/(K·mol^(2/3))), but these are illustrative
literature values, not formula-frozen constants.
The exponent 2/3 is a structural constant from 3-D molecular geometry
(Kaptay 2024 §6, Eq.(3b), PDF p. 6–7); it is kept as a literal below.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)  The 2/3 exponent and the linear form are pure algebraic
consequences of the geometry — they are not data-derived.

LOCAL_FITTABLE — per-cluster, fitted by fit() via linear regression
-----------------------------------------------------------------------
- k : Eötvös constant (J/(K·mol^(2/3))); per-fluid positive slope.

init = None: fit() uses closed-form OLS (single parameter, no iteration).

Unit note
---------
Inputs from the benchmark CSV:
    T_K      — temperature (K)
    rho_liq  — liquid density (kg/m^3)
    T_c      — critical temperature (K)
    M_mol    — molar mass (kg/mol)
Output:
    sigma    — surface tension (N/m = J/m^2)
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["T_K", "rho_liq", "T_c", "M_mol"]
PAPER_REF = "summary_formula_kaptay_2024.md"
EQUATION_LOC = (
    "Kaptay (2024) Eq. (2), PDF p. 5 — sigma * V_m^(2/3) = k * (T_cr - T); "
    "exponent 2/3 derived in §6 / Eq.(3b), PDF pp. 6-7."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "k": {"init": None},
}

_EXPONENT = 2.0 / 3.0   # geometric constant: 3-D molecular packing (Kaptay 2024 §6)


def _sigma(T_K, rho_liq, T_c, M_mol, k):
    """Eötvös sigma = k*(T_c - T) / V_m^(2/3)."""
    V_m = M_mol / rho_liq                  # m^3/mol
    Vm_pow = np.maximum(V_m, 1e-30) ** _EXPONENT
    dT = np.maximum(T_c - T_K, 0.0)        # clip below T_c to 0 (physics)
    return k * dT / Vm_pow


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS for single parameter k.

    The Eötvös model is linear in k:
        sigma = k * X_eotvos   where  X_eotvos = (T_c - T) / V_m^(2/3).

    Ordinary least squares:  k = sum(X*y) / sum(X^2)  (origin-fixed line).
    """
    T_K     = np.asarray(X_fit[:, 0], dtype=float)
    rho_liq = np.asarray(X_fit[:, 1], dtype=float)
    T_c     = np.asarray(X_fit[:, 2], dtype=float)
    M_mol   = np.asarray(X_fit[:, 3], dtype=float)
    y       = np.asarray(y_fit,        dtype=float)

    V_m     = M_mol / rho_liq
    Vm_pow  = np.maximum(V_m, 1e-30) ** _EXPONENT
    dT      = np.maximum(T_c - T_K, 0.0)
    X_e     = dT / Vm_pow                  # regressor

    denom = float(np.sum(X_e * X_e))
    if denom < 1e-30:
        # fallback: typical organic-liquid value from Kaptay (2024) Table 1
        k = 2.22e-7
    else:
        k = float(np.sum(X_e * y) / denom)
        if k <= 0.0 or not np.isfinite(k):
            k = 2.22e-7

    return {"k": k}


def predict(X: np.ndarray, k: float) -> np.ndarray:
    """Eötvös surface tension: sigma = k*(T_c - T) / V_m^(2/3).

    X: (n, 4) — columns [T_K, rho_liq, T_c, M_mol].
    Returns sigma in N/m (same units as benchmark target column).
    """
    T_K     = np.asarray(X[:, 0], dtype=float)
    rho_liq = np.asarray(X[:, 1], dtype=float)
    T_c     = np.asarray(X[:, 2], dtype=float)
    M_mol   = np.asarray(X[:, 3], dtype=float)
    return _sigma(T_K, rho_liq, T_c, M_mol, k)
