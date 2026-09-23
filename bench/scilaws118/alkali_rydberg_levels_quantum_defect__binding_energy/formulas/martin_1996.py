"""Martin & Wiese (1996) extended Ritz quantum-defect formula -- alkali binding energy.

W. C. Martin and W. L. Wiese (1996). Atomic Spectroscopy: A Compendium of Basic
Ideas, Notation, Data, and Formulas. Chapter 10 in G. W. F. Drake (Ed.), Atomic,
Molecular, and Optical Physics Handbook, AIP Press, Woodbury NY. NIST web version
dated 3/3/1999 at https://physics.nist.gov/Pubs/AtSpec/ -- US Government public domain.

Eq. (11) -- Rydberg formula with constant quantum defect (PDF p. 15):

    E_{nl} = Z_c^2 * R_inf / (n - delta)^2

where n is the principal quantum number, delta is the (constant) quantum defect
for the series, Z_c is the core charge (= 1 for all neutral alkalis), and
R_inf = 10 973 731.568 62(9) m^-1 = 109 737.316 cm^-1 (PDF p. 5, eq. 3).

Eq. (12) -- extended Ritz formula for energy-dependent quantum defect (PDF p. 16):

    delta(n) = delta_0 + a / (n - delta_0)^2 + b / (n - delta_0)^4 + ...

Substituting into Eq. (11) produces the full Ritz binding-energy formula.
This module implements a two-Ritz-coefficient variant (delta_0, a, b) that
reduces to the constant-quantum-defect Eq. (11) when a = b = 0.

LAW_CONSTANTS -- empty
----------------------
This is a form-discovery Type II task: the scientific claim is the FUNCTIONAL
FORM (the Ritz quantum-defect hyperbola) plus the per-series quantum defects.
There is no paper-FITTED constant that is invariant across the (element, series)
clusters -- the only cross-cluster invariants here are universal/structural
GIVENS (the Rydberg constant and the core charge), which are OTHER, not LAW.
(Cf. the gold exemplar, where the universal CODATA Coulomb coupling e^2 is OTHER,
and the four-field MANUAL Direction B: a CODATA/universal constant in LAW is
demoted to OTHER -- the SR must not "discover" a value known a priori.)

OTHER_CONSTANTS -- universal / structural givens the formula consumes
---------------------------------------------------------------------
R_inf : 109 737.316 cm^-1 -- the universal CODATA Rydberg constant
        (R_inf = m_e c^2 / 2h). Martin & Wiese (1996) PDF p. 5, eq. (3):
        "R_inf = mec^2/(2h) = 10 973 731.568 62(9) m^-1" = 109 737.3156862 cm^-1,
        rounded to 109 737.316 cm^-1 (< 1 ppm from CODATA 2018). A known
        universal physical constant the formula consumes, NOT a fitted coefficient.
Z_c   : 1 (dimensionless) -- core charge for all neutral alkali atoms. Martin &
        Wiese (1996) PDF p. 15: "Z_c is the charge of the core"; the summary table
        marks it "Read once per element from known atomic data; not fit". A fixed
        structural given (= 1 for Na, K, Rb, Cs), NOT a fitted coefficient.

LOCAL_FITTABLE -- per-cluster (per (element, series)) Ritz quantum-defect params
--------------------------------------------------------------------------------
delta_0 : limiting quantum defect for high-n series members (dimensionless).
          Ritz expansion eq. (12): delta_0 is the limit value of delta for
          large n. Typical range: ~0 (d/f-series) to ~4 (Cs s-series).
a       : first Ritz correction coefficient (dimensionless). Eq. (12).
          Usually positive for core-penetration series (s, p) and negative
          for core-polarization series (d, f). Often |a| < 1.
b       : second Ritz correction coefficient (dimensionless). Eq. (12).
          Small higher-order correction.

Notes:
  - When a = 0 and b = 0, this formula reduces exactly to Eq. (11):
    constant quantum defect.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["n_qm"]
PAPER_REF = "summary_formula_martin_1996.md"
EQUATION_LOC = (
    "Martin & Wiese 1996 Eq. (11), PDF p. 15: E_nl = Z_c^2 * R_inf / (n - delta)^2; "
    "Eq. (12), PDF p. 16: delta = delta_0 + a/(n-delta_0)^2 + b/(n-delta_0)^4."
)

LAW_CONSTANTS = {}
# Universal / structural GIVENS consumed by the form (not fitted; not the SR
# discovery target). Exposed as priors candidates. See docstring + MANUAL §2.
OTHER_CONSTANTS = {
    "R_inf": 109737.316,  # cm^-1; universal CODATA Rydberg constant; PDF p. 5 eq. 3
    "Z_c":   1,           # dimensionless; core charge for neutral alkalis; PDF p. 15
}
LOCAL_FITTABLE = {
    "delta_0": {"init": None},
    "a":       {"init": None},
    "b":       {"init": None},
}


def _delta_ritz(n_qm, delta_0, a, b):
    """Energy-dependent quantum defect via extended Ritz formula (Eq. 12).

    delta(n) = delta_0 + a / (n - delta_0)^2 + b / (n - delta_0)^4
    """
    n_qm = np.asarray(n_qm, dtype=float)
    n_star0 = n_qm - delta_0
    safe = np.where(np.abs(n_star0) > 1e-6, n_star0, 1e-6)
    return delta_0 + a / (safe ** 2) + b / (safe ** 4)


def _binding_energy(n_qm, delta_0, a, b):
    """E_nl = Z_c^2 * R_inf / (n - delta(n))^2 in cm^-1.

    R_inf and Z_c are universal/structural givens (OTHER_CONSTANTS).
    """
    R_inf = OTHER_CONSTANTS["R_inf"]
    Z_c = OTHER_CONSTANTS["Z_c"]
    n_qm = np.asarray(n_qm, dtype=float)
    delta = _delta_ritz(n_qm, delta_0, a, b)
    n_star = n_qm - delta
    n_star = np.where(n_star > 1e-3, n_star, 1e-3)
    return float(Z_c) ** 2 * R_inf / (n_star * n_star)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Fit Ritz quantum-defect parameters (delta_0, a, b) for one cluster.

    Data-derived initialisation:
      delta_0_init: from constant QD approximation --
        median(n - sqrt(Z_c^2 * R_inf / E_bind)).
      a_init = 0.0 (no Ritz correction as starting point).
      b_init = 0.0.
    """
    R_inf = OTHER_CONSTANTS["R_inf"]
    Z_c = OTHER_CONSTANTS["Z_c"]
    n_qm = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    valid = y > 0.0
    if valid.any():
        n_star_est = np.sqrt(float(Z_c) ** 2 * R_inf / y[valid])
        delta0_init = float(np.median(n_qm[valid] - n_star_est))
    else:
        delta0_init = 1.35

    param_lo = [-5.0, -5.0, -5.0]
    param_hi = [ 5.0,  5.0,  5.0]
    delta0_init = float(np.clip(delta0_init, param_lo[0], param_hi[0]))
    p0 = [delta0_init, 0.0, 0.0]

    def residual(p):
        return _binding_energy(n_qm, p[0], p[1], p[2]) - y

    try:
        sol = least_squares(residual, p0,
                            bounds=(param_lo, param_hi),
                            method="trf", max_nfev=4000)
        delta_0, a, b = sol.x
        if not np.all(np.isfinite([delta_0, a, b])):
            raise RuntimeError("non-finite fit")
        return {"delta_0": float(delta_0), "a": float(a), "b": float(b)}
    except Exception:
        return {"delta_0": p0[0], "a": p0[1], "b": p0[2]}


def predict(X: np.ndarray, delta_0: float, a: float, b: float) -> np.ndarray:
    """Extended Ritz binding energy: E_nl = Z_c^2 * R_inf / (n - delta(n))^2.

    X: (n_rows, 1) -- column [n_qm], principal quantum number.
    delta_0, a, b: per-cluster Ritz quantum-defect parameters (LOCAL_FITTABLE).
    R_inf and Z_c are read from OTHER_CONSTANTS (universal/structural givens).
    """
    n_qm = np.asarray(X[:, 0], dtype=float)
    return _binding_energy(n_qm, delta_0, a, b)
