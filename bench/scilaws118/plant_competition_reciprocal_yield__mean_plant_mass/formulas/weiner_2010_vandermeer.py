"""Vandermeer (1984) variant of the CFY formula as reviewed in Weiner & Freckleton (2010) — mean_plant_mass.

Weiner, J. and Freckleton, R. P. (2010). *Constant Final Yield.*
Annual Review of Ecology, Evolution, and Systematics 41:173–192.
DOI:10.1146/annurev-ecolsys-102209-144642.

Weiner & Freckleton (2010) present the Vandermeer (1984) form as
their Eq. 8 (PDF p. 6):

    w = w_m * (1 + a * N**b) ** (-1)

where:
  - N   is plant density (plants per unit area, the benchmark input).
  - w_m is the mass of an isolated plant in the absence of competition
        (positive; same units as w). As N → 0, w → w_m.
  - a   is the per-capita competitive effect coefficient (> 0; units of
        density^(-b) so that a*N^b is dimensionless).
  - b   is the density-scaling exponent (> 0; dimensionless). This differs
        from the Watkinson (1980) Eq. 7 where b is the outer power on the
        bracket; here b appears as the inner exponent on N.

Relationship to Watkinson (1980) / Weiner Eq. 7:
  Vandermeer's Eq. 8 and Watkinson's Eq. 7 coincide at b = 1:
    w = w_m / (1 + a*N)
  which is both (1+aN)^(-1) and (1+aN^1)^(-1). For b ≠ 1 the two
  equations differ at intermediate densities; Weiner & Freckleton (2010,
  PDF p. 6) note both forms and treat them as distinct models.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. Weiner & Freckleton (2010) review the functional FORM from Vandermeer
(1984); no universal numerical values for w_m, a, or b are published.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
The literal 1 in (1 + a*N^b) is the algebraic baseline-normalisation
factor; not an empirical constant.
The outer exponent -1 on the bracket is the defining structural choice of
the Vandermeer form (vs. -b in the Watkinson form).

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- w_m : isolated-plant mass (> 0; g plant^-1 equivalent).
- a   : competitive effect coefficient (> 0).
- b   : density exponent inside bracket (> 0; dimensionless).

init = None on all three: fit() constructs a data-derived start point.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["density"]
PAPER_REF = "summary_formula_weiner_2010.md"
EQUATION_LOC = (
    "Weiner & Freckleton 2010, Annu. Rev. Ecol. Evol. Syst. 41:173-192, "
    "PDF p. 6, Eq. 8: w = w_m * (1 + a*N**b)**(-1). "
    "Originally from Vandermeer (1984) as cited in that review."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "w_m": {"init": None},
    "a":   {"init": None},
    "b":   {"init": None},
}


def _predict_core(N, w_m, a, b):
    """Core formula: w = w_m / (1 + a * N**b)."""
    N = np.asarray(N, dtype=float)
    bracket = np.clip(1.0 + a * np.power(np.clip(N, 0.0, None), b), 1e-300, None)
    return w_m / bracket


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear LS fit of Vandermeer (1984) / Weiner (2010) Eq. 8.

    Data-derived initialisation:
      w_m0 = max(y)  — largest observed mass approximates isolated-plant mass.
      b0   = 1.0     — linear density scaling; a neutral starting point that
                        coincides with the Watkinson b=1 / Michaelis-Menten
                        special case.
      a0   = (w_m0/median(y) - 1) / median(N)**b0
                     — inferred so the formula matches median observed mass
                        at the median observed density.
    """
    N = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    w_m0 = float(np.max(y)) if np.any(y > 0) else 1.0
    b0 = 1.0
    med_N = float(np.median(N[N > 0])) if np.any(N > 0) else 1.0
    med_y = float(np.median(y[y > 0])) if np.any(y > 0) else w_m0 * 0.5
    ratio = w_m0 / med_y if med_y > 0 else 2.0
    # w_m0 / med_y = 1 + a * med_N^b0  =>  a0 = (ratio - 1) / med_N^b0
    a0 = max(1e-10, (ratio - 1.0) / (med_N ** b0 + 1e-30))
    if not np.isfinite(a0) or a0 <= 0:
        a0 = 1.0 / (med_N + 1e-30)

    p0 = [w_m0, a0, b0]

    lo = [1e-10, 1e-12, 0.01]
    hi = [1e12,  1e8,   10.0]
    p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]

    def residual(params):
        w_m, a, b = params
        return _predict_core(N, w_m, a, b) - y

    try:
        sol = least_squares(residual, p0, bounds=(lo, hi),
                            method="trf", max_nfev=6000)
        w_m, a, b = sol.x
        if not np.all(np.isfinite([w_m, a, b])):
            raise RuntimeError("non-finite solution")
        return {"w_m": float(w_m), "a": float(a), "b": float(b)}
    except Exception:                          # noqa: BLE001
        return {"w_m": float(p0[0]), "a": float(p0[1]), "b": float(p0[2])}


def predict(X: np.ndarray, w_m: float, a: float, b: float) -> np.ndarray:
    """Vandermeer (1984) / Weiner (2010) Eq. 8 mean plant mass.

    w = w_m * (1 + a * N**b) ** (-1)  =  w_m / (1 + a * N**b)

    X : (n, 1) — column [density] (plants per unit area).
    Returns w (mean plant mass, same units as training target).
    """
    N = np.asarray(X[:, 0], dtype=float)
    return _predict_core(N, w_m, a, b)
