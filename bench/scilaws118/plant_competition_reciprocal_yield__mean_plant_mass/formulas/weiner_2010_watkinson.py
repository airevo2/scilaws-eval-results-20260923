"""Watkinson (1980) reparameterisation of Bleasdale–Nelder as reviewed in Weiner & Freckleton (2010) — mean_plant_mass.

Weiner, J. and Freckleton, R. P. (2010). *Constant Final Yield.*
Annual Review of Ecology, Evolution, and Systematics 41:173–192.
DOI:10.1146/annurev-ecolsys-102209-144642.

Weiner & Freckleton (2010) present the Watkinson (1980) reparameterisation
of the Bleasdale–Nelder family as their Eq. 7 (PDF p. 6):

    w = w_m * (1 + a * N) ** (-b)

where:
  - N   is plant density (plants per unit area, the benchmark input).
  - w_m is the mass of an isolated plant in the absence of competition
        (positive, same units as w). As N → 0, w → w_m.
  - a   is the ecological neighbourhood area — the area an individual
        requires to achieve w_m (units: area per individual = 1/density).
        Determines the density at which competition begins to suppress w.
  - b   is a dimensionless scaling exponent. When b = 1, total yield Y = N*w
        saturates at w_m / a (the Constant Final Yield plateau). When b ≠ 1,
        total yield can increase (b < 1) or decrease (b > 1) beyond the CFY
        density (PDF p. 6: "b is a dimensionless scaling parameter that
        produces CFY when it equals 1").

Relationship to Bleasdale–Nelder (1960) Eq. 3 notation:
  Setting A = a^theta, B = 1 (absorbed into w_m), theta = 1/b recovers
  the Bleasdale–Nelder single-exponent form with a different parameterisation
  of the same functional family (Weiner & Freckleton 2010 §3.1, PDF p. 5–6).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. Weiner & Freckleton (2010) report the functional FORM from Watkinson
(1980); no universal numerical values for w_m, a, or b are published.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
The literal 1 in (1 + a*N) is the algebraic baseline-normalisation factor
from the Watkinson derivation, not an empirical constant.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- w_m : isolated-plant mass (g plant^-1 equivalent; > 0). Sets the
        asymptotic mass at zero density.
- a   : ecological neighbourhood (> 0; units of 1/density). Controls
        the density at which competition becomes significant.
- b   : scaling exponent (> 0; dimensionless). b = 1 gives CFY;
        b > 1 → steeper per-plant mass decline.

init = None on all three: fit() constructs a data-derived start point.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["density"]
PAPER_REF = "summary_formula_weiner_2010.md"
EQUATION_LOC = (
    "Weiner & Freckleton 2010, Annu. Rev. Ecol. Evol. Syst. 41:173-192, "
    "PDF p. 6, Eq. 7: w = w_m * (1 + a*N)**(-b). "
    "Originally from Watkinson (1980) as cited in that review."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "w_m": {"init": None},
    "a":   {"init": None},
    "b":   {"init": None},
}


def _predict_core(N, w_m, a, b):
    """Core formula: w = w_m * (1 + a*N)**(-b)."""
    N = np.asarray(N, dtype=float)
    bracket = np.clip(1.0 + a * np.clip(N, 0.0, None), 1e-300, None)
    return w_m * bracket ** (-b)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear LS fit of Watkinson (1980) / Weiner (2010) Eq. 7.

    Data-derived initialisation:
      w_m0 = max(y)     — the largest observed mass is a lower bound on the
                           isolated-plant mass.
      b0   = 1.0        — the CFY special case; a neutral physically
                           motivated start (PDF p. 6).
      a0   = (w_m0/median(y))^(1/b0) - 1) / median(N)
                        — inferred so that at median density, the formula
                           predicts roughly the median observed mass.

    Single-start trust-region is robust because the form is well-conditioned
    on a positive density series with w_m, a, b all > 0.
    """
    N = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    w_m0 = float(np.max(y)) if np.any(y > 0) else 1.0
    b0 = 1.0
    med_N = float(np.median(N[N > 0])) if np.any(N > 0) else 1.0
    med_y = float(np.median(y[y > 0])) if np.any(y > 0) else w_m0 * 0.5
    ratio = w_m0 / med_y if med_y > 0 else 2.0
    # w_m0 / med_y = (1 + a*med_N)^b0  =>  a0 = (ratio^(1/b0) - 1) / med_N
    a0 = max(1e-10, (ratio ** (1.0 / b0) - 1.0) / (med_N + 1e-30))
    if not np.isfinite(a0) or a0 <= 0:
        a0 = 1.0 / (med_N + 1e-30)

    p0 = [w_m0, a0, b0]

    lo = [1e-10, 1e-12, 0.01]
    hi = [1e12,  1e6,   20.0]
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
    """Watkinson (1980) / Weiner (2010) Eq. 7 mean plant mass.

    w = w_m * (1 + a * N) ** (-b)

    X : (n, 1) — column [density] (plants per unit area).
    Returns w (mean plant mass, same units as training target).
    """
    N = np.asarray(X[:, 0], dtype=float)
    return _predict_core(N, w_m, a, b)
