"""Zwietering (1990) modified Gompertz bacterial growth curve — OD600.

Zwietering, M. H., Jongenburger, I., Rombouts, F. M., & van 't Riet, K.
(1990). Modeling of the bacterial growth curve. *Applied and Environmental
Microbiology* 56(6):1875-1881. DOI:10.1128/aem.56.6.1875-1881.1990.

The modified Gompertz equation (Eq. 11, PDF p. 3) is:

    y(t) = A * exp{ -exp[ (mu_m * e / A) * (lambda - t) + 1 ] }

where e = exp(1) ≈ 2.71828 is Euler's number (structural constant, footnote
to Table 2, p. 2). Three biologically interpretable per-curve parameters:

    A      — asymptote: y → A as t → ∞ (OD600 units in benchmark)
    mu_m   — maximum specific growth rate; exact slope of y(t) at the
             inflection point (h^-1)
    lambda — lag time; x-intercept of the tangent through the inflection
             point (h)

These identities are constructed exactly by algebraic reparameterisation
(Eqs. 5–8, PDF p. 2); they are not approximations.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The modified Gompertz FORM (Eq. 11) is the scientific claim. A,
mu_m, and lambda are per-curve material parameters; e = exp(1) is a
universal mathematical constant, not a tunable law constant.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
e = exp(1): Euler's number, hard-wired into the reparameterisation;
appears in the footnote to Table 2, p. 2 of Zwietering 1990. It is a
universal mathematical constant (not a tunable parameter), so it is kept
as a Python literal `np.e` in the formula body.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- A      : asymptotic OD600 (> 0). Initialised from the 90th percentile
           of the observed OD600 values (plateau estimate).
- mu_m   : maximum specific growth rate (h^-1, > 0). Initialised from
           the steepest finite-difference slope seen in the curve.
- lambda : lag time (h, >= 0). Initialised from the x-intercept of the
           tangent at the estimated inflection region.

All inits are deterministic and data-derived (no multi-start needed: the
modified Gompertz is well-conditioned on 97-point OD600 data with clear
lag/exponential/stationary phases).
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["time_h"]
PAPER_REF = "summary_formula_zwietering_1990.md"
EQUATION_LOC = (
    "Zwietering 1990 Eq. 11, PDF p. 3 — y = A*exp{-exp[(mu_m*e/A)*(lambda-t)+1]}; "
    "e = exp(1) from footnote to Table 2, PDF p. 2."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "A":      {"init": None},
    "mu_m":   {"init": None},
    "lambda_": {"init": None},
}


def _gompertz(t, A, mu_m, lambda_):
    """Modified Gompertz OD600 prediction (Zwietering 1990 Eq. 11)."""
    arg = np.clip((mu_m * np.e / A) * (lambda_ - t) + 1.0, -700.0, 700.0)
    return A * np.exp(-np.exp(arg))


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the modified Gompertz form.

    Deterministic, data-derived start (following Zwietering 1990 p. 3
    initialisation heuristics):
      A0      = 90th percentile of observed OD600 (plateau estimate).
      mu_m0   = maximum finite-difference slope (steepest-ascent heuristic).
      lambda0 = x-intercept of the tangent at the estimated inflection point
                (tangent = y_inf - mu_m0 * (t_inf - lambda0) = 0).

    Single-start is sufficient: 97 evenly-spaced time points covering the
    full growth curve provide enough curvature information for a smooth,
    well-conditioned loss surface.
    """
    t = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # --- A0: asymptote estimate ---
    A0 = float(np.percentile(y, 90))
    if not (np.isfinite(A0) and A0 > 0):
        A0 = float(np.max(y)) if np.max(y) > 0 else 0.5

    # --- mu_m0: maximum slope from finite differences ---
    if len(t) >= 2:
        order = np.argsort(t)
        t_s = t[order]
        y_s = y[order]
        dt = np.diff(t_s)
        dy = np.diff(y_s)
        slopes = dy / np.where(dt > 0, dt, 1e-12)
        mu_m0 = float(np.max(slopes))
        if not (np.isfinite(mu_m0) and mu_m0 > 0):
            mu_m0 = A0 / 10.0
        # index of steepest slope segment
        idx_max = int(np.argmax(slopes))
        t_inf = float(0.5 * (t_s[idx_max] + t_s[idx_max + 1]))
        y_inf = float(0.5 * (y_s[idx_max] + y_s[idx_max + 1]))
        # tangent x-intercept: lambda0 = t_inf - y_inf / mu_m0
        lambda0 = t_inf - y_inf / mu_m0
    else:
        mu_m0 = A0 / 10.0
        lambda0 = 0.0

    if not np.isfinite(lambda0):
        lambda0 = 0.0
    lambda0 = max(lambda0, 0.0)

    # bounds (fit-procedure config, not formula constants)
    #             A         mu_m    lambda_
    param_lo = [1e-4,     1e-6,    0.0   ]
    param_hi = [10.0,     5.0,     200.0 ]

    p0 = [A0, mu_m0, lambda0]
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, param_lo, param_hi)]

    def residual(p):
        return _gompertz(t, p[0], p[1], p[2]) - y

    try:
        sol = least_squares(residual, p0, bounds=(param_lo, param_hi),
                            method="trf", max_nfev=4000)
        A, mu_m, lambda_ = sol.x
        if not np.all(np.isfinite([A, mu_m, lambda_])):
            raise RuntimeError("non-finite fit")
        return {"A": float(A), "mu_m": float(mu_m), "lambda_": float(lambda_)}
    except Exception:                                      # noqa: BLE001
        return {"A": float(p0[0]), "mu_m": float(p0[1]), "lambda_": float(p0[2])}


def predict(X: np.ndarray, A: float, mu_m: float, lambda_: float) -> np.ndarray:
    """Modified Gompertz OD600 as a function of time.

    X: (n, 1) — column [time_h].
    Returns: (n,) array of predicted OD600 values.
    """
    t = np.asarray(X[:, 0], dtype=float)
    return _gompertz(t, A, mu_m, lambda_)
