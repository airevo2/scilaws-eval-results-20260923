"""Zwietering (1990) modified logistic bacterial growth curve — OD600.

Zwietering, M. H., Jongenburger, I., Rombouts, F. M., & van 't Riet, K.
(1990). Modeling of the bacterial growth curve. *Applied and Environmental
Microbiology* 56(6):1875-1881. DOI:10.1128/aem.56.6.1875-1881.1990.

The modified logistic equation (Table 2, PDF p. 2) is:

    y(t) = A / { 1 + exp[ (4*mu_m/A) * (lambda - t) + 2 ] }

Three biologically interpretable per-curve parameters (same as modified
Gompertz): A (asymptote, OD600 units), mu_m (max specific growth rate,
h^-1), lambda (lag time, h). The reparameterisation guarantees the same
tangent-slope and tangent-intercept identities as the Gompertz variant
(Eqs. 5-8, PDF p. 2).

The factors 4 and 2 in the exponent are algebraic structural constants
arising from differentiating the logistic function and solving for the
inflection-point reparameterisation (Theory, p. 2); they are not
paper-published tunable constants.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The factors 4 and 2 in (4*mu_m/A)*(lambda-t) + 2 are algebraic
structural constants from the inflection-point reparameterisation of the
symmetric logistic (Theory, PDF p. 2).

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- A      : asymptotic OD600 (> 0).
- mu_m   : maximum specific growth rate (h^-1, > 0).
- lambda_: lag time (h, >= 0).

init = None on all: fit() builds deterministic, data-derived starts
using the same heuristics as the modified Gompertz module.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["time_h"]
PAPER_REF = "summary_formula_zwietering_1990.md"
EQUATION_LOC = (
    "Zwietering 1990 Table 2, PDF p. 2 — modified logistic: "
    "y = A / {1 + exp[(4*mu_m/A)*(lambda-t) + 2]}."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "A":       {"init": None},
    "mu_m":    {"init": None},
    "lambda_": {"init": None},
}


def _logistic(t, A, mu_m, lambda_):
    """Modified logistic OD600 prediction (Zwietering 1990 Table 2)."""
    arg = np.clip((4.0 * mu_m / A) * (lambda_ - t) + 2.0, -700.0, 700.0)
    return A / (1.0 + np.exp(arg))


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the modified logistic form.

    Deterministic, data-derived start (same heuristics as the Gompertz
    module, which follow Zwietering 1990 p. 3):
      A0      = 90th percentile of observed OD600.
      mu_m0   = maximum finite-difference slope.
      lambda0 = tangent x-intercept at the estimated inflection region.
    """
    t = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # --- A0: asymptote estimate ---
    A0 = float(np.percentile(y, 90))
    if not (np.isfinite(A0) and A0 > 0):
        A0 = float(np.max(y)) if np.max(y) > 0 else 0.5

    # --- mu_m0 and lambda0 from finite differences ---
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
        idx_max = int(np.argmax(slopes))
        t_inf = float(0.5 * (t_s[idx_max] + t_s[idx_max + 1]))
        y_inf = float(0.5 * (y_s[idx_max] + y_s[idx_max + 1]))
        lambda0 = t_inf - y_inf / mu_m0
    else:
        mu_m0 = A0 / 10.0
        lambda0 = 0.0

    if not np.isfinite(lambda0):
        lambda0 = 0.0
    lambda0 = max(lambda0, 0.0)

    # bounds
    #             A         mu_m    lambda_
    param_lo = [1e-4,     1e-6,    0.0   ]
    param_hi = [10.0,     5.0,     200.0 ]

    p0 = [A0, mu_m0, lambda0]
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, param_lo, param_hi)]

    def residual(p):
        return _logistic(t, p[0], p[1], p[2]) - y

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
    """Modified logistic OD600 as a function of time.

    X: (n, 1) — column [time_h].
    Returns: (n,) array of predicted OD600 values.
    """
    t = np.asarray(X[:, 0], dtype=float)
    return _logistic(t, A, mu_m, lambda_)
