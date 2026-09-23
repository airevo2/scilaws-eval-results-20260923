"""Zwietering (1990) modified Richards bacterial growth curve — OD600.

Zwietering, M. H., Jongenburger, I., Rombouts, F. M., & van 't Riet, K.
(1990). Modeling of the bacterial growth curve. *Applied and Environmental
Microbiology* 56(6):1875-1881. DOI:10.1128/aem.56.6.1875-1881.1990.

The modified Richards equation (Table 2, PDF p. 2) is:

    y(t) = A * { 1 + v * exp(1+v)
                   * exp[ (mu_m/A) * (1+v)^(1+1/v) * (lambda - t) ] }^(-1/v)

where v is a dimensionless shape parameter controlling sigmoid asymmetry
(see also the note: "the modified Stannard equation reduces to the same
expression", PDF p. 3 para. 2). Four biologically interpretable parameters:
A (asymptote), mu_m (max growth rate, h^-1), lambda (lag time, h), v (shape).

The paper explicitly recommends three-parameter models (Gompertz or logistic)
over the Richards form when statistically sufficient (p. 7: "the three-
parameter solution is more stable since the parameters are less correlated").
The Richards variant is included as the four-parameter generalisation baseline.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The expression (1+v)^(1+1/v) and the factor v*exp(1+v) derive
from differentiating the Richards model and solving for the inflection-point
reparameterisation (Theory, p. 2); they are structural algebraic identities,
not tunable constants.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- A      : asymptotic OD600 (> 0).
- mu_m   : maximum specific growth rate (h^-1, > 0).
- lambda_: lag time (h, >= 0).
- v      : shape / asymmetry parameter (dimensionless, > 0). Per Zwietering
           p. 7: "v is difficult to explain biologically"; bounded away from
           0 to keep the exponent (1+1/v) finite.

init = None on all: fit() builds deterministic, data-derived starts.
Multi-start is used for v (3 starting values spanning its practical range)
because the four-parameter loss surface has broader local-minima structure.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["time_h"]
PAPER_REF = "summary_formula_zwietering_1990.md"
EQUATION_LOC = (
    "Zwietering 1990 Table 2, PDF p. 2 — modified Richards: "
    "y = A*{1 + v*exp(1+v)*exp[(mu_m/A)*(1+v)^(1+1/v)*(lambda-t)]}^(-1/v); "
    "equivalence with Stannard noted on PDF p. 3 para. 2."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "A":       {"init": None},
    "mu_m":    {"init": None},
    "lambda_": {"init": None},
    "v":       {"init": None},
}


def _richards(t, A, mu_m, lambda_, v):
    """Modified Richards OD600 prediction (Zwietering 1990 Table 2).

    Numerically guarded: the exponent (1+1/v) can diverge as v→0; v is
    bounded away from 0 in fit(). The inner exp argument is clipped to
    avoid overflow.
    """
    # Structural factors from algebraic reparameterisation (not constants):
    #   v * exp(1+v) — coefficient from inflection-point identity
    #   (1+v)^(1+1/v) — rate factor from inflection-point identity
    exponent = (1.0 + 1.0 / v)
    rate_factor = (1.0 + v) ** exponent
    arg = np.clip((mu_m / A) * rate_factor * (lambda_ - t), -700.0, 700.0)
    base = 1.0 + v * np.exp(1.0 + v) * np.exp(arg)
    # base > 0 always when v > 0 and A > 0; clip for safety
    base = np.clip(base, 1e-12, None)
    return A * base ** (-1.0 / v)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the modified Richards form.

    Deterministic, data-derived start with 3-start sweep over v:
      A0      = 90th percentile of observed OD600.
      mu_m0   = maximum finite-difference slope.
      lambda0 = tangent x-intercept at estimated inflection.
      v0 sweep = [0.5, 1.0, 2.0] — covers sub-logistic, logistic-like, and
                 supra-logistic asymmetry; best-fit start selected by residual.

    The Richards form is more sensitive to initialisation than the Gompertz/
    logistic; the 3-start sweep is deterministic and cheap on 97-point data.
    """
    t = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # --- A0 ---
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
    #             A         mu_m    lambda_   v
    param_lo = [1e-4,     1e-6,    0.0,      0.05 ]
    param_hi = [10.0,     5.0,     200.0,    20.0 ]

    v_starts = [0.5, 1.0, 2.0]   # deterministic sweep, not random

    best_params = None
    best_cost = np.inf

    for v0 in v_starts:
        p0 = [A0, mu_m0, lambda0, v0]
        p0 = [min(max(v_, lo), hi)
              for v_, lo, hi in zip(p0, param_lo, param_hi)]

        def residual(p):                                   # noqa: E306
            return _richards(t, p[0], p[1], p[2], p[3]) - y

        try:
            sol = least_squares(residual, p0, bounds=(param_lo, param_hi),
                                method="trf", max_nfev=4000)
            if np.all(np.isfinite(sol.x)) and sol.cost < best_cost:
                best_cost = sol.cost
                best_params = sol.x.tolist()
        except Exception:                                  # noqa: BLE001
            pass

    if best_params is not None:
        A, mu_m, lambda_, v = best_params
        return {
            "A":       float(A),
            "mu_m":    float(mu_m),
            "lambda_": float(lambda_),
            "v":       float(v),
        }
    # fallback: return centre-of-v-range init
    return {
        "A":       float(A0),
        "mu_m":    float(mu_m0),
        "lambda_": float(lambda0),
        "v":       1.0,
    }


def predict(X: np.ndarray, A: float, mu_m: float, lambda_: float,
            v: float) -> np.ndarray:
    """Modified Richards OD600 as a function of time.

    X: (n, 1) — column [time_h].
    Returns: (n,) array of predicted OD600 values.
    """
    t = np.asarray(X[:, 0], dtype=float)
    return _richards(t, A, mu_m, lambda_, v)
