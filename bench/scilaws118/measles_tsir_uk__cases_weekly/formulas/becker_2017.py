"""Becker & Grenfell (2017) tsiR-package TSIR — linear S, S_bar tied to N.

Becker & Grenfell (2017), PLoS ONE 12(9):e0185528, Eq. 2-3 (PDF p. 2):

    E[I_{t+1}] = beta_{t+1} * S_t * I_t^alpha,   S_t = z_t + Sbar

In log-linear form (Eq. 3):

    log I_{t+1} = log beta_{t+1} + log(z_t + Sbar) + alpha * log I_t

The tsiR package (PDF p. 3, §1.2) recommends two universal-across-cities
defaults derived from the UK measles literature: alpha = 0.97 and
Sbar / N = 0.035, so Sbar = sbar_frac * N is tied to city population
rather than free. Both alpha and sbar_frac are re-fit per city here; the
seasonal 26-vector is Fourier-reduced.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster, computed by fit() via nonlinear LS in log space
-----------------------------------------------------------------------------
- b0, b1, b2 : Fourier components of the seasonal log-force.
- alpha      : density-dependent exponent.
- sbar_frac  : per-city susceptible fraction Sbar / N (paper: ~0.035).
init = None: fit() builds its own data-derived start.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["cases_prev", "biweek", "z_t", "log_pop"]
PAPER_REF = "summary_formula_dataset_becker_2017.md"
EQUATION_LOC = (
    "Becker & Grenfell (2017) Eq. 2-3, PDF p. 2 — "
    "I_t = beta_s * (z_t + sbar_frac * N) * I_{t-1}^alpha, N = 10^log_pop; "
    "Fourier reduction beta_s -> exp(b0 + b1*cos + b2*sin)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "b0":        {"init": None},
    "b1":        {"init": None},
    "b2":        {"init": None},
    "alpha":     {"init": None},
    "sbar_frac": {"init": None},
}


def _log_pred(cases_prev, biweek, z_t, log_pop, b0, b1, b2, alpha, sbar_frac):
    omega = 2.0 * np.pi / 26.0
    s = biweek.astype(float)
    log_beta = b0 + b1 * np.cos(omega * s) + b2 * np.sin(omega * s)
    log_i = alpha * np.log(np.clip(cases_prev, 0.0, None) + 1.0)
    N = np.power(10.0, log_pop)
    s_pool = np.clip(z_t + sbar_frac * N, 1e-6, None)
    return log_beta + log_i + np.log(s_pool)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    cases_prev = np.asarray(X_fit[:, 0], dtype=float)
    biweek     = np.asarray(X_fit[:, 1], dtype=float)
    z_t        = np.asarray(X_fit[:, 2], dtype=float)
    log_pop    = np.asarray(X_fit[:, 3], dtype=float)
    y_log = np.log(np.clip(np.asarray(y_fit, dtype=float), 0.0, None) + 1.0)

    p0 = [-15.0, 0.0, 0.0, 1.0, 0.035]

    def residual(p):
        b0, b1, b2, alpha, sbar_frac = p
        return _log_pred(cases_prev, biweek, z_t, log_pop, b0, b1, b2,
                         alpha, sbar_frac) - y_log

    try:
        sol = least_squares(residual, p0,
                            bounds=([-50.0, -10.0, -10.0, 0.0, 1e-6],
                                    [10.0, 10.0, 10.0, 3.0, 1.0]),
                            method="trf", max_nfev=4000)
        return {"b0": float(sol.x[0]), "b1": float(sol.x[1]),
                "b2": float(sol.x[2]), "alpha": float(sol.x[3]),
                "sbar_frac": float(sol.x[4])}
    except Exception:                              # noqa: BLE001
        return {"b0": p0[0], "b1": p0[1], "b2": p0[2],
                "alpha": p0[3], "sbar_frac": p0[4]}


def predict(X: np.ndarray, b0: float, b1: float, b2: float,
            alpha: float, sbar_frac: float) -> np.ndarray:
    cases_prev = np.asarray(X[:, 0], dtype=float)
    biweek     = np.asarray(X[:, 1], dtype=float)
    z_t        = np.asarray(X[:, 2], dtype=float)
    log_pop    = np.asarray(X[:, 3], dtype=float)
    return np.clip(np.exp(_log_pred(cases_prev, biweek, z_t, log_pop,
                                    b0, b1, b2, alpha, sbar_frac)) - 1.0,
                   0.0, None)
