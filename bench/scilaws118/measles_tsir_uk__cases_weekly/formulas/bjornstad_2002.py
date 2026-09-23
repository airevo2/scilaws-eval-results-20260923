"""Bjornstad-Finkenstadt-Grenfell (2002) TSIR — linear S, alpha per city.

Bjornstad, Finkenstadt & Grenfell (2002), Ecological Monographs
72(2):169-184, Eq. 1 (PDF p. 5):

    lambda_{t+1} = beta_s * (I_t + theta_t)^alpha * (Sbar + z_t)^gamma

The paper sets gamma = 1 throughout (PDF p. 5 Eq. 5 derivation: "we
consider gamma to be a fixed quantity, gamma = 1") and reports a
cross-city weighted-mean alpha ≈ 1.006 with per-city refits (PDF p. 6).
For a deterministic predict (theta_t = 0):

    I_t = beta_s * I_{t-1}^alpha * (z_t + s_bar)

Unlike grenfell_2001 (which fixes alpha = 0.97 universally), here alpha
is re-fit per city. The seasonal 26-vector is Fourier-reduced.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster, computed by fit() via nonlinear LS in log space
-----------------------------------------------------------------------------
- b0, b1, b2 : Fourier components of the seasonal log-force.
- alpha      : density-dependent exponent (~1.006 paper).
- s_bar      : per-city susceptible baseline.
init = None: fit() builds its own data-derived start.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["cases_prev", "biweek", "z_t"]
PAPER_REF = "summary_formula_dataset_bjornstad_2002.md"
EQUATION_LOC = (
    "Bjornstad, Finkenstadt & Grenfell (2002) Eq. 1, PDF p. 5 — "
    "I_t = beta_s * I_{t-1}^alpha * (z_t + s_bar) (gamma fixed at 1); "
    "Fourier reduction beta_s -> exp(b0 + b1*cos + b2*sin)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "b0":    {"init": None},
    "b1":    {"init": None},
    "b2":    {"init": None},
    "alpha": {"init": None},
    "s_bar": {"init": None},
}


def _log_pred(cases_prev, biweek, z_t, b0, b1, b2, alpha, s_bar):
    omega = 2.0 * np.pi / 26.0
    s = biweek.astype(float)
    log_beta = b0 + b1 * np.cos(omega * s) + b2 * np.sin(omega * s)
    log_i = alpha * np.log(np.clip(cases_prev, 0.0, None) + 1.0)
    s_pool = np.clip(z_t + s_bar, 1e-6, None)
    return log_beta + log_i + np.log(s_pool)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    cases_prev = np.asarray(X_fit[:, 0], dtype=float)
    biweek     = np.asarray(X_fit[:, 1], dtype=float)
    z_t        = np.asarray(X_fit[:, 2], dtype=float)
    y_log = np.log(np.clip(np.asarray(y_fit, dtype=float), 0.0, None) + 1.0)

    s_bar0 = float(max(-z_t.min() + 1.0, 1.0))
    p0 = [-15.0, 0.0, 0.0, 1.0, s_bar0]

    def residual(p):
        b0, b1, b2, alpha, s_bar = p
        return _log_pred(cases_prev, biweek, z_t, b0, b1, b2, alpha, s_bar) - y_log

    try:
        sol = least_squares(residual, p0,
                            bounds=([-50.0, -10.0, -10.0, 0.0, max(s_bar0 - 1.0, 1e-6)],
                                    [10.0, 10.0, 10.0, 3.0, 1e12]),
                            method="trf", max_nfev=4000)
        return {"b0": float(sol.x[0]), "b1": float(sol.x[1]),
                "b2": float(sol.x[2]), "alpha": float(sol.x[3]),
                "s_bar": float(sol.x[4])}
    except Exception:                              # noqa: BLE001
        return {"b0": p0[0], "b1": p0[1], "b2": p0[2], "alpha": p0[3], "s_bar": p0[4]}


def predict(X: np.ndarray, b0: float, b1: float, b2: float,
            alpha: float, s_bar: float) -> np.ndarray:
    cases_prev = np.asarray(X[:, 0], dtype=float)
    biweek     = np.asarray(X[:, 1], dtype=float)
    z_t        = np.asarray(X[:, 2], dtype=float)
    return np.clip(np.exp(_log_pred(cases_prev, biweek, z_t, b0, b1, b2,
                                    alpha, s_bar)) - 1.0, 0.0, None)
