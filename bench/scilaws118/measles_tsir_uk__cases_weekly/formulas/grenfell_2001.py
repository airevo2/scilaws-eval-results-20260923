"""Grenfell-Bjornstad-Kappey (2001) spatial TSIR — linear S, alpha frozen.

Grenfell, Bjornstad & Kappey (2001), Nature 414:716-723, Box 2 Eq. 1
(PDF p. 5 / journal p. 720):

    I_t = beta_s * S_t * I_{t-1}^alpha

The Methods (PDF p. 7) fix alpha = 0.97 across all locations as a
universal pre-vaccination measles constant; the seasonal beta_s and the
per-city susceptible baseline S_t = z_t + s_bar are the remaining
structure. Here alpha is the only LAW_CONSTANT (frozen at 0.97); the
seasonal 26-vector is Fourier-reduced to (b0, b1, b2) and s_bar is fit
per city.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
- alpha = 0.97 : universal density-dependent exponent for pre-vaccination
  UK measles (Grenfell et al. 2001 Methods, PDF p. 7).

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster, computed by fit() via nonlinear LS in log space
-----------------------------------------------------------------------------
- b0, b1, b2 : Fourier components of the seasonal log-force ln beta_s.
- s_bar      : per-city susceptible baseline (S_t = z_t + s_bar).
init = None on all four: fit() builds its own data-derived start.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["cases_prev", "biweek", "z_t"]
PAPER_REF = "summary_formula_dataset_grenfell_2001.md"
EQUATION_LOC = (
    "Grenfell, Bjornstad & Kappey (2001) Box 2 Eq. 1, PDF p. 5 / p. 720 — "
    "I_t = beta_s * S_t * I_{t-1}^alpha (alpha = 0.97 from Methods p. 7); "
    "Fourier reduction beta_s -> exp(b0 + b1*cos + b2*sin), S_t = z_t + s_bar."
)

LAW_CONSTANTS = {"alpha": 0.97}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "b0":    {"init": None},
    "b1":    {"init": None},
    "b2":    {"init": None},
    "s_bar": {"init": None},
}


def _log_pred(cases_prev, biweek, z_t, b0, b1, b2, s_bar, alpha):
    omega = 2.0 * np.pi / 26.0
    s = biweek.astype(float)
    log_beta = b0 + b1 * np.cos(omega * s) + b2 * np.sin(omega * s)
    log_i = alpha * np.log(np.clip(cases_prev, 0.0, None) + 1.0)
    s_pool = np.clip(z_t + s_bar, 1e-6, None)
    return log_beta + log_i + np.log(s_pool)


def fit(X_fit: np.ndarray, y_fit: np.ndarray, alpha: float = 0.97) -> dict:
    cases_prev = np.asarray(X_fit[:, 0], dtype=float)
    biweek     = np.asarray(X_fit[:, 1], dtype=float)
    z_t        = np.asarray(X_fit[:, 2], dtype=float)
    y_log = np.log(np.clip(np.asarray(y_fit, dtype=float), 0.0, None) + 1.0)

    s_bar0 = float(max(-z_t.min() + 1.0, 1.0))
    p0 = [-15.0, 0.0, 0.0, s_bar0]

    def residual(p):
        b0, b1, b2, s_bar = p
        return _log_pred(cases_prev, biweek, z_t, b0, b1, b2, s_bar, alpha) - y_log

    try:
        sol = least_squares(residual, p0,
                            bounds=([-50.0, -10.0, -10.0, max(s_bar0 - 1.0, 1e-6)],
                                    [10.0, 10.0, 10.0, 1e12]),
                            method="trf", max_nfev=4000)
        return {"b0": float(sol.x[0]), "b1": float(sol.x[1]),
                "b2": float(sol.x[2]), "s_bar": float(sol.x[3])}
    except Exception:                              # noqa: BLE001
        return {"b0": p0[0], "b1": p0[1], "b2": p0[2], "s_bar": p0[3]}


def predict(X: np.ndarray, alpha: float, b0: float, b1: float, b2: float,
            s_bar: float) -> np.ndarray:
    cases_prev = np.asarray(X[:, 0], dtype=float)
    biweek     = np.asarray(X[:, 1], dtype=float)
    z_t        = np.asarray(X[:, 2], dtype=float)
    return np.clip(np.exp(_log_pred(cases_prev, biweek, z_t, b0, b1, b2,
                                    s_bar, alpha)) - 1.0, 0.0, None)
