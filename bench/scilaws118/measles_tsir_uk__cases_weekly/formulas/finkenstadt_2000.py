"""Finkenstadt & Grenfell (2000) linearised TSIR — log-linear / exp-susceptibility.

Finkenstadt & Grenfell (2000), J. R. Stat. Soc. C 49(2):187-205, Eq. 15
(PDF p. 10): a first-order Taylor expansion of ln S around its mean
absorbs the susceptible-pool factor into an additive linear term, giving
the LOG-LINEAR transmission

    ln I_t = ln r*_s + alpha * ln I_{t-1} + gamma * Z_{t-1}

where r*_s is the seasonal force (s = biweek mod 26), alpha the
density-dependent / homogeneity exponent, and gamma the susceptibility
slope. This is the closed-form, OLS-fitable face of the TSIR transmission
equation; the linear-additive S form (grenfell_2001, bjornstad_2002,
becker_2017) is the alternative parameterisation.

Fourier reduction of the seasonal forcing
-----------------------------------------
The paper's 26-biweek r*_s vector is reduced to a 3-parameter Fourier
series, which captures the school-term / holiday annual cycle the data
actually demand:

    ln r*_s = b0 + b1 * cos(2*pi*s/26) + b2 * sin(2*pi*s/26)

Per-cluster fit
---------------
ln(I_t + 1) = b0 + b1*cos + b2*sin + alpha*ln(I_{t-1} + 1) + gamma*z_t
is linear in (b0, b1, b2, alpha, gamma) — closed-form OLS, no init seeds.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The log-linear structure is the scientific claim; the five
parameters are per-city fits.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty; 26 in 2*pi*s/26 is the year-period structural constant.)

LOCAL_FITTABLE — per-cluster, computed by fit() via closed-form OLS
-------------------------------------------------------------------
- b0, b1, b2 : Fourier components of the seasonal log-force.
- alpha      : density-dependent / homogeneity exponent (~0.97 in UK measles).
- gamma      : susceptibility slope (paper: alpha_2 / Sbar ~ 1.6e-6).
init = None on all five: closed-form OLS is deterministic.
"""

import numpy as np

USED_INPUTS = ["cases_prev", "biweek", "z_t"]
PAPER_REF = "summary_formula_dataset_finkenstadt_2000.md"
EQUATION_LOC = (
    "Finkenstadt & Grenfell (2000) Eq. 15, PDF p. 10 — "
    "ln I_t = ln r*_s + alpha*ln I_{t-1} + gamma*Z_{t-1}; "
    "Fourier reduction r*_s -> exp(b0 + b1*cos(2*pi*s/26) + b2*sin(2*pi*s/26))."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "b0":    {"init": None},
    "b1":    {"init": None},
    "b2":    {"init": None},
    "alpha": {"init": None},
    "gamma": {"init": None},
}


def _design(cases_prev, biweek, z_t):
    omega = 2.0 * np.pi / 26.0
    s = biweek.astype(float)
    log_ip = np.log(np.clip(cases_prev, 0.0, None) + 1.0)
    return np.column_stack([np.ones_like(s), np.cos(omega * s), np.sin(omega * s),
                            log_ip, z_t])


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS of ln(I_t + 1) on (1, cos, sin, ln(I_{t-1}+1), z_t)."""
    cases_prev = np.asarray(X_fit[:, 0], dtype=float)
    biweek     = np.asarray(X_fit[:, 1], dtype=float)
    z_t        = np.asarray(X_fit[:, 2], dtype=float)
    y_log = np.log(np.clip(np.asarray(y_fit, dtype=float), 0.0, None) + 1.0)
    A = _design(cases_prev, biweek, z_t)
    coef, *_ = np.linalg.lstsq(A, y_log, rcond=None)
    return {"b0": float(coef[0]), "b1": float(coef[1]), "b2": float(coef[2]),
            "alpha": float(coef[3]), "gamma": float(coef[4])}


def predict(X: np.ndarray, b0: float, b1: float, b2: float,
            alpha: float, gamma: float) -> np.ndarray:
    """I_t = exp(b0 + b1*cos + b2*sin + alpha*ln(I_prev+1) + gamma*z_t) - 1.

    X: (n, 3) — columns cases_prev, biweek, z_t.
    """
    cases_prev = np.asarray(X[:, 0], dtype=float)
    biweek     = np.asarray(X[:, 1], dtype=float)
    z_t        = np.asarray(X[:, 2], dtype=float)
    A = _design(cases_prev, biweek, z_t)
    return np.clip(np.exp(A @ np.array([b0, b1, b2, alpha, gamma])) - 1.0, 0.0, None)
