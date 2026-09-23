"""Gompertz log-linear adult mortality with linear calendar-year drift.

Extends the pure-Gompertz form (`gompertz_age.py`) by adding a linear
secular trend in calendar year:

    log m(x, t) = alpha + beta * age + gamma * (year - year_center)

The HMD database shows within-country mortality declining over calendar
time at all adult ages (Wilmoth et al. methods protocol, §5). A
constant-alpha Gompertz fit omits this temporal drift; adding the
gamma * (year - year_center) term captures secular mortality improvement.

All three parameters are country-specific and fitted per cluster by
linear OLS (closed-form, no init seeds needed). The year-centering
constant only improves numerical conditioning of the intercept.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The form itself is the scientific claim; the three parameters are
per-country fits.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
- year_center = 2000.0 : a fixed centering year for the year axis. Pure
  numerical-conditioning choice (centering reduces the magnitude of the
  intercept; it does not affect the slope gamma).

LOCAL_FITTABLE — per-cluster, computed by fit() via closed-form OLS
-------------------------------------------------------------------
- alpha : log-mortality intercept at age 0 and year = year_center.
- beta  : Gompertz senescence slope (per year of age).
- gamma : annual log-mortality time trend (negative = improving).
"""

import numpy as np

USED_INPUTS = ["age", "year"]
PAPER_REF = "summary_formula_heligman_1980.md"
EQUATION_LOC = (
    "Gompertz (1825) age component + standard linear-in-year actuarial drift "
    "(no separate equation number in HP 1980; conventional generalisation)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    "year_center": 2000.0,    # numerical-conditioning centering for the year axis
}
LOCAL_FITTABLE = {
    "alpha": {"init": None},
    "beta":  {"init": None},
    "gamma": {"init": None},
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS of log_m_adult on (age, year - year_center)."""
    yc = OTHER_CONSTANTS["year_center"]
    age = np.asarray(X_fit[:, 0], dtype=float)
    year = np.asarray(X_fit[:, 1], dtype=float)
    A = np.column_stack([np.ones_like(age), age, year - yc])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y_fit, dtype=float), rcond=None)
    return {"alpha": float(coef[0]), "beta": float(coef[1]), "gamma": float(coef[2])}


def predict(X: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """log_m_adult = alpha + beta * age + gamma * (year - year_center).

    X: (n, 2) — columns age, year.
    """
    yc = OTHER_CONSTANTS["year_center"]
    age = np.asarray(X[:, 0], dtype=float)
    year = np.asarray(X[:, 1], dtype=float)
    return alpha + beta * age + gamma * (year - yc)
