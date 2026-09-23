"""Gompertz adult mortality with an age-specific (not uniform) calendar drift.

`gompertz_age_year.py` adds a uniform secular drift gamma*(year-year_center):
mortality at every age is assumed to improve at the same proportional rate.
Lee & Carter (1992) reject that assumption — their log-bilinear model

    log m(x, t) = a_x + b_x * k_t

makes the secular improvement age-specific through b_x. The full model
carries per-age a_x, b_x vectors (66 ages here), far beyond this task's
3-parameter cap. This baseline is its minimal rank-1 reduction: take
a_x = alpha + beta*age (linear age profile), b_x proportional to age, and
k_t linear in calendar time. The product b_x*k_t collapses to one
age x year interaction term:

    log m(x, t) = alpha + beta*age + gamma * age * (year - year_center)

Equivalently log m = alpha + (beta + gamma*(year-year_center)) * age — a
Gompertz line whose slope drifts with calendar time. The improvement rate
d(log m)/d(year) = gamma*age is larger at older ages, as observed.

This is inspired by, not equal to, Lee-Carter (1992); the full model is a
separate benchmark task. All three parameters are country-specific and
fitted per cluster by linear OLS (the reduced model is linear in
alpha, beta, gamma — closed form, no init seeds).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The age-specific-drift form is the scientific claim; the three
parameters are per-country fits.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
- year_center = 2000.0 : fixed centering year for the year axis. Pure
  numerical-conditioning choice; does not affect the fitted slopes.

LOCAL_FITTABLE — per-cluster, computed by fit() via closed-form OLS
-------------------------------------------------------------------
- alpha : log-mortality intercept at age 0.
- beta  : Gompertz senescence slope at year = year_center.
- gamma : age-specific secular drift — change in the Gompertz slope per
          calendar year (negative = mortality improving, faster at old age).
"""

import numpy as np

USED_INPUTS = ["age", "year"]
PAPER_REF = "summary_formula_lee_carter_1992.md"
EQUATION_LOC = (
    "Lee & Carter (1992) log-bilinear model log m(x,t) = a_x + b_x*k_t, "
    "JASA 87(419) p. 660 §2; rank-1 reduction with a_x = alpha + beta*age, "
    "b_x proportional to age, k_t linear in (year - year_center)."
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
    """Closed-form OLS of log_m_adult on (age, age*(year - year_center))."""
    yc = OTHER_CONSTANTS["year_center"]
    age = np.asarray(X_fit[:, 0], dtype=float)
    year = np.asarray(X_fit[:, 1], dtype=float)
    A = np.column_stack([np.ones_like(age), age, age * (year - yc)])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y_fit, dtype=float), rcond=None)
    return {"alpha": float(coef[0]), "beta": float(coef[1]), "gamma": float(coef[2])}


def predict(X: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """log_m_adult = alpha + beta*age + gamma*age*(year - year_center).

    X: (n, 2) — columns age, year.
    """
    yc = OTHER_CONSTANTS["year_center"]
    age = np.asarray(X[:, 0], dtype=float)
    year = np.asarray(X[:, 1], dtype=float)
    return alpha + beta * age + gamma * age * (year - yc)
