"""Glei-Horiuchi-style linear sex ratio with secular year drift.

Extends `glei_horiuchi_2007.py` with a linear calendar-year drift term:

    log_sex_ratio = A + B * age + C * (year - year_center)

The original Glei-Horiuchi (2007) analysis pools 1975-1979 data; over
the longer HMD record the level of the sex differential drifts —
male mortality disadvantage has narrowed in most countries through
the 20th century, so the country's intercept A shifts with calendar
time. The C * (year - year_center) term captures that secular drift
without changing the age-shape claim.

This mirrors the gompertz_age / gompertz_age_year pair in the sibling
mortality_gompertz_makeham_hmd task — same dataset family, same
"add a uniform year drift to the per-cluster linear law" pattern.
The Glei-Horiuchi (2007) paper does not publish a year-drift term;
this is a conventional generalisation grounded in the HMD time-series
record.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
- year_center = 2000.0 : numerical-conditioning centering for the year
  axis (does not affect the fitted slopes B, C).

LOCAL_FITTABLE — per-cluster, computed by fit() via closed-form OLS
-------------------------------------------------------------------
- A : log-level sex differential at year=year_center.
- B : Gompertz-slope difference (b_M - b_F), per year of age.
- C : annual log-ratio drift (typically negative — male disadvantage
      shrinks over time).
init = None on all three: linear OLS is deterministic.
"""

import numpy as np

USED_INPUTS = ["age", "year"]
PAPER_REF = "summary_supporting_glei_2007.md"
EQUATION_LOC = (
    "Glei & Horiuchi (2007) linear-in-age form ln alpha = A + B*age "
    "(Eq. 2 / §'Why it appears in this source group'); year drift "
    "C*(year-year_center) is the conventional generalisation, paralleling "
    "gompertz_age_year in the sibling HMD mortality task."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    "year_center": 2000.0,
}
LOCAL_FITTABLE = {
    "A": {"init": None},
    "B": {"init": None},
    "C": {"init": None},
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS of log_sex_ratio on (1, age, year - year_center)."""
    yc = OTHER_CONSTANTS["year_center"]
    age  = np.asarray(X_fit[:, 0], dtype=float)
    year = np.asarray(X_fit[:, 1], dtype=float)
    A = np.column_stack([np.ones_like(age), age, year - yc])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y_fit, dtype=float), rcond=None)
    return {"A": float(coef[0]), "B": float(coef[1]), "C": float(coef[2])}


def predict(X: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    """log_sex_ratio = A + B * age + C * (year - year_center)."""
    yc = OTHER_CONSTANTS["year_center"]
    age  = np.asarray(X[:, 0], dtype=float)
    year = np.asarray(X[:, 1], dtype=float)
    return A + B * age + C * (year - yc)
