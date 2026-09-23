"""Oeppen-Vaupel (2002) linear rise of life expectancy.

Oeppen & Vaupel (2002), Science 296(5570):1029-1031 — "Broken Limits to
Life Expectancy". The paper's central empirical finding is that
best-practice (record) life expectancy has risen in a near-perfect
straight line for ~160 years:

    e0 = a + b * t

with t the calendar year. The paper fits this by OLS to the record
series 1840-2000 and reports b = 0.243 years/year for the female record
(r^2 = 0.992) and b = 0.222 for the male record (r^2 = 0.980) — "a
quarter of a year per year" — with NO detectable deceleration.

For this Type II task each country is a cluster: the linear-rise law is
the structural claim, while the slope b and intercept a are
population-specific and fitted per cluster. (Oeppen & Vaupel themselves
re-estimate the slope per series — female vs male records give different
b — so a, b are LOCAL_FITTABLE, not universal constants.)

    e0 = a + b * (year - year_center)

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The linear-in-calendar-year form is the scientific claim; a and b
are per-country fits.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
- year_center = 1900.0 : a fixed centering year for the year axis. Pure
  numerical-conditioning choice (centering keeps the intercept near the
  observed e0 level); it does not affect the fitted slope b.

LOCAL_FITTABLE — per-cluster, computed by fit() via closed-form OLS
-------------------------------------------------------------------
- a : life expectancy at year = year_center for the country.
- b : annual increment in life expectancy (the Oeppen-Vaupel slope).
init = None on both: linear OLS is deterministic, no multi-start.
"""

import numpy as np

USED_INPUTS = ["year"]
PAPER_REF = "summary_formula_oeppen_2002.md"
EQUATION_LOC = (
    "Oeppen & Vaupel (2002), Science 296(5570):1029-1031, Fig. 1 and text "
    "p. 1 — best-practice e0 = a + b*t, linear in calendar year "
    "(b = 0.243 female / 0.222 male record)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    "year_center": 1900.0,    # numerical-conditioning centering for the year axis
}
LOCAL_FITTABLE = {
    "a": {"init": None},      # closed-form OLS
    "b": {"init": None},
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS of e0 on (year - year_center)."""
    yc = OTHER_CONSTANTS["year_center"]
    year = np.asarray(X_fit[:, 0], dtype=float)
    A = np.column_stack([np.ones_like(year), year - yc])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y_fit, dtype=float), rcond=None)
    return {"a": float(coef[0]), "b": float(coef[1])}


def predict(X: np.ndarray, a: float, b: float) -> np.ndarray:
    """e0 = a + b * (year - year_center).

    X: (n, 1) — column 0 is year.
    """
    yc = OTHER_CONSTANTS["year_center"]
    year = np.asarray(X[:, 0], dtype=float)
    return a + b * (year - yc)
