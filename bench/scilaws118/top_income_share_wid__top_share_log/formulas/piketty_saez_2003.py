"""Piketty-Saez (2003) year-augmented Pareto-Lorenz form.

Piketty & Saez (2003), QJE 118(1):1-39, document a pronounced secular
trend in U.S. top income shares: sharp decline 1914-1945 and recovery from
the early 1970s onward (§§III-IV, PDF pp. 11-15). The static Pareto-Lorenz
slope does not capture this calendar-time drift in alpha.

The year-augmented form adds a linear-in-time term to the base Pareto
self-similarity log-log line:

    top_share_log = a + b * log_p_above + c * (year - YEAR_CENTER)

- a is the country-level intercept absorbing tail-truncation scale.
- b is the Pareto self-similarity slope, b = 1 - 1/alpha_country.
- c captures the country-specific secular drift in inequality.

Centering `year` at YEAR_CENTER = 2000 keeps the design matrix well
conditioned over the 1980-2023 release window without altering the fit
(centering is a numerical convenience for OLS, not a scientific claim).
All three parameters are per-country.

Piketty-Saez 2003's U.S. series is non-monotone over the full 20th
century, but on the 1980-2023 WID benchmark window the linear-in-time
approximation is a reasonable phenomenological capture of the post-1980
inequality rise within each country.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The paper's claim is the *form* (Pareto self-similarity with a
linear secular drift), not a specific universal numeric value.

OTHER_CONSTANTS — universal physics / unit factors
--------------------------------------------------
- year_center = 2000.0 — the year-centering constant. Not a scientific
  claim; a numerical centering chosen for OLS conditioning.

LOCAL_FITTABLE — per-cluster, computed by fit() via closed-form OLS
-------------------------------------------------------------------
- a : intercept.
- b : Pareto self-similarity slope (1 - 1/alpha_country).
- c : per-decade linear inequality drift coefficient.

All three are fitted by ordinary least squares on the cluster's test_fit
rows — closed-form, no multi-start, no INIT seeds (init = None).
"""

import numpy as np

USED_INPUTS = ["log_p_above", "year"]
PAPER_REF = "summary_dataset_piketty_2003.md"
EQUATION_LOC = (
    "Piketty-Saez 2003 §§III-IV (PDF pp. 11-15) time-trend motivation; "
    "Pareto interpolation PDF p. 6 — the year-augmented Pareto-Lorenz "
    "log-log form."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    "year_center": 2000.0,                  # numerical OLS centering
}
LOCAL_FITTABLE = {
    "a": {"init": None},                    # closed-form OLS
    "b": {"init": None},                    # closed-form OLS
    "c": {"init": None},                    # closed-form OLS
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS of top_share_log on (log_p_above, year - year_center)."""
    log_p = np.asarray(X_fit[:, 0], dtype=float)
    year  = np.asarray(X_fit[:, 1], dtype=float)
    log_S = np.asarray(y_fit, dtype=float)
    A = np.column_stack([np.ones_like(log_p), log_p, year - OTHER_CONSTANTS["year_center"]])
    coef, *_ = np.linalg.lstsq(A, log_S, rcond=None)
    a, b, c = float(coef[0]), float(coef[1]), float(coef[2])
    return {"a": a, "b": b, "c": c}


def predict(X: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Year-augmented Pareto-Lorenz prediction:

        top_share_log = a + b * log_p_above + c * (year - year_center)
    """
    log_p = np.asarray(X[:, 0], dtype=float)
    year  = np.asarray(X[:, 1], dtype=float)
    return a + b * log_p + c * (year - OTHER_CONSTANTS["year_center"])
