"""Atkinson-Piketty-Saez (2011) Pareto upper-tail self-similarity.

Atkinson, Piketty & Saez (2011), JEL 49(1):3-71, Eq. 1 (PDF p. 11), gives
the Pareto upper-tail distribution

    1 - F(y) = (k / y)**alpha,   y > k,  alpha > 1

For a Pareto-distributed income with tail exponent alpha, the share of
total income held by the top fraction p of the population satisfies

    S(p) = p**(1 - 1/alpha)        =>    log S(p) = (1 - 1/alpha) * log p

so in log-log space `top_share_log` is a linear function of `log_p_above`
with slope `1 - 1/alpha`. The intercept absorbs the truncation /
normalisation scale of the underlying distribution (Atkinson 2011 §3.1.1
PDF pp. 13-14).

This task releases 5 percentile cutoffs per (country, year), making the
linear fit over-determined; the OLS residuals are real deviations from the
strict Pareto form (lognormal middle, double-Pareto kinks, year-to-year
drift in alpha) and are the SR signal.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The paper's claim is the *form* (Pareto self-similarity), not a
specific universal numeric value; that form is what `predict()` implements.

OTHER_CONSTANTS — universal physics / unit factors
--------------------------------------------------
(empty: the formula is dimensionless.)

LOCAL_FITTABLE — per-cluster, computed by fit() via closed-form OLS
-------------------------------------------------------------------
- a : intercept (country-specific scale).
- b : slope = 1 - 1/alpha (country-specific Pareto exponent).

Both are fitted by ordinary least squares on (log_p_above, top_share_log)
pairs from the cluster's test_fit window — closed-form, no multi-start,
no INIT seeds needed (init = None).
"""

import numpy as np

USED_INPUTS = ["log_p_above"]
PAPER_REF = "summary_formula_dataset_atkinson_2011.md"
EQUATION_LOC = (
    "Atkinson-Piketty-Saez 2011 §5.2, PDF p. 58, journal p. 60 "
    "(log-linear regression); Pareto CDF Eq. 1-2, PDF p. 11, journal p. 13"
)

LAW_CONSTANTS = {}                        # the form IS the claim — no frozen scalar
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "a": {"init": None},                  # closed-form OLS
    "b": {"init": None},                  # closed-form OLS
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS of top_share_log on log_p_above.

    Returns (a, b) = (intercept, slope). Pareto exponent alpha = 1/(1-b).
    """
    log_p = np.asarray(X_fit[:, 0], dtype=float)
    log_S = np.asarray(y_fit, dtype=float)
    A = np.column_stack([np.ones_like(log_p), log_p])
    coef, *_ = np.linalg.lstsq(A, log_S, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return {"a": a, "b": b}


def predict(X: np.ndarray, a: float, b: float) -> np.ndarray:
    """Linear Pareto self-similarity:

        top_share_log = a + b * log_p_above
    """
    log_p = np.asarray(X[:, 0], dtype=float)
    return a + b * log_p
