"""Atkinson-Piketty-Saez (2011) Pareto upper-tail self-similarity.

Atkinson, Piketty & Saez (2011), JEL 49(1):3-71, Eq. 1 (PDF p. 11), gives
the Pareto upper-tail distribution

    1 - F(y) = (k / y)**alpha,   y > k,  alpha > 1

For a Pareto-distributed income with exponent alpha, the share of total
income held by the top fraction p satisfies

    S(p) = p**(1 - 1/alpha)        =>    log S(p) = (1 - 1/alpha) * log p

In log-log space, log_top_share is a **linear function of log_p_above**
with slope `1 - 1/alpha`. The intercept `C` absorbs the truncation /
normalisation scale of the underlying distribution (Atkinson 2011 §3.1.1
PDF pp. 13-14 discusses this for the two-bracket case k > 0; with K > 2
brackets the line is over-determined and an intercept must be fit).

Task framing (v2)
-----------------
Each country contributes 14 years x 5 percentiles = 70 test rows. fit() is
called on the cluster's fit-split (7 years x 5 percentiles = 35 rows) and
performs a single OLS regression of log_top_share on log_p_above to recover
the country's (alpha, C). predict() then plugs alpha and C into the Pareto
self-similarity form.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. Atkinson 2011's "claim" is the *form* (Pareto self-similarity itself)
not a specific numeric value; that form is what predict() implements.

OTHER_CONSTANTS — universal physics / unit factors
--------------------------------------------------
(empty: the formula is dimensionless.)

LOCAL_FITTABLE — per-cluster, computed by fit() via closed-form OLS
------------------------------------------------------------------
- alpha : Pareto upper-tail exponent for the country (steady-state,
          time-averaged within the fit window). Derived from the OLS
          slope as `alpha = 1 / (1 - slope)`.
- C     : intercept absorbing the country's tail-truncation scale.

Both LOCAL params are fit by linear OLS on (log_p_above, log_top_share)
pairs — no multi-start, no INIT seeds needed (init = None).

Why this is no longer R²=1
---------------------------
Real WID income distributions are only **approximately** Pareto in their
upper tail. The 5-point line is over-determined: residuals between OLS
prediction and observed log_top_share reflect deviations from strict
Pareto (lognormal middle, double-Pareto kinks, year-by-year alpha drift).
R² is high but < 1 — those residuals are the SR signal.
"""

import numpy as np

USED_INPUTS = ["log_p_above"]
PAPER_REF = "summary_formula_dataset_atkinson_2011.md"
EQUATION_LOC = (
    "Atkinson-Piketty-Saez 2011 Eq. 1, PDF p. 11 (Pareto law); "
    "the closed-form log-share derivation S(p) = p^(1-1/alpha) "
    "yielding the linear log-log Pareto self-similarity"
)

LAW_CONSTANTS = {}                          # no paper-published scalar; the form IS the claim
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "alpha": {"init": None},                # closed-form OLS
    "C":     {"init": None},                # closed-form OLS
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS of log_top_share vs log_p_above.

    Returns the Pareto exponent alpha (from slope = 1 - 1/alpha) plus the
    intercept C absorbing the country's tail-truncation scale.
    """
    log_p = np.asarray(X_fit[:, 0], dtype=float)
    log_S = np.asarray(y_fit, dtype=float)
    A = np.column_stack([log_p, np.ones_like(log_p)])
    coef, *_ = np.linalg.lstsq(A, log_S, rcond=None)
    slope, C = float(coef[0]), float(coef[1])
    if slope >= 0.999999:
        # Degenerate: would give alpha -> ∞ (non-Pareto tail). Cap.
        alpha = 1e6
    else:
        alpha = 1.0 / (1.0 - slope)
    return {"alpha": float(alpha), "C": C}


def predict(X: np.ndarray, alpha: float, C: float) -> np.ndarray:
    """Pareto self-similarity in log-log space.

    log_top_share = (1 - 1/alpha) * log_p_above + C
    """
    log_p = np.asarray(X[:, 0], dtype=float)
    slope = 1.0 - 1.0 / alpha
    return slope * log_p + C
