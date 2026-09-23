"""Leopold & Maddock (1953) at-a-station hydraulic geometry — chan_width.

Leopold, L. B., and Maddock, T. Jr. (1953). *The hydraulic geometry of
stream channels and some physiographic implications.*  U.S. Geological
Survey Professional Paper 252. DOI: 10.3133/pp252.

At-a-station width power law (Equation 1, PDF p. 8):

    w = a * Q^b

where w is water-surface width (ft), Q is stream discharge (cfs), a is
the width coefficient (value of w at unit discharge, ft · cfs^(-b)), and
b is the dimensionless width exponent (log-log slope).

Identity constraint from Q = w*d*v (derived PDF p. 8):

    b + f + m = 1

where f is the depth exponent and m is the velocity exponent. Because
f > 0 and m > 0, it follows that 0 < b < 1. Leopold & Maddock report an
average b = 0.26 across 20 Great Plains / Southwest at-a-station sites
(PDF p. 9); no single universal value is claimed — b varies by station.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The structural form w = a * Q^b is the scientific claim. Both a
and b are per-station empirical parameters fitted from current-meter
records. No universal numerical constant appears in the formula.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The exponent applies directly in the power-law form; no
geometric or algebraic prefactors are introduced.

LOCAL_FITTABLE — per-cluster (site_no), fitted by fit() via OLS on
log-transformed data
--------------------------------------------------------------------
- a : width coefficient (ft * cfs^(-b), > 0). Intercept of the
      log-log regression line; represents channel width at unit discharge.
- b : width exponent (dimensionless). Slope of log(w) vs log(Q);
      physically constrained to (0, 1) by the continuity identity.
      Leopold & Maddock (1953, PDF p. 9) report average b = 0.26 at-a-
      station; used as init if data-derived estimate is unavailable.

init = None on both: fit() builds a data-derived start from OLS on
log-transformed data. The log-linear model is convex so a single closed-
form OLS solution is globally optimal; no iterative multi-start needed.
"""

import numpy as np

USED_INPUTS = ["chan_discharge"]
PAPER_REF = "summary_formula_leopold_1953.md"
EQUATION_LOC = (
    "Leopold & Maddock (1953) Eq. 1, PDF p. 8: w = a * Q^b; "
    "identity constraint b+f+m=1 also PDF p. 8."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "a": {"init": None},
    "b": {"init": None},
}


def _width(Q, a, b):
    return a * np.power(np.asarray(Q, dtype=float), b)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Fit w = a * Q^b via OLS on log-transformed data.

    log(w) = log(a) + b * log(Q) is a linear model; OLS gives the exact
    globally optimal solution in one closed-form step with no iteration.

    Positive-only rows (Q > 0 and w > 0) are used for the log transform.
    If too few valid rows are available, fall back to Leopold & Maddock's
    reported average exponent b = 0.26 (PDF p. 9) with a scale derived
    from the median observed w.

    Parameters
    ----------
    X_fit : (n, 1) array — column [chan_discharge] in cfs.
    y_fit : (n,) array — chan_width in ft.
    """
    Q = np.asarray(X_fit[:, 0], dtype=float)
    w = np.asarray(y_fit, dtype=float)

    mask = (Q > 0.0) & (w > 0.0)
    if mask.sum() >= 2:
        logQ = np.log(Q[mask])
        logw = np.log(w[mask])
        # OLS: [log(a), b] from design matrix [1, logQ]
        A = np.column_stack([np.ones(mask.sum()), logQ])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, logw, rcond=None)
            log_a, b_hat = coeffs
            a_hat = float(np.exp(log_a))
            b_hat = float(b_hat)
            if np.isfinite(a_hat) and a_hat > 0 and np.isfinite(b_hat):
                return {"a": a_hat, "b": b_hat}
        except Exception:  # noqa: BLE001
            pass

    # Fallback: Leopold & Maddock average b = 0.26 (PDF p. 9), a from median w.
    b_fallback = 0.26
    Q_ref = float(np.nanmedian(Q[Q > 0])) if (Q > 0).any() else 1.0
    w_ref = float(np.nanmedian(w[w > 0])) if (w > 0).any() else 1.0
    a_fallback = w_ref / (Q_ref ** b_fallback) if Q_ref > 0 else 1.0
    return {"a": float(a_fallback), "b": float(b_fallback)}


def predict(X: np.ndarray, a: float, b: float) -> np.ndarray:
    """At-a-station width: w = a * Q^b.

    Parameters
    ----------
    X : (n, 1) array — column [chan_discharge] in cfs.
    a : width coefficient (ft * cfs^(-b), > 0).
    b : width exponent (dimensionless; physically in (0, 1)).

    Returns
    -------
    (n,) array of predicted chan_width in ft.
    """
    Q = np.asarray(X[:, 0], dtype=float)
    return _width(Q, a, b)
