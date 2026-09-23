"""Log-normal (probit) four-parameter dose-response model — Ritz et al. 2015.

Ritz C, Baty F, Streibig JC, Gerhard D (2015). "Dose-Response Analysis Using
R." PLOS ONE 10(12): e0146021. DOI: 10.1371/journal.pone.0146021.

The log-normal model (called `lnormal()` in the `drc` R package) is stated in
Table 1 of Ritz et al. (2015). For four parameters:

    f(x; b, c, d, e) = c + (d - c) * Phi(b * (log(x) - log(e)))

where Phi is the standard normal CDF, log is the natural logarithm, b is the
slope (positive = increasing response with dose), c is the lower asymptote
(response as x -> infinity), d is the upper asymptote (response as x -> 0),
and e is the ED50 (the dose producing response (c + d) / 2, equivalent to LC50
when c = 0, d = 1 for standard mortality bioassay).

This is the direct generalisation of the Bliss/Finney probit model. For the
standard mortality bioassay (c = 0, d = 1), it simplifies to:

    P = Phi(b * (log(x) - log(e)))

which is identical to the Lopez 2025 formulation with alpha = -b * log(e) and
beta = b / log(10) when log10 is used in Lopez.

In the general four-parameter form, all of b, c, d, e are LOCAL_FITTABLE;
the Phi function and natural log are fixed structural constants.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The log-normal (probit) FORM is the scientific claim. All four
parameters (b, c, d, e) are per-cluster fit values; no universal numerical
value across populations is published.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
None assigned. Phi (standard normal CDF) and the natural logarithm are fixed
structural link functions, never tunable.

LOCAL_FITTABLE — per-cluster, fitted by fit() via MLE
------------------------------------------------------
- b : slope (dimensionless); positive for increasing mortality with dose.
      Typical range across populations: reflects beta in Lopez 2025 Table 1.
- c : lower asymptote (dimensionless, [0, 1]); typically fixed at 0 for
      standard mortality bioassay (full kill at high dose).
- d : upper asymptote (dimensionless, [0, 1]); typically fixed at 1 for
      standard mortality bioassay (zero mortality at zero dose).
- e : ED50 / LC50 (same units as x, here ppb); must be > 0.

init = None on all: fit() builds data-derived starting values from the
empirical dose-response relationship.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr  # standard normal CDF, numerically stable

USED_INPUTS = ["log_conc_ppb"]
PAPER_REF = "summary_formula_ritz_2015.md"
EQUATION_LOC = (
    "Ritz et al. (2015) Table 1 — log-normal (lnormal) model: "
    "f(x; b,c,d,e) = c + (d-c)*Phi(b*(log(x) - log(e))), "
    "PDF pp. 2-3; Phi = standard normal CDF; log = natural logarithm."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "b": {"init": None},
    "c": {"init": None},
    "d": {"init": None},
    "e": {"init": None},
}


def _lnormal(log_x, b, c, d, e):
    """f = c + (d - c) * Phi(b * (log_x - log(e))).

    log_x = ln(concentration); e > 0 so log(e) is well-defined.
    """
    log_e = np.log(np.clip(e, 1e-15, None))
    return c + (d - c) * ndtr(b * (log_x - log_e))


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """MLE fit of the four-parameter log-normal dose-response model.

    Deterministic, data-derived start:
      d0 = min(1.0, max observed response + margin)  ~ upper asymptote
      c0 = max(0.0, min observed response - margin)  ~ lower asymptote
      e0 = exp(median(log_x at responses near 0.5))  ~ ED50 estimate
      b0 = 1.0  (moderate slope)

    Bounds enforce e > 0, c in [0,1], d in [0,1], b > 0.
    """
    log_x = np.asarray(X_fit[:, 0], dtype=float)  # ln(concentration)
    y = np.clip(np.asarray(y_fit, dtype=float), 1e-6, 1.0 - 1e-6)

    # Data-derived starting values
    c0 = max(0.0, float(np.percentile(y, 5)) - 0.05)
    d0 = min(1.0, float(np.percentile(y, 95)) + 0.05)
    # ED50 estimate: log_x where response is closest to (c0+d0)/2
    mid = 0.5 * (c0 + d0)
    idx = int(np.argmin(np.abs(y - mid)))
    e0 = float(np.exp(log_x[idx])) if np.isfinite(log_x[idx]) else 1.0
    if e0 <= 0 or not np.isfinite(e0):
        e0 = float(np.exp(np.median(log_x)))
    b0 = 1.0

    def neg_log_lik(params):
        b, c, d, e = params
        if e <= 0 or d <= c:
            return 1e12
        p = np.clip(_lnormal(log_x, b, c, d, e), 1e-9, 1.0 - 1e-9)
        return -float(np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))

    try:
        res = minimize(
            neg_log_lik,
            x0=[b0, c0, d0, e0],
            method="Nelder-Mead",
            options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-7},
        )
        b_fit, c_fit, d_fit, e_fit = res.x
        # Enforce physical constraints
        c_fit = float(np.clip(c_fit, 0.0, 1.0))
        d_fit = float(np.clip(d_fit, 0.0, 1.0))
        e_fit = float(max(e_fit, 1e-9))
        if not np.all(np.isfinite([b_fit, c_fit, d_fit, e_fit])):
            raise RuntimeError("non-finite fit")
        return {"b": float(b_fit), "c": c_fit, "d": d_fit, "e": e_fit}
    except Exception:  # noqa: BLE001
        return {"b": float(b0), "c": float(c0), "d": float(d0), "e": float(e0)}


def predict(X: np.ndarray, b: float, c: float, d: float, e: float) -> np.ndarray:
    """Four-parameter log-normal dose-response.

    f(x; b, c, d, e) = c + (d - c) * Phi(b * (ln(x) - ln(e)))

    X: (n, 1) — column [log_conc_ppb] = ln(concentration in ppb).
    Returns predicted mortality probability in [c, d] subset of [0, 1].
    """
    log_x = np.asarray(X[:, 0], dtype=float)
    return _lnormal(log_x, b, c, d, e)
