"""Four-parameter log-logistic model (LL.4) — Ritz et al. 2015.

Ritz C, Baty F, Streibig JC, Gerhard D (2015). "Dose-Response Analysis Using
R." PLOS ONE 10(12): e0146021. DOI: 10.1371/journal.pone.0146021.

The four-parameter log-logistic model is Eq. (2) of Ritz et al. (2015), PDF
p. 2:

    f(x; b, c, d, e) = c + (d - c) / (1 + exp(b * (log(x) - log(e))))

where log is the natural logarithm, b is the slope parameter (negative for
decreasing response — i.e. mortality increasing with dose, since mortality
goes from d at low dose to c at high dose when b < 0; or positive for
increasing response), c is the lower asymptote (response as x -> infinity),
d is the upper asymptote (response as x -> 0), and e is the ED50 (the dose
producing the midpoint response (c + d) / 2, equivalent to LC50 for standard
mortality bioassay with c = 0, d = 1).

For the standard mortality bioassay parameterisation (c = 0, d = 1, b < 0):

    P = 1 / (1 + exp(b * (log(x) - log(e))))

which is equivalent to the logistic sigmoid with log-dose argument. This is
the most widely used alternative to the probit model in ecotoxicology (Ritz
2015, PDF p. 1-3). The LL.4 is the `drc` default dose-response model.

Special cases:
  LL.3 (c = 0 fixed): f = d / (1 + exp(b*(log(x) - log(e))))  — PDF p. 4
  LL.2 (c = 0, d = 1 fixed): f = 1 / (1 + exp(b*(log(x) - log(e))))  — PDF p. 4

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The LL.4 FORM is the scientific claim. All four parameters (b, c, d, e)
are per-cluster fit values; no universal numerical value across populations is
published.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
None assigned. The natural logarithm and the structural "1" in the logistic
denominator are fixed algebraic components, not tunable constants.

LOCAL_FITTABLE — per-cluster, fitted by fit() via MLE
------------------------------------------------------
- b : slope (dimensionless); negative for increasing mortality with dose
      (standard insecticide bioassay convention). The |b| controls steepness.
- c : lower asymptote (dimensionless, [0, 1]); typically 0 for standard
      mortality bioassay (full kill at high dose).
- d : upper asymptote (dimensionless, [0, 1]); typically 1 for standard
      mortality bioassay (zero mortality at zero dose).
- e : ED50 / LC50 (ppb); must be > 0. Structural ED50 of the curve.

init = None on all: fit() builds data-derived starting values from the
empirical dose-response relationship.
"""

import numpy as np
from scipy.optimize import minimize

USED_INPUTS = ["log_conc_ppb"]
PAPER_REF = "summary_formula_ritz_2015.md"
EQUATION_LOC = (
    "Ritz et al. (2015) Eq. (2), PDF p. 2 — LL.4: "
    "f(x; b,c,d,e) = c + (d-c)/(1 + exp(b*(log(x) - log(e)))); "
    "log = natural logarithm."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "b": {"init": None},
    "c": {"init": None},
    "d": {"init": None},
    "e": {"init": None},
}


def _ll4(log_x, b, c, d, e):
    """f = c + (d - c) / (1 + exp(b * (log_x - log(e)))).

    log_x = ln(concentration); e > 0 so ln(e) is well-defined.
    Numerically stable: use -expit form to avoid exp overflow.
    """
    log_e = np.log(np.clip(e, 1e-15, None))
    arg = b * (log_x - log_e)
    # clip arg to avoid overflow in exp
    arg = np.clip(arg, -500.0, 500.0)
    return c + (d - c) / (1.0 + np.exp(arg))


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """MLE fit of the four-parameter log-logistic model.

    Deterministic, data-derived start:
      d0 = min(1.0, max observed response + margin)  ~ upper asymptote
      c0 = max(0.0, min observed response - margin)  ~ lower asymptote
      e0 = exp(median(log_x at responses near 0.5))  ~ ED50 estimate
      b0 = -1.0  (moderate negative slope: increasing mortality with dose)

    Minimises binary cross-entropy (= probit/logit MLE for fractional data).
    """
    log_x = np.asarray(X_fit[:, 0], dtype=float)  # ln(concentration)
    y = np.clip(np.asarray(y_fit, dtype=float), 1e-6, 1.0 - 1e-6)

    # Data-derived starting values
    c0 = max(0.0, float(np.percentile(y, 5)) - 0.05)
    d0 = min(1.0, float(np.percentile(y, 95)) + 0.05)
    mid = 0.5 * (c0 + d0)
    idx = int(np.argmin(np.abs(y - mid)))
    e0 = float(np.exp(log_x[idx])) if np.isfinite(log_x[idx]) else 1.0
    if e0 <= 0 or not np.isfinite(e0):
        e0 = float(np.exp(np.median(log_x)))
    b0 = -1.0  # negative slope: mortality increases with dose

    def neg_log_lik(params):
        b, c, d, e = params
        if e <= 0 or d <= c:
            return 1e12
        p = np.clip(_ll4(log_x, b, c, d, e), 1e-9, 1.0 - 1e-9)
        return -float(np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))

    try:
        res = minimize(
            neg_log_lik,
            x0=[b0, c0, d0, e0],
            method="Nelder-Mead",
            options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-7},
        )
        b_fit, c_fit, d_fit, e_fit = res.x
        c_fit = float(np.clip(c_fit, 0.0, 1.0))
        d_fit = float(np.clip(d_fit, 0.0, 1.0))
        e_fit = float(max(e_fit, 1e-9))
        if not np.all(np.isfinite([b_fit, c_fit, d_fit, e_fit])):
            raise RuntimeError("non-finite fit")
        return {"b": float(b_fit), "c": c_fit, "d": d_fit, "e": e_fit}
    except Exception:  # noqa: BLE001
        return {"b": float(b0), "c": float(c0), "d": float(d0), "e": float(e0)}


def predict(X: np.ndarray, b: float, c: float, d: float, e: float) -> np.ndarray:
    """Four-parameter log-logistic dose-response (LL.4).

    f(x; b, c, d, e) = c + (d - c) / (1 + exp(b * (ln(x) - ln(e))))

    X: (n, 1) — column [log_conc_ppb] = ln(concentration in ppb).
    Returns predicted mortality probability in [c, d] subset of [0, 1].
    """
    log_x = np.asarray(X[:, 0], dtype=float)
    return _ll4(log_x, b, c, d, e)
