"""Maillet (1905) linear-reservoir baseflow recession — Q(t).

Roques, C., Rupp, D. E., de Dreuzy, J.-R., Longuevergne, L., Jachens, E. R.,
Grant, G., Aquilina, L., and Selker, J. S. (2022). Recession discharge from
compartmentalized bedrock hillslopes. *Hydrology and Earth System Sciences*
26(16):4391–4405. DOI:10.5194/hess-26-4391-2022.

Roques et al. (2022) Appendix A, Eq. (A2) (PDF p. 11) gives the standard
Maillet (1905) single-linear-reservoir exponential recession formula for a
fast (f) or slow (s) drainage compartment:

    Q_f(t) = Q_f(0) * exp(-t / k_f)
    Q_s(t) = Q_s(0) * exp(-t / k_s)

For the single-reservoir (non-compartmentalized) benchmark case these reduce
to the primary formula:

    Q(t) = Q_0 * exp(-t / k)

where:
  - Q_0  = Q(t=0), the observed discharge at the recession-start day (day 0),
            treated as a known per-cluster covariate, not a fit parameter.
  - t    = elapsed time in days since the start of the recession episode
            (integer day index ≥ 0).
  - k    = the recession time constant in days (k > 0); encodes the aquifer
            geometry and hydraulic conductivity. Equivalent to 1/a in the
            Brutsaert-Nieber (1977) power-law notation, confirmed by Roques
            et al. (2022) PDF p. 2: "When b = 1, the aquifer drainage is
            exponential, with a characteristic timescale τ = 1/a."

The formula is the b = 1 limit of the BN77 power-law ODE –dQ/dt = a Q^b
(Eq. 1 of Roques 2022, PDF p. 1), integrated analytically to give Q(t).

Events are selected by the Vogel-Kroll (1992) ≥5-day strict monotonic
decrease rule, so the linear-reservoir assumption (b ≈ 1) is a reasonable
approximation for these episodes; per-event k is fitted by log-linear
regression within each recession episode.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The single-exponential FORM is the scientific claim. The structural
constants e (base of natural logarithm) and the sign of the exponent (-1)
are algebraic consequences of the linear ODE and are not free parameters.
k is a per-event aquifer property; Q_0 is read directly from data.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The exponential form exp(-t/k) is algebraically exact for a
linear first-order ODE; no additional universal numerical constants appear.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- k : recession time constant (days, > 0). Roques et al. (2022) PDF p. 2
      cites τ = 1/a ≈ 5–60 days across fast mountain to slow glacial
      catchments. Estimated per recession event by log-linear OLS on
      log(Q) vs. t (since log Q = log Q_0 - t/k is linear in t),
      then polished by bounded nonlinear LS on Q to avoid bias from
      log-space heteroscedasticity.

init = None: fit() builds its own deterministic, data-derived start from
log-linear OLS — a single-start NLS polishing step is well-conditioned
because the Maillet formula is smooth and unimodal in k on recession data.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["t", "Q_0"]
PAPER_REF = "summary_formula_roques_2022.md"
EQUATION_LOC = (
    "Roques et al. (2022) Appendix A, Eq. (A2), PDF p. 11: "
    "Q_f(t) = Q_f(0) * exp(-t / k_f); single-reservoir limit Q(t) = Q_0 * exp(-t / k). "
    "b=1 linear-reservoir identification on PDF p. 2: tau = 1/a."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "k": {"init": None},
}


def _Q_t(t, Q_0, k):
    """Q(t) = Q_0 * exp(-t / k); k must be > 0."""
    return Q_0 * np.exp(-t / k)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear LS fit of the Maillet single-exponential recession.

    Deterministic, data-derived start:
      k0 — estimated by log-linear OLS on log(Q) ~ log(Q_0) - t/k, i.e.,
           a simple linear regression of log(Q/Q_0) on -t gives slope 1/k.
           Rows where Q_0 <= 0 or y <= 0 are excluded from the OLS start.
    Single-start NLS is sufficient: the Maillet form is smooth and the
    residual has a single minimum in k for well-behaved recession data.

    Parameters
    ----------
    X_fit : (n, 2) array — columns [t, Q_0].
    y_fit : (n,) array — observed Q_t.

    Returns
    -------
    dict with key "k" (float, days > 0).
    """
    t   = np.asarray(X_fit[:, 0], dtype=float)
    Q_0 = np.asarray(X_fit[:, 1], dtype=float)
    y   = np.asarray(y_fit, dtype=float)

    # Log-linear OLS for deterministic k0 estimate.
    # log(Q / Q_0) = -t / k  =>  slope of log(Q/Q_0) on -t gives 1/k.
    valid = (Q_0 > 0) & (y > 0) & np.isfinite(t) & np.isfinite(Q_0) & np.isfinite(y)
    if valid.sum() >= 2:
        log_ratio = np.log(y[valid] / Q_0[valid])   # = -t/k + noise
        neg_t = -t[valid]
        # OLS: log_ratio = (1/k) * neg_t  (no intercept; Q_0 anchors the level)
        # => 1/k = (neg_t . log_ratio) / (neg_t . neg_t)
        denom = float(np.dot(neg_t, neg_t))
        if denom > 0:
            inv_k = float(np.dot(neg_t, log_ratio)) / denom
            k0 = 1.0 / inv_k if inv_k > 0 else 20.0
        else:
            k0 = 20.0
    else:
        k0 = 20.0

    if not np.isfinite(k0) or k0 <= 0:
        k0 = 20.0

    # Clamp k0 into bounds before passing to NLS.
    # Bounds: k in [0.1, 365] days — physically motivated by Roques 2022 p. 2
    # (τ = 1/a for fast mountain to slow glacial catchments).
    k_lo, k_hi = 0.1, 365.0
    k0 = float(np.clip(k0, k_lo, k_hi))

    def residual(p):
        return _Q_t(t, Q_0, p[0]) - y

    try:
        sol = least_squares(
            residual, [k0],
            bounds=([k_lo], [k_hi]),
            method="trf",
            max_nfev=4000,
        )
        k_fit = float(sol.x[0])
        if not np.isfinite(k_fit) or k_fit <= 0:
            raise RuntimeError("non-finite or non-positive k")
        return {"k": k_fit}
    except Exception:                               # noqa: BLE001
        return {"k": k0}


def predict(X: np.ndarray, k: float) -> np.ndarray:
    """Maillet exponential recession: Q(t) = Q_0 * exp(-t / k).

    Parameters
    ----------
    X : (n, 2) array — columns [t, Q_0].
    k : recession time constant (days, > 0).

    Returns
    -------
    (n,) array of predicted discharge Q_t.
    """
    t   = np.asarray(X[:, 0], dtype=float)
    Q_0 = np.asarray(X[:, 1], dtype=float)
    return _Q_t(t, Q_0, k)
