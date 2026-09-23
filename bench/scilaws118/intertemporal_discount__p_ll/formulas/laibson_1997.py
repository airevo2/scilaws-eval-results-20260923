"""Laibson (1997) quasi-hyperbolic (beta-delta) discounting — LL choice rate.

Laibson (1997), "Golden Eggs and Hyperbolic Discounting", QJE 112(2),
proposes the quasi-hyperbolic discount weights {1, beta*delta,
beta*delta^2, ...}: the subjective value of a reward of amount V at
delay t is

    SV(V, t) = V                  if t == 0   (no present-bias penalty)
    SV(V, t) = V * beta * delta^t  if t  > 0

0 < beta <= 1 is the present-bias factor (beta = 1 recovers pure
exponential discounting); 0 < delta < 1 is the per-day long-run patience
factor. The kink at t = 0 — the immediate reward keeps full value while
any delayed reward is additionally scaled by beta — is the present-bias
that distinguishes quasi-hyperbolic from both exponential and Mazur
hyperbolic discounting.

For a smaller-sooner (SS) vs larger-later (LL) choice the probability of
choosing LL is the logistic of the subjective-value difference:

    p_ll = 1 / (1 + exp(-(SV_LL - SV_SS) / sigma))

sigma > 0 is a logistic choice-noise scale that also absorbs each
study's currency unit. The benchmark target `p_ll` is the per-bin
aggregated choice rate.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The beta-delta form is the scientific claim; beta, delta, sigma
are per-study fits.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster, computed by fit() via nonlinear least squares
---------------------------------------------------------------------------
- beta  : present-bias factor (0..1; 1 = exponential discounting).
- delta : per-day long-run discount factor (0..1).
- sigma : logistic choice-noise scale (study currency units).
init = None on all three: fit() builds its own data-derived start.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

USED_INPUTS = ["ss_value", "ss_time_days", "ll_value", "ll_time_days"]
PAPER_REF = "summary_formula_laibson_1997.md"
EQUATION_LOC = (
    "Laibson (1997), QJE 112(2):443-477 — quasi-hyperbolic discount "
    "weights {1, beta*delta, beta*delta^2, ...}; SV = V (t=0) / "
    "V*beta*delta^t (t>0)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "beta":  {"init": None},
    "delta": {"init": None},
    "sigma": {"init": None},
}


def _sv(v, t, beta, delta):
    delayed = v * beta * np.power(delta, np.clip(t, 0.0, None))
    return np.where(t < 0.5, v, delayed)           # immediate reward keeps full value


def _p_ll(v_ss, t_ss, v_ll, t_ll, beta, delta, sigma):
    diff = _sv(v_ll, t_ll, beta, delta) - _sv(v_ss, t_ss, beta, delta)
    return expit(diff / sigma)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear least-squares fit of (beta, delta, sigma) to the choice rate."""
    v_ss = np.asarray(X_fit[:, 0], dtype=float)
    t_ss = np.asarray(X_fit[:, 1], dtype=float)
    v_ll = np.asarray(X_fit[:, 2], dtype=float)
    t_ll = np.asarray(X_fit[:, 3], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    s0 = float(np.median(np.abs(v_ll - v_ss)))
    s0 = s0 if s0 > 1e-9 else 1.0
    p0 = [0.9, 0.999, s0]

    def residual(p):
        beta, delta, sigma = p
        return _p_ll(v_ss, t_ss, v_ll, t_ll, beta, delta, sigma) - y

    try:
        sol = least_squares(residual, p0,
                            bounds=([1e-4, 1e-4, 1e-9], [1.0, 1.0, 1e12]),
                            method="trf", max_nfev=6000)
        return {"beta": float(sol.x[0]), "delta": float(sol.x[1]),
                "sigma": float(sol.x[2])}
    except Exception:                              # noqa: BLE001
        return {"beta": 0.9, "delta": 0.999, "sigma": s0}


def predict(X: np.ndarray, beta: float, delta: float, sigma: float) -> np.ndarray:
    """p_ll = logistic((SV_LL - SV_SS)/sigma), SV the beta-delta value.

    X: (n, 4) — columns ss_value, ss_time_days, ll_value, ll_time_days.
    """
    return _p_ll(np.asarray(X[:, 0], dtype=float), np.asarray(X[:, 1], dtype=float),
                 np.asarray(X[:, 2], dtype=float), np.asarray(X[:, 3], dtype=float),
                 beta, delta, sigma)
