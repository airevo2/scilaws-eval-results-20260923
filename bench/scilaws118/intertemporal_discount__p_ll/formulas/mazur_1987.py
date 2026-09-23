"""Mazur (1987) hyperbolic discounting — larger-later choice rate.

Mazur (1987) models the subjective (present) value of a reward of amount
V received after delay t as a HYPERBOLIC decline:

    SV(V, t) = V / (1 + k * t)

with k > 0 the per-study discount rate. (Frederick, Loewenstein &
O'Donoghue 2002 §4.1 fn. 12 reproduce D(t) = 1/(1+k t) and attribute it
to Herrnstein 1981 / Mazur 1987 — see reference/frederick_2002.pdf.)

For a smaller-sooner (SS) vs larger-later (LL) choice, the probability
of choosing LL is the logistic of the subjective-value difference:

    p_ll = 1 / (1 + exp(-(SV_LL - SV_SS) / sigma))

sigma > 0 is a logistic choice-noise scale; it also absorbs each study's
currency unit (rewards are in study-specific currency). The benchmark
target `p_ll` is the per-bin aggregated choice rate.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The hyperbolic form is the scientific claim; k and sigma are
per-study fits (the denominator "1" is a structural constant, inlined).

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster, computed by fit() via nonlinear least squares
---------------------------------------------------------------------------
- k     : hyperbolic discount rate (1/day).
- sigma : logistic choice-noise scale (study currency units).
init = None on both: fit() builds its own data-derived start.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

USED_INPUTS = ["ss_value", "ss_time_days", "ll_value", "ll_time_days"]
PAPER_REF = "summary_formula_mazur_1987.md"
EQUATION_LOC = (
    "Mazur (1987) hyperbolic discount function SV = V/(1+k*t); reproduced "
    "as D(t)=1/(1+k t) in Frederick, Loewenstein & O'Donoghue (2002) "
    "§4.1 fn. 12 (reference/frederick_2002.pdf)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "k":     {"init": None},
    "sigma": {"init": None},
}


def _p_ll(v_ss, t_ss, v_ll, t_ll, k, sigma):
    sv_ss = v_ss / (1.0 + k * t_ss)
    sv_ll = v_ll / (1.0 + k * t_ll)
    return expit((sv_ll - sv_ss) / sigma)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear least-squares fit of (k, sigma) to the per-bin choice rate.

    Start: k from a small default; sigma from the study's reward scale
    (the median absolute raw amount difference), which sets the currency
    unit so the logistic argument is O(1).
    """
    v_ss = np.asarray(X_fit[:, 0], dtype=float)
    t_ss = np.asarray(X_fit[:, 1], dtype=float)
    v_ll = np.asarray(X_fit[:, 2], dtype=float)
    t_ll = np.asarray(X_fit[:, 3], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    s0 = float(np.median(np.abs(v_ll - v_ss)))
    s0 = s0 if s0 > 1e-9 else 1.0
    k0 = 0.01

    def residual(p):
        k, sigma = p
        return _p_ll(v_ss, t_ss, v_ll, t_ll, k, sigma) - y

    try:
        sol = least_squares(residual, [k0, s0],
                            bounds=([1e-9, 1e-9], [1e3, 1e12]),
                            method="trf", max_nfev=4000)
        return {"k": float(sol.x[0]), "sigma": float(sol.x[1])}
    except Exception:                              # noqa: BLE001
        return {"k": k0, "sigma": s0}


def predict(X: np.ndarray, k: float, sigma: float) -> np.ndarray:
    """p_ll = logistic((SV_LL - SV_SS)/sigma), SV = V/(1+k*t).

    X: (n, 4) — columns ss_value, ss_time_days, ll_value, ll_time_days.
    """
    return _p_ll(np.asarray(X[:, 0], dtype=float), np.asarray(X[:, 1], dtype=float),
                 np.asarray(X[:, 2], dtype=float), np.asarray(X[:, 3], dtype=float),
                 k, sigma)
