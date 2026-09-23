"""Benzekry 2014 — Exponential-linear biphasic tumor growth model.

Benzekry S, Lamont C, Beheshti A, Tracz A, Ebos JML, Hlatky L, Hahnfeldt P.
(2014). Classical Mathematical Models for Description and Prediction of
Experimental Tumor Growth. *PLoS Computational Biology*, 10(8): e1003800.
DOI:10.1371/journal.pcbi.1003800.

Eq. 1 (PDF p. 3), sub-model (c): initial volume fixed at V0 = 1 mm³, both
exponential and linear phases active:

    dV/dt = a0 * V,   t <= tau
    dV/dt = a1,       t > tau
    V(t=0) = 1 mm³

The switch time tau is uniquely determined by C1-continuity of the solution
(PDF p. 3):

    tau = (1/a0) * log(a1 / (a0 * V0))

Explicit piecewise solution (V0 = 1 mm³):

    t <= tau:  V(t) = V0 * exp(a0 * t)
    t > tau :  V(t) = V(tau) + a1 * (t - tau)
                     = (a1/a0) * [1 + a0 * (t - tau)]

Median fitted parameters from Table 3 (PDF p. 9): a0 ~ 0.49 day^-1 (LLC),
a1 ~ 115.6 mm³/day (LLC).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
(empty.)

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
- V0: 1.0 mm³ — assumed (given) initial condition, NOT a fitted coefficient.
  Benzekry 2014 PDF p. 3 / L255 calls 1 mm³ "a reasonable approximation for
  V(t=0)" (~10^6 injected cells) — an invariant-but-GIVEN injection volume, so
  it is filed under OTHER (structural/given) rather than LAW (fitted-law).

LOCAL_FITTABLE — per-cluster, fitted by fit()
----------------------------------------------
- a0 : exponential phase proliferation rate (day^-1, > 0).
- a1 : linear phase growth rate (mm³/day, > 0).  Must satisfy a1 > a0*V0
       so that tau > 0; otherwise the linear phase begins immediately.
  init = None on both: fit() derives data-driven starts.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["time_day"]
PAPER_REF = "summary_formula_dataset_benzekry_2014.md"
EQUATION_LOC = (
    "Benzekry 2014 Eq. 1, PDF p. 3 — exponential-linear sub-model (c): "
    "dV/dt = a0*V (t<=tau), a1 (t>tau); V(0)=1; "
    "tau = log(a1/(a0*V0)) / a0."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    "V0": 1.0,   # mm^3; assumed initial condition, Benzekry 2014 PDF p. 3
}
LOCAL_FITTABLE = {
    "a0": {"init": None},
    "a1": {"init": None},
}


def _predict_core(t: np.ndarray, V0: float, a0: float, a1: float) -> np.ndarray:
    # Switch time (scalar); if a1 <= a0*V0 the linear phase never activates
    # within the modeled domain; clamp tau to 0 in that case.
    if a1 > a0 * V0:
        tau = np.log(a1 / (a0 * V0)) / a0
    else:
        tau = 0.0
    V_tau = V0 * np.exp(a0 * tau)   # = a1 / a0 when a1 > a0*V0

    V = np.where(
        t <= tau,
        V0 * np.exp(a0 * t),
        V_tau + a1 * (t - tau),
    )
    return V


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear LS fit of the exponential-linear model.

    Init: a0_0 from log-linear regression on the first half of the time
    series; a1_0 from the slope of the second half (linear phase proxy).
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    order = np.argsort(t)
    t_s, y_s = t[order], y[order]
    n = len(t_s)

    # Estimate a0 from first half (log slope)
    n_exp = max(2, n // 2)
    log_y = np.log(np.clip(y_s[:n_exp], 1e-6, None))
    if t_s[:n_exp].std() > 1e-9:
        a0_0 = float(np.polyfit(t_s[:n_exp], log_y, 1)[0])
    else:
        a0_0 = 0.3
    a0_0 = max(1e-3, min(a0_0, 5.0))

    # Estimate a1 from last half (linear slope)
    if n >= 4 and t_s[n_exp:].std() > 1e-9:
        a1_0 = float(np.polyfit(t_s[n_exp:], y_s[n_exp:], 1)[0])
    else:
        a1_0 = float(np.ptp(y_s) / (np.ptp(t_s) + 1e-9)) if n > 1 else 50.0
    a1_0 = max(1e-3, min(a1_0, 5000.0))
    # Ensure a1 > a0 * V0 so tau > 0
    a1_0 = max(a1_0, a0_0 * V0 * 1.1)

    def residual(p):
        return _predict_core(t, V0, p[0], p[1]) - y

    try:
        sol = least_squares(residual, [a0_0, a1_0],
                            bounds=([1e-6, 1e-6], [20.0, 1e5]),
                            method="trf", max_nfev=4000)
        return {"a0": float(sol.x[0]), "a1": float(sol.x[1])}
    except Exception:                               # noqa: BLE001
        return {"a0": a0_0, "a1": a1_0}


def predict(X: np.ndarray, a0: float, a1: float) -> np.ndarray:
    """Piecewise exponential-linear volume.

    X: (n, 1) — column [time_day]. V0 read from OTHER_CONSTANTS.
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X[:, 0], dtype=float)
    return _predict_core(t, V0, a0, a1)
