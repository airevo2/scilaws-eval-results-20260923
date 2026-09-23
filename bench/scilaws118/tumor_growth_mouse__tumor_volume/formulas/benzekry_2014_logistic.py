"""Benzekry 2014 — Logistic tumor growth model.

Benzekry S, Lamont C, Beheshti A, Tracz A, Ebos JML, Hlatky L, Hahnfeldt P.
(2014). Classical Mathematical Models for Description and Prediction of
Experimental Tumor Growth. *PLoS Computational Biology*, 10(8): e1003800.
DOI:10.1371/journal.pcbi.1003800.

Eq. 2 (PDF p. 3):

    dV/dt = a * V * (1 - V/K),   V(t=0) = 1 mm³

Closed-form solution with V0 = 1 mm³:

    V(t) = K / [1 + (K - 1) * exp(-a * t)]

The relative growth rate (1/V)(dV/dt) = a*(1 - V/K) decreases linearly
with volume. The logistic model is the n=1 special case of the generalized
logistic (Eq. 3). Median fitted parameters from Table 3 (PDF p. 9):
a ~ 0.502 day^-1 (LLC), K ~ 1297 mm³ (LLC).

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
- a : proliferation rate coefficient (day^-1, > 0).
- K : asymptotic carrying capacity (mm³, > V0 = 1).
  init = None on both: fit() derives data-driven starts.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["time_day"]
PAPER_REF = "summary_formula_dataset_benzekry_2014.md"
EQUATION_LOC = (
    "Benzekry 2014 Eq. 2, PDF p. 3 — logistic: dV/dt = a*V*(1 - V/K), "
    "V(0)=1; solution V(t) = K / [1 + (K-1)*exp(-a*t)]."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    "V0": 1.0,   # mm^3; assumed initial condition, Benzekry 2014 PDF p. 3
}
LOCAL_FITTABLE = {
    "a": {"init": None},
    "K": {"init": None},
}


def _predict_core(t: np.ndarray, V0: float, a: float, K: float) -> np.ndarray:
    denom = 1.0 + (K - V0) / V0 * np.exp(-a * t)
    return K / np.maximum(denom, 1e-12)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear LS fit of the logistic growth model.

    Init: K_0 = max(observed V) * 2 (carrying capacity not yet reached);
    a_0 from log-linear regression on early points.
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    K_0 = float(np.max(y)) * 2.0
    K_0 = max(K_0, 2.0)

    log_y = np.log(np.clip(y, 1e-6, None))
    if t.std() > 1e-9:
        a_0 = float(np.polyfit(t, log_y, 1)[0])
    else:
        a_0 = 0.3
    a_0 = max(1e-4, min(a_0, 5.0))

    def residual(p):
        return _predict_core(t, V0, p[0], p[1]) - y

    try:
        sol = least_squares(residual, [a_0, K_0],
                            bounds=([1e-6, 1.01], [20.0, 1e6]),
                            method="trf", max_nfev=4000)
        return {"a": float(sol.x[0]), "K": float(sol.x[1])}
    except Exception:                               # noqa: BLE001
        return {"a": a_0, "K": K_0}


def predict(X: np.ndarray, a: float, K: float) -> np.ndarray:
    """V(t) = K / [1 + (K - V0)/V0 * exp(-a*t)].

    X: (n, 1) — column [time_day]. V0 read from OTHER_CONSTANTS.
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X[:, 0], dtype=float)
    return _predict_core(t, V0, a, K)
