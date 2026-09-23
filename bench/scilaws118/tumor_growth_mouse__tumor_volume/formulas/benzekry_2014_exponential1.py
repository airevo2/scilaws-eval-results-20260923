"""Benzekry 2014 — Exponential-1 tumor growth model (one-parameter exponential).

Benzekry S, Lamont C, Beheshti A, Tracz A, Ebos JML, Hlatky L, Hahnfeldt P.
(2014). Classical Mathematical Models for Description and Prediction of
Experimental Tumor Growth. *PLoS Computational Biology*, 10(8): e1003800.
DOI:10.1371/journal.pcbi.1003800.

Eq. 1 (PDF p. 3), sub-model (a): initial volume fixed at V0 = 1 mm³ and no
linear phase (a1 = +inf). Pure exponential growth:

    dV/dt = a0 * V,   V(t=0) = 1 mm³

Closed-form solution:

    V(t) = exp(a0 * t)

This is the simplest of the three exponential sub-models; it has exactly one
free per-animal parameter (a0).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
(empty.)

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
- V0: 1.0 mm³ — assumed (given) initial volume (1 mm³ ~ 10^6 injected cells),
  NOT a fitted coefficient. Benzekry 2014 PDF p. 3 / L255: "we considered
  1 mm^3 ... as a reasonable approximation for V(t = 0)." An invariant-but-
  GIVEN injection volume, so it is filed under OTHER (structural/given)
  rather than LAW (fitted-law).

LOCAL_FITTABLE — per-cluster, fitted by fit()
----------------------------------------------
- a0 : exponential proliferation rate (day^-1, > 0).
  init = None: fit() derives a data-driven start.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["time_day"]
PAPER_REF = "summary_formula_dataset_benzekry_2014.md"
EQUATION_LOC = (
    "Benzekry 2014 Eq. 1, PDF p. 3 — exponential sub-model (a): "
    "dV/dt = a0*V, V(0)=1; solution V(t) = exp(a0*t)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    "V0": 1.0,   # mm^3; assumed initial condition, Benzekry 2014 PDF p. 3
}
LOCAL_FITTABLE = {
    "a0": {"init": None},
}


def _predict_core(t: np.ndarray, V0: float, a0: float) -> np.ndarray:
    return V0 * np.exp(a0 * t)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear least-squares fit of the Exponential-1 model.

    Data-driven init: a0_0 estimated by log-linear regression of log(V) vs t.
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # Log-linear init (robust to zero/negative via clipping)
    log_y = np.log(np.clip(y, 1e-6, None))
    if t.std() > 1e-9:
        a0_0 = float(np.polyfit(t, log_y, 1)[0])
    else:
        a0_0 = 0.1
    a0_0 = max(1e-4, min(a0_0, 5.0))

    def residual(p):
        return V0 * np.exp(p[0] * t) - y

    try:
        sol = least_squares(residual, [a0_0], bounds=([1e-6], [20.0]),
                            method="trf", max_nfev=2000)
        return {"a0": float(sol.x[0])}
    except Exception:                               # noqa: BLE001
        return {"a0": a0_0}


def predict(X: np.ndarray, a0: float) -> np.ndarray:
    """V(t) = V0 * exp(a0 * t).

    X: (n, 1) — column [time_day]. V0 read from OTHER_CONSTANTS.
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X[:, 0], dtype=float)
    return _predict_core(t, V0, a0)
