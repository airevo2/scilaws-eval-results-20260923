"""Benzekry 2014 — Exponential-V0 tumor growth model (V0 free).

Benzekry S, Lamont C, Beheshti A, Tracz A, Ebos JML, Hlatky L, Hahnfeldt P.
(2014). Classical Mathematical Models for Description and Prediction of
Experimental Tumor Growth. *PLoS Computational Biology*, 10(8): e1003800.
DOI:10.1371/journal.pcbi.1003800.

Eq. 1 (PDF p. 3), sub-model (b): no linear phase (a1 = +inf), initial volume
V0 left as a free per-animal parameter. Pure exponential growth:

    dV/dt = a0 * V,   V(t=0) = V0  (free)

Closed-form solution:

    V(t) = V0 * exp(a0 * t)

Paper notation (PDF p. 3): "b) free initial volume and no linear phase,
referred to as exponential V0." Median fitted V0 reported as 13.2 mm³ (LLC)
and 68.2 mm³ (breast) in Table 3–4. Two free per-animal parameters: a0, V0.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. V0 is explicitly free (per-cluster fittable) for this sub-model.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster, fitted by fit()
----------------------------------------------
- a0 : exponential proliferation rate (day^-1, > 0).
- V0 : initial tumor volume (mm³, > 0).
  init = None on both: fit() derives data-driven starts.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["time_day"]
PAPER_REF = "summary_formula_dataset_benzekry_2014.md"
EQUATION_LOC = (
    "Benzekry 2014 Eq. 1, PDF p. 3 — exponential sub-model (b): "
    "dV/dt = a0*V, V(0)=V0 free; solution V(t) = V0*exp(a0*t)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "a0": {"init": None},
    "V0": {"init": None},
}


def _predict_core(t: np.ndarray, a0: float, V0: float) -> np.ndarray:
    return V0 * np.exp(a0 * t)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear LS fit of Exponential-V0.

    Init: V0_0 = first (smallest-t) observed volume; a0_0 from log-linear
    regression of log(y) vs t.
    """
    t = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # V0 init: value at smallest t (or intercept from log-linear fit)
    order = np.argsort(t)
    t_s, y_s = t[order], y[order]

    log_y = np.log(np.clip(y_s, 1e-6, None))
    if t_s.std() > 1e-9:
        coeffs = np.polyfit(t_s, log_y, 1)
        a0_0 = float(coeffs[0])
        V0_0 = float(np.exp(coeffs[1]))
    else:
        a0_0 = 0.1
        V0_0 = float(y_s[0]) if len(y_s) > 0 else 1.0
    a0_0 = max(1e-4, min(a0_0, 5.0))
    V0_0 = max(0.1, min(V0_0, 5000.0))

    def residual(p):
        return p[1] * np.exp(p[0] * t) - y

    try:
        sol = least_squares(residual, [a0_0, V0_0],
                            bounds=([1e-6, 1e-3], [20.0, 1e4]),
                            method="trf", max_nfev=2000)
        return {"a0": float(sol.x[0]), "V0": float(sol.x[1])}
    except Exception:                               # noqa: BLE001
        return {"a0": a0_0, "V0": V0_0}


def predict(X: np.ndarray, a0: float, V0: float) -> np.ndarray:
    """V(t) = V0 * exp(a0 * t).

    X: (n, 1) — column [time_day].
    """
    t = np.asarray(X[:, 0], dtype=float)
    return _predict_core(t, a0, V0)
