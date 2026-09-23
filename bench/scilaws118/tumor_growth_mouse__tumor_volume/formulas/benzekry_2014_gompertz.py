"""Benzekry 2014 — Gompertz tumor growth model.

Benzekry S, Lamont C, Beheshti A, Tracz A, Ebos JML, Hlatky L, Hahnfeldt P.
(2014). Classical Mathematical Models for Description and Prediction of
Experimental Tumor Growth. *PLoS Computational Biology*, 10(8): e1003800.
DOI:10.1371/journal.pcbi.1003800.

Eq. 4 (PDF p. 3):

    dV/dt = a * exp(-beta * t) * V,   V(t=0) = 1 mm³

Closed-form solution with V0 = 1 mm³ (PDF p. 3):

    V(t) = V0 * exp( (a/beta) * (1 - exp(-beta*t)) )

Asymptotic carrying capacity (from the closed form, as t -> inf):

    K = V0 * exp(a/beta)

"a is the initial proliferation rate (at V = 1 mm³) and b [beta] is the
rate of exponential decay of this proliferation rate." (PDF p. 3.)

"The essential characteristic of the Gompertz model is that it exhibits
exponential decay of the relative growth rate." (PDF p. 3.)

Best descriptive model for LLC (lung) data; also excellent for LM2-4LUC
(breast). Median parameters from Table 3 (PDF p. 9):
a ~ 0.743 day^-1, beta ~ 0.0792 day^-1 (LLC).

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
- a    : initial proliferation rate (day^-1, > 0).
- beta : exponential decay rate of proliferation rate (day^-1, > 0).
  init = None on both: fit() derives data-driven starts.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["time_day"]
PAPER_REF = "summary_formula_dataset_benzekry_2014.md"
EQUATION_LOC = (
    "Benzekry 2014 Eq. 4, PDF p. 3 — Gompertz: dV/dt = a*exp(-beta*t)*V, "
    "V(0)=1; solution V(t) = V0*exp((a/beta)*(1 - exp(-beta*t)))."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    "V0": 1.0,   # mm^3; assumed initial condition, Benzekry 2014 PDF p. 3
}
LOCAL_FITTABLE = {
    "a":    {"init": None},
    "beta": {"init": None},
}


def _predict_core(t: np.ndarray, V0: float, a: float, beta: float) -> np.ndarray:
    return V0 * np.exp((a / beta) * (1.0 - np.exp(-beta * t)))


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear LS fit of the Gompertz model.

    Init: a_0 from max observed log-slope; beta_0 = 0.05 (moderate decay
    of relative growth rate — within the published LLC range 0.05-0.15).
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # Estimate a from early slope of log(y) vs t
    log_y = np.log(np.clip(y, 1e-6, None))
    if t.std() > 1e-9:
        a_0 = float(np.polyfit(t, log_y, 1)[0])
    else:
        a_0 = 0.5
    a_0 = max(1e-3, min(a_0, 5.0))
    beta_0 = 0.05

    def residual(p):
        return _predict_core(t, V0, p[0], p[1]) - y

    try:
        sol = least_squares(residual, [a_0, beta_0],
                            bounds=([1e-6, 1e-6], [20.0, 5.0]),
                            method="trf", max_nfev=4000)
        return {"a": float(sol.x[0]), "beta": float(sol.x[1])}
    except Exception:                               # noqa: BLE001
        return {"a": a_0, "beta": beta_0}


def predict(X: np.ndarray, a: float, beta: float) -> np.ndarray:
    """V(t) = V0 * exp((a/beta) * (1 - exp(-beta*t))).

    X: (n, 1) — column [time_day]. V0 read from OTHER_CONSTANTS.
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X[:, 0], dtype=float)
    return _predict_core(t, V0, a, beta)
