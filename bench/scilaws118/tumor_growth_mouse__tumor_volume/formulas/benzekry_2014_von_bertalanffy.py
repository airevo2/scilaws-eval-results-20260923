"""Benzekry 2014 — Von Bertalanffy tumor growth model.

Benzekry S, Lamont C, Beheshti A, Tracz A, Ebos JML, Hlatky L, Hahnfeldt P.
(2014). Classical Mathematical Models for Description and Prediction of
Experimental Tumor Growth. *PLoS Computational Biology*, 10(8): e1003800.
DOI:10.1371/journal.pcbi.1003800.

Eq. 6 (PDF p. 4):

    dV/dt = a * V^gamma - b * V,   V(t=0) = 1 mm³

Explicit solution with V0 = 1 mm³ (PDF p. 4, Eq. 8):

    V(t) = [ a/b + (V0^(1-gamma) - a/b) * exp(-b*(1-gamma)*t) ]^(1/(1-gamma))

"Employing our usual assumption that V(t=0) = 1 mm³, we will refer to this
model as the von Bertalanffy model." (PDF p. 4.)

The b=0 limit reduces to the power-law model (PDF p. 4: "taking b=0 ...
termed the power law model"). Three free per-animal parameters: a, b, gamma.

Median fitted parameters from Table 3 (PDF p. 9): a ~ (not directly stated
for Von Bert.), b ~ 6.75 day^-1 (LLC, large CV), gamma ~ 0.947 (LLC).

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
- a     : synthesis rate coefficient (mm^{3(1-gamma)} day^-1, > 0).
- b     : catabolism rate coefficient (day^-1, >= 0).
- gamma : allometric exponent (dimensionless, in (0, 1) for subexponential).
  init = None on all: fit() derives data-driven starts.

Note: Three-parameter model; poorly identifiable from short series
(Benzekry 2014 Table 3 large coefficient of variation, PDF p. 9).
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["time_day"]
PAPER_REF = "summary_formula_dataset_benzekry_2014.md"
EQUATION_LOC = (
    "Benzekry 2014 Eq. 6, PDF p. 4 — von Bertalanffy: "
    "dV/dt = a*V^gamma - b*V, V(0)=1; "
    "solution Eq. 8: V(t) = [a/b + (V0^(1-gamma) - a/b)*exp(-b*(1-gamma)*t)]^(1/(1-gamma))."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    "V0": 1.0,   # mm^3; assumed initial condition, Benzekry 2014 PDF p. 3
}
LOCAL_FITTABLE = {
    "a":     {"init": None},
    "b":     {"init": None},
    "gamma": {"init": None},
}


def _predict_core(t: np.ndarray, V0: float, a: float, b: float,
                  gamma: float) -> np.ndarray:
    exp_term = 1.0 - gamma  # exponent (1-c) in the Benzekry notation
    base_val = a / b
    inner = base_val + (V0 ** exp_term - base_val) * np.exp(-b * exp_term * t)
    inner = np.maximum(inner, 1e-30)
    return inner ** (1.0 / exp_term)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear LS fit of the von Bertalanffy model.

    Init: gamma_0 = 0.9 (near LLC median 0.947); a_0 from observed slope;
    b_0 = 0.01 (small relative to a_0 so growth dominates initially).
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    log_y = np.log(np.clip(y, 1e-6, None))
    if t.std() > 1e-9:
        a_0 = float(np.polyfit(t, log_y, 1)[0])
    else:
        a_0 = 0.5
    a_0 = max(1e-3, min(a_0, 10.0))
    b_0     = 0.1
    gamma_0 = 0.9

    def residual(p):
        return _predict_core(t, V0, p[0], p[1], p[2]) - y

    try:
        sol = least_squares(residual, [a_0, b_0, gamma_0],
                            bounds=([1e-6, 1e-6, 1e-4], [1e3, 1e3, 0.9999]),
                            method="trf", max_nfev=6000)
        return {"a": float(sol.x[0]), "b": float(sol.x[1]),
                "gamma": float(sol.x[2])}
    except Exception:                               # noqa: BLE001
        return {"a": a_0, "b": b_0, "gamma": gamma_0}


def predict(X: np.ndarray, a: float, b: float, gamma: float) -> np.ndarray:
    """Von Bertalanffy tumor volume (explicit closed-form solution).

    X: (n, 1) — column [time_day]. V0 read from OTHER_CONSTANTS.
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X[:, 0], dtype=float)
    return _predict_core(t, V0, a, b, gamma)
