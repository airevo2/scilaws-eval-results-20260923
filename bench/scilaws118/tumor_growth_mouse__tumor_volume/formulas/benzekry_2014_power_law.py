"""Benzekry 2014 — Power-law tumor growth model.

Benzekry S, Lamont C, Beheshti A, Tracz A, Ebos JML, Hlatky L, Hahnfeldt P.
(2014). Classical Mathematical Models for Description and Prediction of
Experimental Tumor Growth. *PLoS Computational Biology*, 10(8): e1003800.
DOI:10.1371/journal.pcbi.1003800.

Derived from Eq. 6 (PDF p. 4) by taking b=0 (neglecting the loss term):

    dV/dt = a * V^gamma,   V(t=0) = 1 mm³

Closed-form solution (V0 = 1 mm³, 0 < gamma < 1):

    V(t) = [ V0^(1-gamma) + a*(1-gamma)*t ]^(1/(1-gamma))
         = [ 1 + a*(1-gamma)*t ]^(1/(1-gamma))

"Any power 0 < c < 1 gives a tumor growth with decreasing growth fraction
(and thus decreasing relative growth rate)." (PDF p. 4.)
"This model will be termed the power law model." (PDF p. 4.)

gamma=2/3 corresponds to surface-limited proliferation (radius grows
linearly); gamma=1 recovers pure exponential growth.

Best descriptive model for LLC (lung) data alongside Gompertz (PDF p. 7-9).

Median fitted parameters from Table 3 (PDF p. 9):
a ~ 0.921 mm^{3(1-gamma)} day^-1 (LLC); gamma ~ 0.788 (LLC),
0.58 (breast, Table 4).

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
- a     : synthesis/proliferation rate coefficient
          (mm^{3(1-gamma)} day^-1, > 0).
- gamma : allometric power exponent (dimensionless; physically (0, 1)).
  init = None on both: fit() derives data-driven starts.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["time_day"]
PAPER_REF = "summary_formula_dataset_benzekry_2014.md"
EQUATION_LOC = (
    "Benzekry 2014 PDF p. 4 — power law (b=0 limit of Eq. 6): "
    "dV/dt = a*V^gamma, V(0)=1; "
    "solution V(t) = [1 + a*(1-gamma)*t]^(1/(1-gamma))."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    "V0": 1.0,   # mm^3; assumed initial condition, Benzekry 2014 PDF p. 3
}
LOCAL_FITTABLE = {
    "a":     {"init": None},
    "gamma": {"init": None},
}


def _predict_core(t: np.ndarray, V0: float, a: float, gamma: float) -> np.ndarray:
    exp_term = 1.0 - gamma  # (1 - gamma) > 0 when gamma < 1
    inner = V0 ** exp_term + a * exp_term * t
    inner = np.maximum(inner, 1e-30)
    return inner ** (1.0 / exp_term)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear LS fit of the power-law model.

    Init: gamma_0 = 0.7 (midpoint of typical range 0.58-0.95);
    a_0 estimated from observed growth rate at median V.
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    gamma_0 = 0.7

    # Rough a init: use the analytic inversion at first non-trivial point
    order = np.argsort(t)
    t_s, y_s = t[order], y[order]
    if len(t_s) >= 2 and t_s[-1] > 1e-9:
        t_ref = t_s[-1]
        y_ref = float(y_s[-1])
        # a ~ (y_ref^(1-gamma) - V0^(1-gamma)) / ((1-gamma)*t_ref)
        exp_term = 1.0 - gamma_0
        a_0 = (max(y_ref, V0) ** exp_term - V0 ** exp_term) / (exp_term * t_ref)
    else:
        a_0 = 0.5
    a_0 = max(1e-4, min(a_0, 1e3))

    def residual(p):
        return _predict_core(t, V0, p[0], p[1]) - y

    try:
        sol = least_squares(residual, [a_0, gamma_0],
                            bounds=([1e-6, 1e-4], [1e4, 0.9999]),
                            method="trf", max_nfev=4000)
        return {"a": float(sol.x[0]), "gamma": float(sol.x[1])}
    except Exception:                               # noqa: BLE001
        return {"a": a_0, "gamma": gamma_0}


def predict(X: np.ndarray, a: float, gamma: float) -> np.ndarray:
    """Power-law tumor volume V(t) = [1 + a*(1-gamma)*t]^(1/(1-gamma)).

    X: (n, 1) — column [time_day]. V0 read from OTHER_CONSTANTS.
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X[:, 0], dtype=float)
    return _predict_core(t, V0, a, gamma)
