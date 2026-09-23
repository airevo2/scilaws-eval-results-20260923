"""Benzekry 2014 — Dynamic carrying capacity (CC) tumor growth model.

Benzekry S, Lamont C, Beheshti A, Tracz A, Ebos JML, Hlatky L, Hahnfeldt P.
(2014). Classical Mathematical Models for Description and Prediction of
Experimental Tumor Growth. *PLoS Computational Biology*, 10(8): e1003800.
DOI:10.1371/journal.pcbi.1003800.

Eq. 5 (PDF p. 3-4):

    dV/dt = a * V * log(K / V)
    dK/dt = b * V^(2/3)
    V(t=0) = 1 mm³,   K(t=0) = K0

The dynamic CC model couples tumor volume V (mm³) to a time-varying carrying
capacity K (mm³) representing tumor vasculature. Stimulation of K is
proportional to the tumor surface area (V^(2/3) scaling). Three free
per-animal parameters: a, b, K0.

The exponent 2/3 in dK/dt = b * V^(2/3) is a structural constant reflecting
the surface-area scaling assumption (PDF p. 3-4: "stimulation of the carrying
capacity is proportional to the tumor surface"). It is never refit.

Note: This model was originally developed for anti-angiogenic therapy
modeling, not strictly for growth description (PDF p. 4). It is included here
as the paper explicitly evaluates it (Eq. 5) alongside the other models.
Parameters are poorly identifiable from short series (3+ free params; see
Benzekry 2014 Table 3 large coefficient of variation, PDF p. 9).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
(empty.)

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
- surf_exp: 2/3 — surface-area allometric exponent in dK/dt = b*V^(2/3);
  structural assumption of the dynamic-CC model, never refit
  (PDF p. 3: "stimulation of the carrying capacity is proportional to
  the tumor surface").
- V0: 1.0 mm³ — assumed (given) initial condition, NOT a fitted coefficient.
  Benzekry 2014 PDF p. 3 / L255 calls 1 mm³ "a reasonable approximation for
  V(t=0)" (~10^6 injected cells) — an invariant-but-GIVEN injection volume, so
  it is filed under OTHER (structural/given) rather than LAW (fitted-law).

LOCAL_FITTABLE — per-cluster, fitted by fit()
----------------------------------------------
- a  : Gompertz-type growth rate coefficient (day^-1, > 0).
- b  : CC stimulation rate (mm^-2 day^-1, > 0).
- K0 : initial carrying capacity (mm³, > 0).
  init = None on all: fit() derives data-driven starts.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

USED_INPUTS = ["time_day"]
PAPER_REF = "summary_formula_dataset_benzekry_2014.md"
EQUATION_LOC = (
    "Benzekry 2014 Eq. 5, PDF p. 3-4 — dynamic CC: "
    "dV/dt = a*V*log(K/V), dK/dt = b*V^(2/3), V(0)=1, K(0)=K0."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    "surf_exp": 2.0 / 3.0,   # surface-area allometric exponent; PDF p. 3-4
    "V0": 1.0,               # mm^3; assumed initial condition, PDF p. 3
}
LOCAL_FITTABLE = {
    "a":  {"init": None},
    "b":  {"init": None},
    "K0": {"init": None},
}


def _solve(t_eval: np.ndarray, V0: float, a: float, b: float, K0: float) -> np.ndarray:
    """Integrate the dynamic-CC ODE system and return V at t_eval."""
    surf = OTHER_CONSTANTS["surf_exp"]

    t_span = (0.0, float(np.max(t_eval)))
    y0 = [V0, K0]

    def rhs(t, y):
        V, K = y
        V = max(V, 1e-12)
        K = max(K, V * 1e-12)   # K >= V to keep log non-negative
        dVdt = a * V * np.log(K / V)
        dKdt = b * V ** surf
        return [dVdt, dKdt]

    try:
        sol = solve_ivp(rhs, t_span, y0, t_eval=np.sort(t_eval),
                        method="RK45", rtol=1e-6, atol=1e-9,
                        dense_output=False)
        if sol.success:
            # Re-order to match original t_eval order
            sort_idx = np.argsort(t_eval)
            inv_idx  = np.argsort(sort_idx)
            return sol.y[0][inv_idx]
        else:
            return np.full_like(t_eval, V0, dtype=float)
    except Exception:                               # noqa: BLE001
        return np.full_like(t_eval, V0, dtype=float)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear LS fit of the dynamic-CC model.

    Init: a_0 from Gompertz-like slope; K0_0 = 2*max(y); b_0 small.
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    K0_0 = float(np.max(y)) * 2.0
    K0_0 = max(K0_0, 2.0)

    log_y = np.log(np.clip(y, 1e-6, None))
    if t.std() > 1e-9:
        a_0 = float(np.polyfit(t, log_y, 1)[0])
    else:
        a_0 = 0.5
    a_0 = max(1e-3, min(a_0, 5.0))
    b_0 = 1.0   # mm^-2 day^-1; rough order of magnitude from Table 3 (LLC)

    def residual(p):
        return _solve(t, V0, p[0], p[1], p[2]) - y

    try:
        sol = least_squares(residual, [a_0, b_0, K0_0],
                            bounds=([1e-6, 1e-6, 1e-3], [20.0, 1e4, 1e6]),
                            method="trf", max_nfev=6000)
        return {"a": float(sol.x[0]), "b": float(sol.x[1]),
                "K0": float(sol.x[2])}
    except Exception:                               # noqa: BLE001
        return {"a": a_0, "b": b_0, "K0": K0_0}


def predict(X: np.ndarray, a: float, b: float, K0: float) -> np.ndarray:
    """Dynamic-CC tumor volume via ODE integration.

    X: (n, 1) — column [time_day]. V0 read from OTHER_CONSTANTS.
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X[:, 0], dtype=float)
    return _solve(t, V0, a, b, K0)
