"""Gordon-Ng universal chiller model -- one_over_COP_minus_1 = 1/COP - 1.

Sreedharan, P. (2001). "Comparison of chiller models for use in model-based
fault detection." LBNL-48856 / UC Berkeley thesis, Eq. 5.5 (thesis PDF p. 32).
Originating model: J. M. Gordon & K. C. Ng, "Thermodynamic modeling of
reciprocating chillers" / "Cool Thermodynamics" (Cambridge Univ. Press, 2000).

The Gordon-Ng universal thermodynamic model for a vapor-compression chiller
writes the (dimensionless) thermodynamic loss balance as Eq. 5.5:

    (T_ei/T_ci)*(1 + 1/COP) - 1
        = Delta_S_T * (T_ei/Q_e)
        + Q_leak_eqv * (T_ci - T_ei)/(T_ci*Q_e)
        + R * (Q_e/T_ci) * (1 + 1/COP)

with COP = Q_evap/W_comp the (measured) coefficient of performance, T_ei/T_ci
the evaporator/condenser inlet water temperatures [K], Q_e the cooling load [kW].
The three coefficients are per-chiller thermodynamic properties.

This task's TARGET is the clean performance quantity  y = 1/COP - 1.  Writing
u = 1 + 1/COP = y + 2, the model above is implicit in u; solving for u gives an
explicit predictor that needs only (T_ei, T_ci, Q_e) and the three parameters:

    u   = (1 + Delta_S_T*(T_ei/Q_e) + Q_leak_eqv*(T_ci-T_ei)/(T_ci*Q_e))
          / ((T_ei - R*Q_e)/T_ci)
    y   = u - 2          (= 1/COP - 1)

LAW_CONSTANTS -- empty
----------------------
Form-discovery Type II: the Gordon-Ng FORM is the scientific claim. There is no
universal numerical constant to recover -- the structural 1's are algebra (the
energy/entropy balance), not tunable constants. The only cross-cluster invariant
is the functional form itself.

OTHER_CONSTANTS -- none.

LOCAL_FITTABLE -- per-cluster (per chiller test run) thermodynamic parameters
------------------------------------------------------------------------------
Delta_S_T  : total internal entropy generation rate [kW/K].
Q_leak_eqv : equivalent heat-leak rate [kW].
R          : combined condenser+evaporator effective thermal resistance [K/kW].
Each test run (normal or fault condition) has its own (Delta_S_T, Q_leak_eqv, R);
fit() recovers them per cluster by the linear regression of Eq. 5.5.
"""

import numpy as np

USED_INPUTS = ["T_ei", "T_ci", "Q_e"]
PAPER_REF = "summary_formula_dataset_sreedharan_2001.md"
EQUATION_LOC = (
    "Sreedharan (2001) Eq. 5.5, thesis PDF p. 32: "
    "(T_ei/T_ci)*(1+1/COP)-1 = Delta_S_T*(T_ei/Q_e) "
    "+ Q_leak_eqv*(T_ci-T_ei)/(T_ci*Q_e) + R*(Q_e/T_ci)*(1+1/COP)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "Delta_S_T":  {"init": None},
    "Q_leak_eqv": {"init": None},
    "R":          {"init": None},
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Recover (Delta_S_T, Q_leak_eqv, R) for one cluster by Gordon-Ng OLS.

    Linearise Eq. 5.5 in the three parameters using u = 1 + 1/COP = y + 2:
        lhs = (T_ei/T_ci)*u - 1
            = Delta_S_T*x1 + Q_leak_eqv*x2 + R*x3,
        x1 = T_ei/Q_e,  x2 = (T_ci-T_ei)/(T_ci*Q_e),  x3 = (Q_e/T_ci)*u.
    Bounded NLS fallback if OLS yields non-physical parameters.
    """
    T_ei = X_fit[:, 0].astype(float)
    T_ci = X_fit[:, 1].astype(float)
    Q_e  = X_fit[:, 2].astype(float)
    u    = y_fit.astype(float) + 2.0          # 1 + 1/COP

    lhs = (T_ei / T_ci) * u - 1.0
    x1 = T_ei / Q_e
    x2 = (T_ci - T_ei) / (T_ci * Q_e)
    x3 = (Q_e / T_ci) * u
    A  = np.column_stack([x1, x2, x3])

    try:
        p, _, _, _ = np.linalg.lstsq(A, lhs, rcond=None)
        dS, Ql, R = p
        if np.all(np.isfinite(p)) and dS > 0 and Ql >= 0 and R > 0:
            return {"Delta_S_T": float(dS), "Q_leak_eqv": float(Ql), "R": float(R)}
    except Exception:
        pass

    try:
        from scipy.optimize import least_squares
        Q_med = max(float(np.median(Q_e)), 1.0)
        p0 = [0.08, 0.05 * Q_med, 0.08]
        sol = least_squares(lambda p: A @ p - lhs, p0,
                            bounds=([1e-6, 0.0, 1e-6], [1.0, 500.0, 1.0]),
                            method="trf")
        dS, Ql, R = sol.x
        return {"Delta_S_T": float(dS), "Q_leak_eqv": float(Ql), "R": float(R)}
    except Exception:
        return {"Delta_S_T": 0.08, "Q_leak_eqv": 5.0, "R": 0.08}


def predict(X: np.ndarray, Delta_S_T: float, Q_leak_eqv: float, R: float) -> np.ndarray:
    """Predict y = 1/COP - 1 from (T_ei, T_ci, Q_e), no observed COP needed.

    Solves the implicit Gordon-Ng relation for u = 1 + 1/COP, then returns u - 2.
    """
    T_ei = X[:, 0].astype(float)
    T_ci = X[:, 1].astype(float)
    Q_e  = X[:, 2].astype(float)

    numerator   = (1.0
                   + Delta_S_T  * (T_ei / Q_e)
                   + Q_leak_eqv * (T_ci - T_ei) / (T_ci * Q_e))
    denominator = (T_ei - R * Q_e) / T_ci
    denom_safe  = np.where(np.abs(denominator) < 1e-12,
                           np.sign(denominator + 1e-15) * 1e-12, denominator)
    u = numerator / denom_safe
    return u - 2.0
