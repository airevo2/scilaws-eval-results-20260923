"""Li (2024) KG-DSR-5 Penman-Monteith with E-1 surface-conductance form.

Li (2024), npj Climate and Atmospheric Science 7:321, DOI
10.1038/s41612-024-00861-5. The KG-DSR-5 Pareto model recovers the
classical Penman-Monteith equation (Fig. 1, PDF p. 3):

    LE = (Delta * (Rnet - Qg) + rho_a * Cp * VPD * Ga)
         / (Delta + gamma * (1 + Ga / G_s))

paired with the E-1 expert-guided surface-conductance parameterisation
(Fig. 2 panel b/c, PDF p. 4):

    G_s ∝ RH * f(SWdown, a_1) * exp(-theta_FC),
    f(R_s, a_1) = R_s * (1000 + a_1) / (1000 * (R_s + a_1)).

The proportionality (`scale`, m/s) and the radiation half-saturation
constant `a_1` are per-site fits (PDF p. 4 caption, summary §3.2); the
PM closure constants rho_a, Cp, gamma are universal physics and
inlined (Li 2024 PDF p. 13; FAO-56 standard).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None for the SR target — the universal PM closure constants are
structural (rho_a, Cp, gamma) and live in OTHER_CONSTANTS.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
- RHO_A = 1.225  kg/m^3       (air density, FAO-56 standard)
- CP    = 1013.0 J/(kg K)     (specific heat of moist air, FAO-56)
- GAMMA = 0.0664 kPa/K        (psychrometric constant at ~95 kPa)

LOCAL_FITTABLE — per-cluster, computed by fit() via nonlinear LS
-----------------------------------------------------------------
- scale : proportionality constant in m/s for the E-1 G_s expression.
- a1    : radiation half-saturation constant in W/m^2.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["SWdown", "VPD", "RH", "Rnet", "Qg", "Ga", "delta", "theta_FC"]
PAPER_REF = "summary_formula_dataset_li_2024.md"
EQUATION_LOC = (
    "Li (2024) KG-DSR-5 PM equation (Fig. 1, PDF p. 3); E-1 surface-"
    "conductance form (Fig. 2 b/c, PDF p. 4) — JA radiation kernel "
    "f(R_s, a_1)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    "RHO_A": 1.225,
    "CP":    1013.0,
    "GAMMA": 0.0664,
}
LOCAL_FITTABLE = {
    "scale": {"init": None},
    "a1":    {"init": None},
}


def _gs(SWdown, RH, theta_FC, scale, a1):
    a1_safe = np.maximum(a1, 1.0)
    f_Rs = SWdown * (1000.0 + a1_safe) / (1000.0 * (SWdown + a1_safe))
    return np.maximum(scale * RH * f_Rs * np.exp(-theta_FC), 1.0e-6)


def _le(X, scale, a1):
    SWdown, VPD, RH, Rnet, Qg, Ga, delta, theta_FC = (
        X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7]
    )
    gs = _gs(SWdown, RH, theta_FC, scale, a1)
    num = delta * (Rnet - Qg) + OTHER_CONSTANTS["RHO_A"] * OTHER_CONSTANTS["CP"] * VPD * Ga
    den = delta + OTHER_CONSTANTS["GAMMA"] * (1.0 + Ga / gs)
    return num / den


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    X = np.asarray(X_fit, dtype=float)
    y = np.asarray(y_fit, dtype=float)
    p0 = [0.01, 300.0]            # scale ~ 10 mm/s; a1 ~ mid-range W/m^2
    bounds = ([1e-6, 1.0], [10.0, 5000.0])

    def residual(p):
        return _le(X, p[0], p[1]) - y

    try:
        sol = least_squares(residual, p0, bounds=bounds,
                            method="trf", max_nfev=2000)
        return {"scale": float(sol.x[0]), "a1": float(sol.x[1])}
    except Exception:                              # noqa: BLE001
        return {"scale": p0[0], "a1": p0[1]}


def predict(X: np.ndarray, scale: float, a1: float) -> np.ndarray:
    """LE = PM(X) with E-1 surface conductance Gs(SWdown, RH, theta_FC; scale, a1).

    X: (n, 8) — columns SWdown, VPD, RH, Rnet, Qg, Ga, delta, theta_FC.
    """
    return _le(np.asarray(X, dtype=float), scale, a1)
