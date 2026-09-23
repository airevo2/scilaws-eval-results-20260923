"""Stewart (1988) Jarvis-type surface conductance + Penman-Monteith LE.

Stewart (1988), Agric. For. Meteorol. 43:19-35. The canopy surface
conductance g_s is the Jarvis product (Stewart PDF p. 26):

    g_s = LAI * K1 * g(SWdown) * g(VPD) * g(Tair)

with sub-functions
    g(SWdown) = K7 * SWdown / (SWdown + K2),     K7 = (1000 + K2)/1000
    g(VPD)    = max(1 - K3 * dq, 1 - K3 * K4),   dq = 0.622*VPD/95 (g/kg)
    g(Tair)   = (T - T_L) * (T_H - T)^K8 /
                ((K5 - T_L) * (T_H - K5)^K8),    K8 = (T_H - K5)/(K5 - T_L)

T_L = 0 degC and T_H = 40 degC are fixed structural constants (Stewart
PDF p. 26). The soil-moisture sub-function g(delta_theta) (Stewart Eqs.
22-24) is dropped because the released csv carries only PFT-typical
theta_FC / theta_WP, not in-situ soil moisture (its time-mean per-site
effect is absorbed into K1, Stewart's most sensitive parameter; PDF
Table 4, p. 31).

LE closes via Penman-Monteith (Li 2024 KG-DSR-5, PDF p. 3 / Stewart Eq.
5, PDF p. 21):

    LE = (Delta * (Rnet - Qg) + rho_a * Cp * VPD * Ga)
         / (Delta + gamma * (1 + Ga / g_s)).

Initial K1..K5 follow Stewart's Thetford-Forest fit (Table 2 row
"g_s = f(S_t, delta_q, T, delta_theta) non-linear functions, All days";
K1 = 23.53 mm/s, K2 = 104.4 W/m^2, K3 = 0.0812, K4 = 9.44, K5 = 18.35;
PDF p. 27). These are per-site fits in the paper, so they are
LOCAL_FITTABLE; the Thetford values seed each per-cluster nonlinear LS.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
- RHO_A = 1.225  kg/m^3
- CP    = 1013.0 J/(kg K)
- GAMMA = 0.0664 kPa/K
- T_L   = 0.0    degC   (Jarvis low temperature cutoff)
- T_H   = 40.0   degC   (Jarvis high temperature cutoff)

LOCAL_FITTABLE — per-cluster, computed by fit() via nonlinear LS
-----------------------------------------------------------------
- K1 : Jarvis-conductance scale (mm/s) — most sensitive parameter.
- K2 : SWdown half-saturation (W/m^2).
- K3 : VPD-stress slope (1/(g/kg)).
- K4 : VPD-stress saturation (g/kg).
- K5 : optimum temperature (degC).
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["SWdown", "VPD", "Tair", "Rnet", "Qg", "LAI", "Ga", "delta"]
PAPER_REF = "summary_formula_stewart_1988.md"
EQUATION_LOC = (
    "Stewart (1988) Eqs. 12, 17-18, 19a-19b, 20-21 (PDF p. 26); Table 2 "
    "Thetford 'All days' nonlinear fit (PDF p. 27); PM closure Eq. 5 "
    "(PDF p. 21)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    "RHO_A": 1.225,
    "CP":    1013.0,
    "GAMMA": 0.0664,
    "T_L":   0.0,
    "T_H":   40.0,
}
LOCAL_FITTABLE = {
    "K1": {"init": None},
    "K2": {"init": None},
    "K3": {"init": None},
    "K4": {"init": None},
    "K5": {"init": None},
}


def _gs(SWdown, VPD, Tair_K, LAI, K1, K2, K3, K4, K5):
    T_L = OTHER_CONSTANTS["T_L"]
    T_H = OTHER_CONSTANTS["T_H"]
    dq = 0.622 * VPD / 95.0
    T = Tair_K - 273.15
    K7 = (1000.0 + K2) / 1000.0
    K8 = (T_H - K5) / (K5 - T_L)
    g_S  = K7 * SWdown / (SWdown + K2)
    g_dq = np.where(dq < K4, 1.0 - K3 * dq, 1.0 - K3 * K4)
    Tc = np.clip(T, T_L + 1.0e-6, T_H - 1.0e-6)
    num_T = (Tc - T_L) * np.power(T_H - Tc, K8)
    den_T = (K5 - T_L) * np.power(T_H - K5, K8)
    g_T = num_T / den_T
    g_T = np.where((T <= T_L) | (T >= T_H), 0.0, g_T)
    gs_mm_s = LAI * K1 * g_S * g_dq * g_T
    return np.maximum(gs_mm_s * 1.0e-3, 1.0e-6)


def _le(X, K1, K2, K3, K4, K5):
    SWdown, VPD, Tair, Rnet, Qg, LAI, Ga, delta = (
        X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7]
    )
    gs = _gs(SWdown, VPD, Tair, LAI, K1, K2, K3, K4, K5)
    num = delta * (Rnet - Qg) + OTHER_CONSTANTS["RHO_A"] * OTHER_CONSTANTS["CP"] * VPD * Ga
    den = delta + OTHER_CONSTANTS["GAMMA"] * (1.0 + Ga / gs)
    return num / den


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    X = np.asarray(X_fit, dtype=float)
    y = np.asarray(y_fit, dtype=float)
    p0 = [23.53, 104.4, 0.0812, 9.44, 18.35]         # Stewart Thetford fit
    bounds = ([0.1,   10.0,   1e-4, 1.0,  5.0],
              [200.0, 2000.0, 5.0,  50.0, 35.0])

    def residual(p):
        return _le(X, *p) - y

    try:
        sol = least_squares(residual, p0, bounds=bounds,
                            method="trf", max_nfev=3000)
        return {n: float(v) for n, v in zip(("K1", "K2", "K3", "K4", "K5"), sol.x)}
    except Exception:                              # noqa: BLE001
        return dict(zip(("K1", "K2", "K3", "K4", "K5"), p0))


def predict(X: np.ndarray, K1: float, K2: float, K3: float,
            K4: float, K5: float) -> np.ndarray:
    """LE = PM(X) with Stewart Jarvis-type Gs(SWdown, VPD, Tair, LAI; K1..K5).

    X: (n, 8) — columns SWdown, VPD, Tair, Rnet, Qg, LAI, Ga, delta.
    """
    return _le(np.asarray(X, dtype=float), K1, K2, K3, K4, K5)
