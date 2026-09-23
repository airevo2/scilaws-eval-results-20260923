"""Carrillo et al. (2013) cubic piecewise wind power curve (Type I, frozen).

Carrillo, Obando Montaño, Cidrás, Díaz-Dorado (2013), *Renewable and
Sustainable Energy Reviews* 21:572-581.  The cubic model (Eq. 8 +
Eq. 3, PDF p. 3) is the canonical wind-turbine power-curve form:

    P(v) = 0                              if v < v_ci or v > v_co
    P(v) = (1/2) * rho * A * Cp_eq * v^3   if v_ci <= v < v_r
    P(v) = P_r                            if v_r <= v <= v_co

Symbol map → released CSV column:
    v       → wind_speed_mps  (m/s)
    P       → power_kW        (kW)

Physical constants frozen from Senvion MM82 datasheet (the turbine
model deployed at La Haute Borne wind farm — 4 × R80711/R80721/
R80736/R80790 are all Senvion MM82 2 MW machines):
    rho     = 1.225 kg/m^3   (standard sea-level air density)
    A       = 5281 m^2       (rotor swept area = π·(82/2)^2)
    v_ci    = 3.5 m/s        (cut-in wind speed)
    v_r     = 14.0 m/s       (rated wind speed)
    v_co    = 25.0 m/s       (cut-out wind speed)
    P_r     = 2050 kW        (rated electrical power)
    Cp_eq   = 0.4249         (equivalent power coefficient; pre-fit on
                              v2 train cubic-regime rows v ∈ [3.5, 14)
                              with P > 50 kW.  Carrillo Table 2 quotes
                              Cp_eq = 0.490 for a generic 2 MW reference
                              turbine; the lower v2 value reflects La
                              Haute Borne's actual operating efficiency.)

This is a TYPE I baseline: LOCAL_FITTABLE is empty, no fit() function,
predict() invoked once on the test set with LAW_CONSTANTS only.

LAW_CONSTANTS — frozen
----------------------
- V_CI   = 3.5     (Senvion MM82 datasheet)
- V_R    = 14.0    (Senvion MM82 datasheet)
- V_CO   = 25.0    (Senvion MM82 datasheet)
- P_R    = 2050.0  (Senvion MM82 datasheet)
- CP_EQ  = 0.4249  (pre-fit on v2 train cubic regime)

OTHER_CONSTANTS — universal physics
-----------------------------------
- RHO    = 1.225   kg/m^3   (standard air density)
- A      = 5281.0  m^2      (rotor swept area from 82 m diameter)
"""

import numpy as np

USED_INPUTS = ["wind_speed_mps"]
PAPER_REF = "summary_formula_carrillo_2013.md"
EQUATION_LOC = "Carrillo et al. 2013 Eq. 8 + Eq. 3, PDF p. 3 — piecewise cubic; Cp_eq pre-fit on v2 train cubic regime."

LAW_CONSTANTS = {
    "V_CI":  3.5,
    "V_R":   14.0,
    "V_CO":  25.0,
    "P_R":   2050.0,
    "CP_EQ": 0.4249,
}
OTHER_CONSTANTS = {
    "RHO": 1.225,
    "A":   5281.0,
}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray,
            V_CI: float = 3.5, V_R: float = 14.0, V_CO: float = 25.0,
            P_R: float = 2050.0, CP_EQ: float = 0.4249) -> np.ndarray:
    """Piecewise cubic Betz power curve."""
    v = np.asarray(X[:, 0], dtype=float)
    rho = OTHER_CONSTANTS["RHO"]
    A   = OTHER_CONSTANTS["A"]
    P_cubic = 0.5 * rho * A * CP_EQ * v**3 / 1000.0   # kW
    return np.where((v < V_CI) | (v > V_CO), 0.0,
           np.where(v < V_R, P_cubic, P_R))
