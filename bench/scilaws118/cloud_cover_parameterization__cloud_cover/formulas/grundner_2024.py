"""PySR-discovered cloud cover parameterization — DYAMOND offline fit.

Citation: Grundner, Beucler, Gentine, Eyring (2024). "Data-Driven Equation
Discovery of a Cloud Cover Parameterization." J. Advances in Modeling Earth
Systems (JAMES), 16(3). DOI: 10.1029/2023MS003763. arXiv:2304.08063.

Formula (Eq. 10 + Eq. 6 + Eq. 11, PDF pp. 20-21):

    f(RH, T, dz_RH, q_c, q_i) = I1(RH, T) + I2(dz_RH) + I3(q_c, q_i)

    I1 = a1 + a2*(RH_eff - RH_bar) + a3*(T - T_bar)
         + (a4/2)*(RH_eff - RH_bar)**2
         + (a5/2)*(T - T_bar)**2 * (RH_eff - RH_bar)

    I2 = a6**3 * (dz_RH + 3*a7/2) * dz_RH**2

    I3 = -1 / (q_c/a8 + q_i/a9 + epsilon)

    C = 0            if q_c + q_i == 0
        clip(f,0,1)  otherwise

    RH_eff = max{RH, c1 - c2*(T - T_bar)**2}   [Eq. 11, PDF p. 21]
      c1 = RH_bar - a2/a4,  c2 = a5/(2*a4)

LAW_CONSTANTS — paper-published DYAMOND-fit values (PDF p. 21):
  a1 = 0.4435        — intercept (I1 contribution at mean RH, T)
  a2 = 1.1593        — linear RH gradient at (RH_bar, T_bar)
  a3 = -0.0145 K^-1  — linear T gradient
  a4 = 4.06          — quadratic RH curvature
  a5 = 1.3176e-3 K^-2 — cross-term coefficient (T^2 * dRH)
  a6 = 584.8036 m    — vertical-gradient scale (unit m; I2 prefactor is a6^3)
  a7 = 2.0e-3 m^-1   — paper quotes "2 km^-1"; converted: 2 km^-1 = 2e-3 m^-1
  a8 = 1.1573e-6 kg/kg — paper quotes "1.1573 mg/kg"; 1 mg/kg = 1e-6 kg/kg
  a9 = 3.073e-7 kg/kg  — paper quotes "0.3073 mg/kg" = 3.073e-7 kg/kg
  epsilon = 1.06     — positive offset preventing I3 blow-up

All 10 values at PDF p. 21, line "the best values for the coefficients to
be {a1,...,a9,eps} = {0.4435, 1.1593, -0.0145 K^-1, 4.06, 1.3176e-3 K^-2,
584.8036 m, 2 km^-1, 1.1573 mg/kg, 0.3073 mg/kg, 1.06}".

OTHER_CONSTANTS — paper-fixed centering values (PDF p. 21):
  RH_bar = 0.6025   — average RH of the DYAMOND training set
  T_bar  = 257.06 K — average T of the DYAMOND training set
  (Also explicitly designated "fixed" in Grundner 2025 PDF p. 18.)

Type designation: Type I — each (grid cell, level, time) row is
independent; no per-cluster refit. LOCAL_FITTABLE = {}.

Column mapping: paper RH -> CSV col RH; T -> T; partial_z(RH) -> dz_RH;
q_c -> q_c; q_i -> q_i. (The 100× factor in the paper maps percent-cloud
to fraction; the released CSV target is in [0,1] so this factor is absent.)

Calibration domain: DYAMOND global SRM (~1.8e8 rows). Test domain here is
NARVAL-II tropical-Atlantic snapshot (40k rows). Domain mismatch is
expected and documented — R² may be negative on NARVAL (over-prediction
bias ~5%) per data_raw/README.md. This is an accepted OOD probe per
data_spec §9.18.
"""

import numpy as np

USED_INPUTS = ["q_c", "q_i", "RH", "T", "dz_RH"]
PAPER_REF = "summary_formula_grundner_2024.md"
EQUATION_LOC = "Eq. 10 + Eq. 6 + Eq. 11, PDF pp. 20-21"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a1":      0.4435,      # PDF p. 21
    "a2":      1.1593,      # PDF p. 21
    "a3":     -0.0145,      # PDF p. 21  [K^-1]
    "a4":      4.06,        # PDF p. 21
    "a5":      1.3176e-3,   # PDF p. 21  [K^-2]
    "a6":      584.8036,    # PDF p. 21  [m]
    "a7":      2.0e-3,      # PDF p. 21  "2 km^-1" → 2e-3 m^-1
    "a8":      1.1573e-6,   # PDF p. 21  "1.1573 mg/kg" → kg/kg
    "a9":      3.073e-7,    # PDF p. 21  "0.3073 mg/kg" → kg/kg
    "epsilon": 1.06,        # PDF p. 21
}

# === OTHER_CONSTANTS — paper-fixed centering (DYAMOND training-set means) ===
OTHER_CONSTANTS = {
    "RH_bar": 0.6025,   # training-set mean RH; PDF p. 21
    "T_bar":  257.06,   # training-set mean T [K]; PDF p. 21
}

LOCAL_FITTABLE = {}    # Type I — no per-cluster parameters


def predict(X: np.ndarray, a1: float, a2: float, a3: float, a4: float,
            a5: float, a6: float, a7: float, a8: float,
            a9: float, epsilon: float) -> np.ndarray:
    """Cloud cover fraction per Grundner 2024 Eq. 10+6+11.

    X shape: (n, 5) — columns in USED_INPUTS order: q_c, q_i, RH, T, dz_RH.
    Returns predicted cloud area fraction in [0, 1].
    """
    RH_bar = OTHER_CONSTANTS["RH_bar"]
    T_bar  = OTHER_CONSTANTS["T_bar"]

    q_c   = np.asarray(X[:, 0], dtype=float)
    q_i   = np.asarray(X[:, 1], dtype=float)
    RH    = np.asarray(X[:, 2], dtype=float)
    T     = np.asarray(X[:, 3], dtype=float)
    dz_RH = np.asarray(X[:, 4], dtype=float)

    # PC3 RH-clamp (Eq. 11, PDF p. 21): replace RH by max(RH, c1 - c2*(T-T_bar)^2)
    c1 = RH_bar - a2 / a4
    c2 = a5 / (2.0 * a4)
    RH_eff = np.maximum(RH, c1 - c2 * (T - T_bar) ** 2)

    dRH = RH_eff - RH_bar
    dT  = T - T_bar

    I1 = (a1
          + a2 * dRH
          + a3 * dT
          + 0.5 * a4 * dRH ** 2
          + 0.5 * a5 * dT ** 2 * dRH)
    I2 = (a6 ** 3) * (dz_RH + 1.5 * a7) * (dz_RH ** 2)
    I3 = -1.0 / (q_c / a8 + q_i / a9 + epsilon)

    f = I1 + I2 + I3
    C = np.clip(f, 0.0, 1.0)
    C = np.where(q_c + q_i == 0.0, 0.0, C)
    return C
