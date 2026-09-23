"""PySR cloud cover parameterization — ICON-A-MLe online-tuned coefficients.

Citation: Grundner, Beucler, Savre, Lauer, Schlund, Eyring (2025).
"Reduced cloud cover errors in a hybrid AI-climate model through equation
discovery and automatic tuning." arXiv:2505.04358v4.
DOI: 10.48550/ARXIV.2505.04358.

Formula (Eq. 3, PDF p. 18; identical structural form to Grundner 2024):

    f(RH, T, dz_RH, q_c, q_i) = I1(RH, T) + I2(dz_RH) + I3(q_c, q_i)

    I1 = a1 + a2*(RH_eff - RH_bar) + a3*(T - T_bar)
         + (a4/2)*(RH_eff - RH_bar)**2
         + (a5/2)*(T - T_bar)**2 * (RH_eff - RH_bar)

    I2 = a6**3 * (dz_RH + 3*a7/2) * dz_RH**2

    I3 = -1 / (q_c/a8 + q_i/a9 + epsilon)

    C = 0            if q_c + q_i == 0
        clip(f,0,1)  otherwise

    RH_eff = max{RH, c1 - c2*(T - T_bar)**2},  c1 = RH_bar - a2/a4,
             c2 = a5/(2*a4)   [inherited from Grundner 2024 Eq. 11]

LAW_CONSTANTS — ICON-A-MLe online-tuned values (PDF p. 18, Table S3 PDF p. 32):
  a1 = 0.118         — PDF p. 18 and Table S3 row 1
  a2 = 1.234         — PDF p. 18 and Table S3 row 2
  a3 = -0.0265 K^-1  — PDF p. 18 (text); Table S3 shows -0.027 (rounded)
  a4 = 5.65          — PDF p. 18 and Table S3 row 4
  a5 = 1.56e-3 K^-2  — PDF p. 18 and Table S3 row 5
  a6 = 591.68 m      — PDF p. 18 and Table S3 row 6
  a7 = 2.22e-3 m^-1  — PDF p. 18 "2.22 km^-1"; Table S3: 2.22e-3; unit m^-1
  a8 = 1.47e-6 kg/kg — PDF p. 18 "1.47 mg/kg" = 1.47e-6 kg/kg
  a9 = 3.44e-7 kg/kg — PDF p. 18 "0.344 mg/kg" = 3.44e-7 kg/kg
  epsilon = 0.615    — PDF p. 18 and Table S3 row 10

All values confirmed at: PDF p. 18 line "{a1,...,a9,eps} = {0.118, 1.234,
-0.0265 K^-1, 5.65, 1.56e-3 K^-2, 591.68 m, 2.22 km^-1, 1.47 mg/kg,
0.344 mg/kg, 0.615}" and Table S3 at PDF p. 32.

OTHER_CONSTANTS — paper-fixed centering values (PDF p. 18):
  RH_bar = 0.6025   — "fixed" per PDF p. 18 (inherited from Grundner 2024)
  T_bar  = 257.06 K — "fixed" per PDF p. 18 (inherited from Grundner 2024)

Type designation: Type I — no per-cluster refit. LOCAL_FITTABLE = {}.

Column mapping: same as Grundner 2024 (structural form is identical).

Calibration domain: ICON-A-MLe AMIP online tuning against CERES/GPCP-SG/
ISCCP satellite observations over 1979-1999. Coefficients differ from
Grundner 2024 (offline DYAMOND-fit) due to a different calibration regime.
Both formula files represent alternative published parameterizations of the
same structural form.

Note on a3: the text at PDF p. 18 gives -0.0265 K^-1; Table S3 at PDF p. 32
shows -0.027 (rounded to 3 decimal places). LAW_CONSTANTS uses the more
precise value from PDF p. 18 text.
"""

import numpy as np

USED_INPUTS = ["q_c", "q_i", "RH", "T", "dz_RH"]
PAPER_REF = "summary_formula_grundner_2025.md"
EQUATION_LOC = "Eq. 3, PDF p. 18; Table S3, PDF p. 32 (ICON-A-MLe coefficients)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a1":      0.118,       # PDF p. 18, Table S3 PDF p. 32
    "a2":      1.234,       # PDF p. 18, Table S3 PDF p. 32
    "a3":     -0.0265,      # PDF p. 18 [K^-1]; Table S3 rounds to -0.027
    "a4":      5.65,        # PDF p. 18, Table S3 PDF p. 32
    "a5":      1.56e-3,     # PDF p. 18 [K^-2], Table S3 PDF p. 32
    "a6":      591.68,      # PDF p. 18 [m], Table S3 PDF p. 32
    "a7":      2.22e-3,     # PDF p. 18 "2.22 km^-1" → 2.22e-3 m^-1
    "a8":      1.47e-6,     # PDF p. 18 "1.47 mg/kg" → kg/kg
    "a9":      3.44e-7,     # PDF p. 18 "0.344 mg/kg" → kg/kg
    "epsilon": 0.615,       # PDF p. 18, Table S3 PDF p. 32
}

# === OTHER_CONSTANTS — paper-fixed centering (DYAMOND training-set means) ===
OTHER_CONSTANTS = {
    "RH_bar": 0.6025,   # fixed per PDF p. 18
    "T_bar":  257.06,   # fixed per PDF p. 18 [K]
}

LOCAL_FITTABLE = {}    # Type I — no per-cluster parameters


def predict(X: np.ndarray, a1: float, a2: float, a3: float, a4: float,
            a5: float, a6: float, a7: float, a8: float,
            a9: float, epsilon: float) -> np.ndarray:
    """Cloud cover fraction per Grundner 2025 Eq. 3 (ICON-A-MLe coefficients).

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

    # RH-clamp (same monotonicity constraint as Grundner 2024 Eq. 11)
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
