"""Chen & Kipping (2016) four-segment continuous piecewise power law for
the mass-radius relation of sub-stellar and stellar objects ("Forecaster").

Citation: Chen & Kipping, ApJ 834(1):17, Dec 2016, DOI 10.3847/1538-4357/834/1/17.
PDF pages verified: 4 (Eqs. 1-2), 6 (Eqs. 5-6, 11), 7-9 (Table 1).

Formula (Eq. 1-2, PDF p. 4; Eq. 11 piecewise, PDF p. 6):
    log10(R/R_earth) = C^(j) + S^(j) * log10(M/M_earth)    in segment j

with continuity constraint (Eq. 6, PDF p. 6):
    C^(j+1) = C^(j) + (S^(j) - S^(j+1)) * T^(j),   j = 1, 2, 3.

Only C^(1), the four slopes S^(1..4), and the three log10-mass transitions
T^(1..3) are free; C^(2..4) follow by continuity.

LAW_CONSTANTS — paper-published posterior medians (Table 1, PDF pp. 7-9):
    C1 = log10(1.008)  : 10^C^(1) = 1.008 R_earth  (PDF p. 7)
    S1 = 0.2790        : Terran power-law index      (PDF p. 7)
    S2 = 0.589         : Neptunian power-law index   (PDF p. 7)
    S3 = -0.044        : Jovian power-law index      (PDF p. 7-8)
    S4 = 0.881         : Stellar power-law index     (PDF p. 7)
    T1 = log10(2.04)   : 10^T^(1) = 2.04 M_earth    (PDF p. 8)
    T2 = log10(0.414 * M_JUP_PER_EARTH)  : 10^T^(2) = 0.414 M_Jup  (PDF p. 9)
    T3 = log10(0.0800 * M_SUN_PER_EARTH) : 10^T^(3) = 0.0800 M_sun (PDF p. 9)

Note: T2 and T3 are given in the paper in M_Jupiter and M_sun units respectively;
conversion to log10(M_earth) is done via the IAU 2015 nominal mass ratios in
OTHER_CONSTANTS. The values 0.414 M_Jup and 0.0800 M_sun are directly from Table 1.

OTHER_CONSTANTS — universal conversion factors (not paper discoveries):
    M_JUP_PER_EARTH = 317.8284  : IAU 2015 nominal M_Jupiter / M_earth ratio
    M_SUN_PER_EARTH = 332946.0  : IAU 2015 nominal M_sun / M_earth ratio

Type designation: Type I — one universal formula for all objects; no per-planet
or per-cluster parameters. LOCAL_FITTABLE is empty ({}).

Column mapping (paper notation -> released CSV):
    M (M_earth) -> column "M" (input)
    R (R_earth) -> column "R" (target, column 0)

Caveats:
- This module returns the deterministic median only (Eqs. 1-2). The
  probabilistic intrinsic scatter sigma_R^(j) (Eq. 3, Table 1) is not
  implemented; the benchmark target is a deterministic R from M.
- Paper stated validity: ~3e-4 M_earth to 3e5 M_earth (PDF p. 13); the
  released test CSV extends to ~9e6 M_earth (stellar regime), where the
  formula is used outside its original paper calibration range.
"""

import numpy as np

USED_INPUTS = ["M"]
PAPER_REF = "summary_formula+dataset_chen_2016.md"
EQUATION_LOC = "Eq. 1-2, PDF p. 4; Eq. 6 (continuity), PDF p. 6; Eq. 11 (piecewise), PDF p. 6; Table 1, PDF pp. 7-9"

# IAU 2015 nominal mass ratios used to convert T2 (M_Jupiter) and T3 (M_sun)
# into log10(M/M_earth) space.
# === OTHER_CONSTANTS — IAU mass-ratio conversion factors ===
OTHER_CONSTANTS = {
    "M_JUP_PER_EARTH": 317.8284,   # IAU 2015 nominal Jupiter/Earth mass ratio
    "M_SUN_PER_EARTH": 332946.0,   # IAU 2015 nominal Sun/Earth mass ratio
}

_M_JUP = OTHER_CONSTANTS["M_JUP_PER_EARTH"]
_M_SUN = OTHER_CONSTANTS["M_SUN_PER_EARTH"]

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "C1": np.log10(1.008),              # Table 1 PDF p. 7: 10^C^(1) = 1.008 R_earth
    "S1": 0.2790,                       # Table 1 PDF p. 7: Terran slope
    "S2": 0.589,                        # Table 1 PDF p. 7: Neptunian slope
    "S3": -0.044,                       # Table 1 PDF p. 7-8: Jovian slope
    "S4": 0.881,                        # Table 1 PDF p. 7: Stellar slope
    "T1": np.log10(2.04),              # Table 1 PDF p. 8: 10^T^(1) = 2.04 M_earth
    "T2": np.log10(0.414 * _M_JUP),   # Table 1 PDF p. 9: 10^T^(2) = 0.414 M_Jup
    "T3": np.log10(0.0800 * _M_SUN),  # Table 1 PDF p. 9: 10^T^(3) = 0.0800 M_sun
}

LOCAL_FITTABLE = {}    # Type I — no per-cluster parameters


def predict(X: np.ndarray, C1: float, S1: float, S2: float, S3: float,
            S4: float, T1: float, T2: float, T3: float) -> np.ndarray:
    """Predicted radius R (R_earth) for each object.

    X: (n, 1) — column M (mass in M_earth).
    Params: LAW_CONSTANTS keys passed as keyword arguments by the harness.
    OTHER_CONSTANTS (mass ratio conversions) are read from module namespace.
    """
    M = np.asarray(X[:, 0], dtype=float)
    # Guard against non-positive masses before log10.
    M_safe = np.where(M > 0.0, M, np.nan)
    Mt = np.log10(M_safe)

    # Continuity constraint, Eq. 6, PDF p. 6:
    # C^(j+1) = C^(j) + (S^(j) - S^(j+1)) * T^(j)
    C2 = C1 + (S1 - S2) * T1
    C3 = C2 + (S2 - S3) * T2
    C4 = C3 + (S3 - S4) * T3

    # Eq. 11 piecewise model, PDF p. 6; deterministic median only.
    Rt = np.where(
        Mt <= T1,
        C1 + Mt * S1,
        np.where(
            Mt <= T2,
            C2 + Mt * S2,
            np.where(
                Mt <= T3,
                C3 + Mt * S3,
                C4 + Mt * S4,
            ),
        ),
    )
    return np.power(10.0, Rt)
