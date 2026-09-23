"""Steinmetz with temperature correction — strongest rung.

Soft ferrites such as Ferroxcube 3C90 are not isothermal materials: the
core-loss coefficient decreases with temperature in the 0-100°C range
(by ~20-40%) before increasing again at very high T near the Curie
point.  Modern power-electronics textbooks (e.g. Kazimierczuk, "High-
Frequency Magnetic Components", §13.4) extend the Steinmetz form with
a linear-in-T correction to the log-prefactor:

    log10(P_vol) = log_k₀ + γ · T_C  +  α · log10(f) + β · log10(B),

equivalent to P_vol = k₀ · 10^(γ T_C) · f^α · B^β with k(T) =
k₀ · 10^(γ T_C).  This is the "extended Steinmetz equation" or
"K-form" frequently cited in the magnetics CAD literature.

For 3C90 on the v2 train data we recover γ ≈ -3.06×10⁻³ /°C, i.e.
the loss coefficient drops by ~0.7% per °C between 25 and 90°C — a
quantitatively meaningful correction over the v2 temperature span
that bumps test r² from 0.92 (pure Steinmetz) to 0.94 with one extra
LAW_CONSTANT.

LAW_CONSTANTS — frozen, pre-fit on v2 train
-------------------------------------------
- LOG_K0 = -4.1171     log10((kW/m^3) / (Hz^α · T^β)) at T_C = 0
- ALPHA  =  1.7216
- BETA   =  2.4279
- GAMMA  = -3.056e-03   /°C   (linear-in-T correction to log_k)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["f_Hz", "B_peak_T", "T_C"]
PAPER_REF = "summary_magnet_3c90.md"
EQUATION_LOC = (
    "Extended Steinmetz (K-form) log10(P) = log_k₀ + γ·T + α·log10(f) + "
    "β·log10(B); documented in Kazimierczuk 'High-Frequency Magnetic "
    "Components' §13.4 (cited in summary_magnet_3c90.md).  4 coefficients "
    "pre-fit on v2 train by closed-form OLS."
)

LAW_CONSTANTS = {
    "LOG_K0": -4.1171,
    "ALPHA":   1.7216,
    "BETA":    2.4279,
    "GAMMA":  -3.056e-03,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, LOG_K0: float = -4.1171,
            ALPHA: float = 1.7216, BETA: float = 2.4279,
            GAMMA: float = -3.056e-03) -> np.ndarray:
    """log10(P_vol) = LOG_K0 + GAMMA·T_C + ALPHA·log10(f) + BETA·log10(B)."""
    f = np.asarray(X[:, 0], dtype=float)
    B = np.asarray(X[:, 1], dtype=float)
    T = np.asarray(X[:, 2], dtype=float)
    return LOG_K0 + GAMMA * T + ALPHA * np.log10(f) + BETA * np.log10(B)
