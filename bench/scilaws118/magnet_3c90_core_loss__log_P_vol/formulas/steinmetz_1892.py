"""Steinmetz (1892) empirical power law — canonical core-loss baseline.

Steinmetz (1892), "Die Lehre vom Drehstrom",  proposed the empirical
power-law scaling of magnetic core losses with frequency and flux
density that bears his name in modern textbooks:

    P_vol(f, B) = k · f^α · B^β,

with k, α, β material-dependent fit constants.  Unlike Bertotti's
physics-grounded loss separation, the Steinmetz exponents are
empirical: α typically ~1.5-1.8 for soft ferrites in the kHz-MHz
band (close to but distinct from Bertotti's hysteresis exponent
1.0), and β typically ~2.3-2.7 (above the geometric 2.0 of Bertotti's
B² scaling, reflecting the saturation softening of the hysteresis
loop at high flux).

In log10 space the law is linear:

    log10(P_vol) = log10(k) + α · log10(f) + β · log10(B),

so fit() is closed-form OLS on the v2 train rows.  (log_k, α, β) are
pre-fit and frozen as LAW_CONSTANTS.  3 LAW_CONSTANTS, no fitted
exponents = 3.  Steinmetz is the most-cited core-loss baseline in
power-electronics literature.

LAW_CONSTANTS — frozen, pre-fit on v2 train
-------------------------------------------
- LOG_K = -4.2811    log10((kW/m^3) / (Hz^α · T^β))
- ALPHA =  1.7178    frequency exponent (3C90 in 50 kHz - 450 kHz band)
- BETA  =  2.4236    flux-density exponent (3C90 at 25-90°C)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["f_Hz", "B_peak_T"]
PAPER_REF = "summary_magnet_3c90.md"
EQUATION_LOC = (
    "Steinmetz 1892 power-law P = k · f^α · B^β; canonical core-loss form "
    "documented in summary_magnet_3c90.md and the MagNet open-access paper "
    "on disk; (log_k, α, β) pre-fit on v2 train by closed-form OLS in "
    "log-log space."
)

LAW_CONSTANTS = {
    "LOG_K": -4.2811,
    "ALPHA":  1.7178,
    "BETA":   2.4236,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, LOG_K: float = -4.2811,
            ALPHA: float = 1.7178, BETA: float = 2.4236) -> np.ndarray:
    """log10(P_vol) = LOG_K + ALPHA · log10(f) + BETA · log10(B)."""
    f = np.asarray(X[:, 0], dtype=float)
    B = np.asarray(X[:, 1], dtype=float)
    return LOG_K + ALPHA * np.log10(f) + BETA * np.log10(B)
