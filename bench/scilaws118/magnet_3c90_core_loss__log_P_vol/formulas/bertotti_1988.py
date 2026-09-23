"""Bertotti (1988) loss separation — weakest rung, physics-grounded.

Bertotti (1988), IEEE Trans. Magn. 24(1):621-630, "General properties
of power losses in soft ferromagnetic materials", decomposes ac core
losses into two physical mechanisms whose frequency dependence is
distinct: a static-hysteresis term (∝ f) and a dynamic eddy-current
term (∝ f²).  For a pure-sinusoidal flux excitation with peak B,

    P_vol(f, B) = k_h · f · B²  +  k_e · f² · B²,

where k_h is the hysteresis-loss coefficient (depends on the
ferromagnetic domain structure) and k_e is the eddy-current
coefficient (depends on bulk electrical conductivity and lamination
thickness).  A third "excess" term (∝ f^1.5 · B^1.5; Bertotti's
"anomalous" loss) is omitted here because its coefficient is not
strongly constrained by the available 3C90 measurements.

This is the simplest physically-grounded baseline: 2 LAW_CONSTANTS
(k_h, k_e), no fitted exponents.  Predictions are produced in
LINEAR P_vol units (kW/m^3) and then log10-transformed to match the
v2 task target `log_P_vol`.  (k_h, k_e) are pre-fit on the v2 train
rows by scipy curve_fit on linear P_vol.

LAW_CONSTANTS — frozen, pre-fit on v2 train
-------------------------------------------
- K_H = 1.1351e-01   ((kW/m^3) / (Hz · T^2))   hysteresis-loss coeff
- K_E = 1.5515e-07   ((kW/m^3) / (Hz^2 · T^2)) eddy-current coeff

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["f_Hz", "B_peak_T"]
PAPER_REF = "summary_magnet_3c90.md"
EQUATION_LOC = (
    "Bertotti 1988 loss-separation theorem (IEEE Trans. Magn. 24(1):621); "
    "two-term P = k_h·f·B² + k_e·f²·B²; (k_h, k_e) pre-fit on v2 train."
)

LAW_CONSTANTS = {
    "K_H": 1.1351e-01,
    "K_E": 1.5515e-07,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, K_H: float = 1.1351e-01,
            K_E: float = 1.5515e-07) -> np.ndarray:
    """log10(P_vol) where P_vol = K_H·f·B² + K_E·f²·B² (kW/m^3)."""
    f = np.asarray(X[:, 0], dtype=float)
    B = np.asarray(X[:, 1], dtype=float)
    P_lin = K_H * f * B * B + K_E * (f * B) * (f * B)
    return np.log10(np.maximum(P_lin, 1.0e-9))
