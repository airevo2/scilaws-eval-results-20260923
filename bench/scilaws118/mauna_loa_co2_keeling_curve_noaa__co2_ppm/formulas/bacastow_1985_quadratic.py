"""Bacastow et al. (1985) quadratic secular trend — middle rung of the ladder.

Bacastow, Keeling & Whorf (1985), "Seasonal amplitude increase in
atmospheric CO2 concentration at Mauna Loa, Hawaii, 1959-1982", JGR 90
(D6) 10529-10540, documented that the growth rate of atmospheric CO2 has
itself increased over time, well-described by a quadratic-in-t secular
trend.  The form is also part of the curve-fit hierarchy spelled out in
Keeling et al. 2001 §3 (PDF on disk).

This Type I baseline ships

    co2(t) = A + B * (t - T0) + C * (t - T0)^2

with (A, B, C) pre-fit on the v2 train calibration window (1958-2019,
742 rows) and frozen as LAW_CONSTANTS.  Compared to the linear rung,
the quadratic captures the accelerating-growth-rate signal and brings
test r² from -4.95 to ~0.74 — the remaining residual is dominated by
the unmodelled ~3 ppm seasonal cycle (added by the NOAA curve-fit
rung).

LAW_CONSTANTS — frozen, pre-fit on v2 train calibration window
--------------------------------------------------------------
- A  = 337.5921  ppm at t = T0
- B  =   1.3395  ppm / yr at t = T0
- C  =   0.012806 ppm / yr^2 (acceleration of secular growth)

OTHER_CONSTANTS — structural epoch
----------------------------------
- T0 = 1980.0   yr (fixed reference epoch)

LOCAL_FITTABLE
--------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["year_decimal"]
PAPER_REF = "summary_keeling_curve.md"
EQUATION_LOC = (
    "Bacastow-Keeling-Whorf 1985 acceleration observation; functional form "
    "replicated as the secular-trend term of the Keeling 2001 / NOAA GML "
    "curve-fit hierarchy. (A, B, C) pre-fit on v2 train calibration window."
)

LAW_CONSTANTS = {
    "A": 337.5921,
    "B":   1.3395,
    "C":   0.012806,
}
OTHER_CONSTANTS = {
    "T0": 1980.0,
}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A: float = 337.5921,
            B: float = 1.3395, C: float = 0.012806) -> np.ndarray:
    """co2(t) = A + B*(t-T0) + C*(t-T0)^2."""
    t = np.asarray(X[:, 0], dtype=float)
    dt = t - OTHER_CONSTANTS["T0"]
    return A + B * dt + C * dt * dt
