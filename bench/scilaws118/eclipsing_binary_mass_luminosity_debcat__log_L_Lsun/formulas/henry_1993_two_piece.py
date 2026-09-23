"""Henry-McCarthy (1993) two-piece M-L relation — middle rung.

Henry & McCarthy (1993), AJ 106:773, "The Mass-Luminosity Relation for
Stars of Mass 1.0 to 0.08 M_sun", showed that the M-L power-law slope
breaks around M ≈ 0.5-0.6 M_sun, where low-mass stars transition from
radiative to convective energy transport.  Below the break, the slope
is shallower (~1.6-2.0); above the break, the slope steepens to ~4.5.

The two-piece form:

    log10(L/L_sun) = SLOPE_LOW  · log10(M/M_sun) + INTERCEPT_LOW    if M ≤ M_BREAK
    log10(L/L_sun) = SLOPE_HIGH · log10(M/M_sun) + INTERCEPT_HIGH   if M >  M_BREAK

with M_BREAK = 0.6 M_sun is one of several published variants
(Henry-McCarthy 1993 used 0.43 M_sun; Salaris-Cassisi 2005 textbook
uses ~0.5; Eker 2018 multi-piecewise uses 0.45 as the lowest break).
For this Type I baseline M_BREAK is frozen at 0.6 M_sun and the four
(slope, intercept) pairs are pre-fit on the v2 train data.

LAW_CONSTANTS — frozen, pre-fit on v2 train
-------------------------------------------
- M_BREAK        = 0.6        M_sun   (break in slope, frozen)
- SLOPE_LOW      = 1.6019     (low-mass radiative-convective regime slope)
- INTERCEPT_LOW  = -0.9652    (low-mass intercept)
- SLOPE_HIGH     = 4.3548     (main-sequence high-mass slope)
- INTERCEPT_HIGH = 0.1839     (main-sequence intercept)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["log_M_Msun"]
PAPER_REF = "summary_formula+dataset_henry_1993.md"
EQUATION_LOC = (
    "Henry-McCarthy 1993 AJ 106:773 two-piece M-L break at M ≈ 0.5-0.6 "
    "M_sun.  Slopes + intercepts pre-fit on v2 train; M_BREAK frozen "
    "at 0.6 M_sun (intermediate between Henry-McCarthy 1993 and Eker "
    "2018 published values)."
)

LAW_CONSTANTS = {
    "M_BREAK":        0.6,
    "SLOPE_LOW":      2.9128,
    "INTERCEPT_LOW": -0.5454,
    "SLOPE_HIGH":     3.9203,
    "INTERCEPT_HIGH": 0.1284,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, M_BREAK: float = 0.6,
            SLOPE_LOW: float = 2.9128, INTERCEPT_LOW: float = -0.5454,
            SLOPE_HIGH: float = 3.9203, INTERCEPT_HIGH: float = 0.1284) -> np.ndarray:
    """log10(L/L_sun) piecewise on log10(M/M_sun) with break at M = M_BREAK."""
    log_M = np.asarray(X[:, 0], dtype=float)
    M = 10.0 ** log_M
    return np.where(M <= M_BREAK,
                    SLOPE_LOW  * log_M + INTERCEPT_LOW,
                    SLOPE_HIGH * log_M + INTERCEPT_HIGH)
