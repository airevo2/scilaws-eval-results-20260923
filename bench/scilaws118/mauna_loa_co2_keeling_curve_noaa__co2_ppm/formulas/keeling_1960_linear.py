"""Keeling (1960) linear secular rise — weakest rung of the Keeling-curve ladder.

Keeling (1960), "The concentration and isotopic abundances of carbon dioxide
in the atmosphere", Tellus 12, 200-203, first reported the secular rise of
atmospheric CO2 from the initial Mauna Loa records.  At the 1958-1959 time
horizon a simple linear-in-time fit was adequate; later work (Bacastow et
al. 1985, Keeling et al. 2001 §3 — the available PDF on disk) showed the
growth rate itself increases over time and requires a higher-order trend.

This Type I baseline ships the simplest functional form

    co2(t) = A + B * (t - T0)

with (A, B) pre-fit on the v2 train calibration window (1958-2019, 742
rows) and frozen as LAW_CONSTANTS.  The benchmark question is whether
this 2-parameter linear fit extrapolates to 2020-2026 — empirically it
does not (test r² ≈ −5), because the post-2015 acceleration sends actual
CO2 well above the linear trend.

LAW_CONSTANTS — frozen, pre-fit on v2 train calibration window
--------------------------------------------------------------
- A  = 340.6164  ppm at t = T0 (intercept)
- B  =   1.5721  ppm / yr at t = T0 (linear trend slope)

OTHER_CONSTANTS — structural epoch
----------------------------------
- T0 = 1980.0   yr (fixed reference epoch, used only to keep OLS conditioning
                    well-scaled; not a fitted quantity)

LOCAL_FITTABLE
--------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["year_decimal"]
PAPER_REF = "summary_keeling_curve.md"
EQUATION_LOC = (
    "Keeling 1960 secular-rise observation; functional form replicated as the "
    "leading-order term of the Keeling 2001 curve-fit hierarchy. (A, B) "
    "pre-fit on v2 train calibration window."
)

LAW_CONSTANTS = {
    "A": 340.6164,
    "B":   1.5721,
}
OTHER_CONSTANTS = {
    "T0": 1980.0,
}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A: float = 340.6164, B: float = 1.5721) -> np.ndarray:
    """co2(t) = A + B * (t - T0)."""
    t = np.asarray(X[:, 0], dtype=float)
    return A + B * (t - OTHER_CONSTANTS["T0"])
