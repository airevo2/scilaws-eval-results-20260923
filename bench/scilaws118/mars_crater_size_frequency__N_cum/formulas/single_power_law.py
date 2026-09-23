"""Single power-law crater SFD — weakest rung.

The earliest crater size-frequency studies (Young 1940; Hartmann 1964)
fitted a single straight line to the cumulative number of craters
versus diameter on a log-log plot:

    N(>=D) = A * D^(-b),   i.e.   log10 N = log10 A - b * log10 D.

A single power law assumes a scale-invariant impactor population — one
constant slope across the entire diameter range.  This captures the
gross trend (fewer big craters than small ones) but cannot reproduce
the well-documented S-shaped curvature of the real SFD: the slope of
the Mars cumulative SFD steepens at large diameters (the impact-basin
regime) and flattens in the 2-30 km range.  Neukum (2001, §2.1)
explicitly abandoned single power-law segments in the 1970s precisely
because they fail to capture this multi-segment structure.

On the v2 range-OOD split (train D < 128 km, test D >= 128 km) the
single slope fitted on small/medium craters extrapolates poorly into
the large-basin regime: the true SFD steepens beyond the fitted slope,
so the power law badly over-predicts the basin density (test
nmse_log ~ 2.8 — worse than predicting the mean).

LAW_CONSTANTS — frozen, pre-fit on v2 train (56 bins, D < 128 km)
-----------------------------------------------------------------
- A = 3.1219e-03   (cumulative density at D = 1 km, km^-2)
- B = 1.4046       (power-law slope, log-log OLS on train)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["D_lower_km"]
PAPER_REF = "summary_formula_neukum_2001.md"
EQUATION_LOC = (
    "Classical single power-law crater SFD N(>=D) = A·D^(-b) (Young 1940; "
    "Hartmann 1964; described and superseded in Neukum 2001 §2.1).  (A, B) "
    "pre-fit on v2 train by log-log OLS."
)

LAW_CONSTANTS = {
    "A": 3.1219e-03,
    "B": 1.4046,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A: float = 3.1219e-03, B: float = 1.4046) -> np.ndarray:
    """N(>=D) = A · D^(-B); single power-law cumulative SFD."""
    D = np.asarray(X[:, 0], dtype=float)
    return A * D ** (-B)
