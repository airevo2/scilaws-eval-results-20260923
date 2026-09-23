"""Linear Hottel-Whillier-Bliss efficiency — best rung.

The first-order Hottel-Whillier-Bliss (HWB) collector-efficiency law
(Hottel & Whillier 1958; ISO 9806:2017 in its Bliss inlet-temperature
variant, dropping the second-order term):

    eta = a0 - a1 * (T_in - T_amb) / G

The governing variable is the REDUCED TEMPERATURE x = (T_in - T_amb)/G:
efficiency falls linearly as the collector runs hotter relative to
ambient (more conductive/convective loss) and rises with irradiance.
a0 is the optical/heat-removal efficiency (intercept); a1 is the linear
heat-loss coefficient [W/(m^2 K)].

The reduced temperature x = (T_in - T_amb)/G is the non-trivial physical
insight an SR system must discover: it is not obvious a priori that the
temperature difference should be divided by irradiance.  Recovering this
structure lifts the test r^2 from ~0 (constant) to ~0.29.

Coefficients pre-fit on the v2 train.  The fitted intercept a0 = 0.681
matches the documented Bliss inlet-form optical efficiency for this
Arcon-Sunmark HTHEATstore-35/10 array (Tschopp 2023).  On this dataset's
narrow reduced-temperature range (0.035-0.071), the linear law is the
best of the three rungs; the second-order term (quadratic rung) is
within the measurement-noise floor and does not improve held-out error.

LAW_CONSTANTS — frozen, pre-fit on v2 train
-------------------------------------------
- A0 = 0.6811   (optical / heat-removal efficiency, intercept)
- A1 = 3.1077   (linear heat-loss coefficient, W/(m^2 K))

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["G_W_m2", "T_in_C", "T_amb_C"]
PAPER_REF = "summary_solar_collector.md"
EQUATION_LOC = (
    "Hottel-Whillier-Bliss linear efficiency eta = a0 - a1*(T_in-T_amb)/G "
    "(Hottel & Whillier 1958; ISO 9806:2017 Bliss inlet form, first order). "
    "a0, a1 pre-fit on v2 train."
)

LAW_CONSTANTS = {
    "A0": 0.6811,
    "A1": 3.1077,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A0: float = 0.6811, A1: float = 3.1077) -> np.ndarray:
    """eta = A0 - A1 * (T_in - T_amb) / G; linear HWB."""
    G    = np.asarray(X[:, 0], dtype=float)
    T_in = np.asarray(X[:, 1], dtype=float)
    Tamb = np.asarray(X[:, 2], dtype=float)
    return A0 - A1 * (T_in - Tamb) / G
