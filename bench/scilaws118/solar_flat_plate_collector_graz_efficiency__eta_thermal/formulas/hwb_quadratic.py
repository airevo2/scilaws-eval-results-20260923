"""Quadratic Hottel-Whillier-Bliss efficiency (full ISO 9806 form).

The full second-order Hottel-Whillier-Bliss collector-efficiency law
(Hottel & Whillier 1958; standardised in ISO 9806:2017, Bliss
inlet-temperature variant):

    eta = a0 - a1 * (T_in - T_amb)/G - a2 * (T_in - T_amb)^2 / G

The second-order term -a2*(T_in - T_amb)^2/G captures the radiative
(re-radiation) component of the heat loss, which grows faster than
linearly as the absorber heats up; a2 is the quadratic loss coefficient
[W/(m^2 K^2)].  This is the certified test form reported for the
Arcon-Sunmark HTHEATstore-35/10 collector.

Coefficients pre-fit on the v2 train give a0 = 0.670, a1 = 2.290,
a2 = 0.0133, consistent with the documented Bliss inlet-form values
(a0 = 0.678, a1 = 2.58, a2 = 0.0104; Tschopp 2023).  IMPORTANT: on this
dataset's narrow reduced-temperature range (0.035-0.071 K m^2/W), the
quadratic term is within the measurement-noise floor — it does NOT
improve held-out error over the linear rung (test rmse ~ 0.0204 vs the
linear rung's ~0.0203).  This is physically expected: the a2 curvature
only becomes important at high reduced temperatures (stagnation-
approaching operation) not sampled here.  The rung is included as the
structurally-complete ISO 9806 form; it is essentially tied with the
linear rung on this data.

LAW_CONSTANTS — frozen, pre-fit on v2 train
-------------------------------------------
- A0 = 0.6700   (optical / heat-removal efficiency, intercept)
- A1 = 2.2905   (linear heat-loss coefficient, W/(m^2 K))
- A2 = 0.01326  (quadratic heat-loss coefficient, W/(m^2 K^2))

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["G_W_m2", "T_in_C", "T_amb_C"]
PAPER_REF = "summary_solar_collector.md"
EQUATION_LOC = (
    "Hottel-Whillier-Bliss full quadratic efficiency "
    "eta = a0 - a1*(T_in-T_amb)/G - a2*(T_in-T_amb)^2/G (ISO 9806:2017 "
    "Bliss inlet form; Tschopp 2023 Table 3).  a0,a1,a2 pre-fit on v2 train."
)

LAW_CONSTANTS = {
    "A0": 0.6700,
    "A1": 2.2905,
    "A2": 0.01326,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A0: float = 0.6700, A1: float = 2.2905,
            A2: float = 0.01326) -> np.ndarray:
    """eta = A0 - A1*(T_in-T_amb)/G - A2*(T_in-T_amb)^2/G; quadratic HWB."""
    G    = np.asarray(X[:, 0], dtype=float)
    T_in = np.asarray(X[:, 1], dtype=float)
    Tamb = np.asarray(X[:, 2], dtype=float)
    dT = T_in - Tamb
    return A0 - A1 * dT / G - A2 * dT * dT / G
