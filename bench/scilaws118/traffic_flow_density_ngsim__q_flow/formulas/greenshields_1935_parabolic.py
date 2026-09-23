"""Greenshields (1935) parabolic fundamental diagram — weakest rung.

Greenshields, Bruce D. (1935), "A Study of Traffic Capacity",
Proceedings of the Highway Research Board, Vol. 14, pp. 448-477.
The first empirical study to formalise the speed-density and
flow-density relationships of vehicular traffic, based on aerial
photographic measurements of cars on a 2-lane Ohio highway.

Greenshields proposed a linear speed-density relation

    v(k) = V_F · (1 − k / K_JAM)

(free-flow speed V_F at zero density, falling linearly to zero at
the jam density K_JAM).  Multiplied by k to convert speed into flow
gives the parabolic flow-density fundamental diagram

    q(k) = V_F · k · (1 − k / K_JAM),

a downward-opening parabola with q(0) = q(K_JAM) = 0 and a
single maximum (capacity) at k = K_JAM/2 with q_cap = V_F·K_JAM/4.
This is the OLDEST published fundamental-diagram closed form and is
still routinely used as a pedagogical starting point.

The parabola has only 2 LAW_CONSTANTS, no piecewise structure, and
imposes a hard relation q_cap/v_f = K_JAM/4 between capacity and jam
density.  It generally over-predicts flow at low densities (the
free-flow speed of real highways is ~100 km/h, but Greenshields'
single-parameter form forces V_F to absorb both the free-flow speed
and a global rescaling to match the observed capacity).

LAW_CONSTANTS — frozen, pre-fit on v2 train (2286 cells)
--------------------------------------------------------
- V_F    = 34.1818  km/h   (effective free-flow speed coefficient)
- K_JAM  = 351.8164 veh/km (jam density)

Test rmse ≈ 633.8 veh/h/lane, r² ≈ 0.545 (572 cells).

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["k_veh_km_lane"]
PAPER_REF = "summary_formula_greenshields_1935.md"
EQUATION_LOC = (
    "Greenshields 1935 Highway Research Board Proc. 14:448, eq. relating "
    "v = V_F·(1 − k/K_JAM) → q = V_F·k·(1 − k/K_JAM).  (V_F, K_JAM) "
    "pre-fit on v2 train by SciPy curve_fit; no piecewise structure."
)

LAW_CONSTANTS = {
    "V_F":   34.1818,
    "K_JAM": 351.8164,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, V_F: float = 34.1818, K_JAM: float = 351.8164) -> np.ndarray:
    """q = V_F · k · (1 − k / K_JAM); parabolic q-k fundamental diagram."""
    k = np.asarray(X[:, 0], dtype=float)
    return V_F * k * (1.0 - k / K_JAM)
