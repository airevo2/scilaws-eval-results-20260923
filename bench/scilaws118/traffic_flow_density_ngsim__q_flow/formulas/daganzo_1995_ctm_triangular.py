"""Daganzo (1995) trapezoidal CTM fundamental diagram — physical rung.

Daganzo, Carlos F. (1995), "The cell transmission model, part II:
network traffic", Transportation Research Part B 29(2):79-93,
DOI 10.1016/0191-2615(94)00022-R.  Part II of Daganzo's Cell
Transmission Model extends Part I (Daganzo 1994) from a single
highway link to general networks; both parts assume the same
trapezoidal flow-density fundamental diagram:

    q(k) = min{ V_F · k,  Q_CAP,  W · (K_JAM − k) }    for 0 ≤ k ≤ K_JAM.

Three linear regimes:
  - free-flow:  q = V_F · k             (slope = free-flow speed)
  - capacity:   q = Q_CAP               (plateau at maximum flow)
  - congested:  q = W · (K_JAM − k)     (backward wave speed W)

Theoretical motivation: the Lighthill-Whitham-Richards (LWR)
kinematic-wave PDE q_t + dq/dk · q_x = 0 admits exact analytical
solutions when q(k) is piecewise-linear, and the triangular /
trapezoidal form is the simplest closed form that reproduces both
the free-flow and congested branches of empirical fundamental
diagrams (Cassidy 1998; Daganzo 1994; Newell 1993 — see
`reference/cassidy_bertini_1999.pdf` and
`reference/newell_1993_simplified.pdf`).

4 LAW_CONSTANTS — the most flexible Type I form in this bank:

LAW_CONSTANTS — frozen, pre-fit on v2 train (2286 cells)
--------------------------------------------------------
- V_F    =  28.3826    km/h    (free-flow speed)
- Q_CAP  = 2770.6410   veh/h   (capacity / plateau)
- W      =  15.9268    km/h    (backward wave speed magnitude)
- K_JAM  = 415.6429    veh/km  (jam density)

Test rmse ≈ 659.0 veh/h/lane, r² ≈ 0.508 (572 cells).

Why the trapezoid does NOT beat Underwood on NGSIM:
  Empirical NGSIM data has substantial scatter from both trajectory
  extraction error (Punzo, Borzacchiello & Ciuffo 2011, TR-C 19:1243)
  and the limited corridor diversity (us-101 + i-80 only).  The
  trapezoid's hard capacity ceiling Q_CAP and hard zero at K_JAM are
  too rigid to absorb this scatter — the smoothly-decaying Underwood
  exponential ends up with lower point-wise rmse on the held-out
  test split, even though Daganzo's form is the LWR-consistent
  choice for entropy solutions on a homogeneous link.  Manti et al.
  (2025) make the same observation in their NGSIM symbolic-
  regression study (see `reference/manti_2025.pdf`).

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["k_veh_km_lane"]
PAPER_REF = "summary_formula_daganzo_1995.md"
EQUATION_LOC = (
    "Daganzo 1995 TR-B 29(2):79, eq. 1 / Fig. 1 (PDF p. 3): "
    "q = min{V_F·k, Q_CAP, W·(K_JAM − k)}.  4 LAW constants pre-fit "
    "on v2 train by SciPy curve_fit."
)

LAW_CONSTANTS = {
    "V_F":    28.3826,
    "Q_CAP": 2770.6410,
    "W":      15.9268,
    "K_JAM": 415.6429,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray,
            V_F: float = 28.3826, Q_CAP: float = 2770.6410,
            W: float = 15.9268, K_JAM: float = 415.6429) -> np.ndarray:
    """q = min{V_F·k, Q_CAP, W·(K_JAM − k)}; trapezoidal q-k FD."""
    k = np.asarray(X[:, 0], dtype=float)
    return np.minimum(V_F * k, np.minimum(Q_CAP, W * (K_JAM - k)))
