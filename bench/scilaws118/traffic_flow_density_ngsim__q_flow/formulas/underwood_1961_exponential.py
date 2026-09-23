"""Underwood (1961) exponential fundamental diagram — middle rung.

Underwood, R. T. (1961), "Speed, Volume, and Density Relationships:
Quality and Theory of Traffic Flow", Yale Bureau of Highway Traffic,
proceedings volume; the canonical reproduction is by Drake, Schofer
and May (1967), "A Statistical Analysis of Speed-Density
Hypotheses", Highway Research Record No. 154, pp. 53-87 — see
`reference/drake_schofer_may_1967.pdf` for the formula and its
comparison against Greenshields, Greenberg, Edie, Drew, and
Pipes-Munjal alternatives on the Eisenhower Expressway dataset.

Underwood proposed a smooth exponential speed-density relation that
does not force speed to zero at any finite density:

    v(k) = V_F · exp(−k / K_M),

with V_F the free-flow speed (at k → 0) and K_M the density at
which speed equals V_F / e ≈ 0.368 · V_F (the "optimum" or
peak-flow density).  Multiplied by k:

    q(k) = V_F · k · exp(−k / K_M).

The peak (capacity) sits at k = K_M with q_cap = V_F · K_M / e.

Compared to Greenshields' parabola, Underwood's exponential:
  - falls off more gradually in the congested regime (q never quite
    reaches zero at finite density);
  - has its peak at k = K_M rather than at K_JAM/2;
  - generally fits noisy field data (NGSIM, M25, Eisenhower) better
    than the parabolic form because the asymmetric exponential
    matches the empirically wider free-flow scatter and tighter
    congested branch.

Still only 2 LAW_CONSTANTS — no piecewise structure, same parameter
budget as Greenshields.

LAW_CONSTANTS — frozen, pre-fit on v2 train (2286 cells)
--------------------------------------------------------
- V_F  = 41.4677  km/h   (free-flow speed)
- K_M  = 188.7337 veh/km (optimum / peak-flow density)

Test rmse ≈ 616.6 veh/h/lane, r² ≈ 0.569 (572 cells) — best of the
three Type I baselines.

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["k_veh_km_lane"]
PAPER_REF = "summary_formula_drake_underwood_1961_1967.md"
EQUATION_LOC = (
    "Underwood 1961 (Yale Bureau of Highway Traffic) v = V_F·exp(−k/K_M) "
    "→ q = V_F·k·exp(−k/K_M); canonical reproduction in Drake, Schofer "
    "& May 1967 HRR No. 154 pp. 53-87 (`drake_schofer_may_1967.pdf`).  "
    "(V_F, K_M) pre-fit on v2 train by SciPy curve_fit."
)

LAW_CONSTANTS = {
    "V_F": 41.4677,
    "K_M": 188.7337,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, V_F: float = 41.4677, K_M: float = 188.7337) -> np.ndarray:
    """q = V_F · k · exp(−k / K_M); Underwood-form exponential q-k FD."""
    k = np.asarray(X[:, 0], dtype=float)
    return V_F * k * np.exp(-k / K_M)
