"""Lenormand et al. (2016) — gravity law with power-law distance decay.

Citation: Lenormand M., Bassolas A., Ramasco J.J. (2016). "Systematic
comparison of trip distribution laws and models." Journal of Transport
Geography 51, 158–169. DOI: 10.1016/j.jtrangeo.2015.12.008. arXiv: 1506.04889.
Equations: Eq. (2) + Eq. (4), PDF p. 4. Parameter scaling: Fig. 8b, PDF p. 10.

Formula (Eqs. 2 + 4, PDF p. 4):
    T_ij = K * m_i * m_j * d_ij^(-beta)

where K absorbs the production-constraint normalisation that Lenormand
applies via IPF (Eqs. 9–13, PDF pp. 4–5). On log10 scale:
    log10(T_ij + 1) ≈ log10_K + log10(m_i) + log10(m_j) - beta * log10(d_ij)

Lenormand 2016 presents both the exponential (Eq. 3) and power (Eq. 4)
distance-decay forms as co-equal primary candidates (PDF p. 3, abstract;
compared across all eight case studies in Figs. 4–7). The power form handles
long-distance tails better than the exponential form (PDF p. 9, §3.9).

LAW_CONSTANTS — paper's scientific claim
-----------------------------------------
  beta: 3.32 (dimensionless)
    — power-law distance-decay exponent for US counties.
    Scaling law: β = 1.4 * <S>^0.11, evaluated at <S> = 2596.8 km²
    (US counties; Table 1, PDF p. 4): 1.4 * 2596.8^0.11 ≈ 3.32.
    (Fig. 8b, PDF p. 10; extracted-text line 1244: "β = 1.4 ⋅ <S>0.11").

  log10_K: 0.0 (dimensionless)
    — log10 of the normalisation scalar K. No published LODES-specific value;
    additive neutral default. Harness fits it per dataset.

OTHER_CONSTANTS — structural
------------------------------
  (none — the power-law form requires no unit-conversion constants;
   log10 of both sides removes unit factors algebraically)

Type designation: Type I — each county-pair is an independent row.
LOCAL_FITTABLE = {} (no per-cluster refitting).

Column mapping (paper → released CSV):
    m_i   → m_i      (origin resident workers)
    m_j   → m_j      (destination jobs)
    d_ij  → d_ij_km  (great-circle distance in km)

Target: log10_Tij = log10(T_ij + 1).

Caveat (Lenormand §3.9, PDF p. 9): the power form handles long-distance
tails better than exponential decay but may overestimate short-range flows.
The OOD test split (d_ij_km ≥ 600 km) probes the long-haul regime where
the power form has the comparative advantage.
"""

import numpy as np

USED_INPUTS = ["m_i", "m_j", "d_ij_km"]
PAPER_REF = "summary_formula_dataset_lenormand_2016.md"
EQUATION_LOC = "Eqs. (2) + (4), PDF p. 4; beta init from Fig. 8b scaling law, PDF p. 10"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "beta": 3.32,       # dimensionless; 1.4 * 2596.8^0.11 (Fig. 8b, PDF p. 10)
    "log10_K": 0.0,     # additive default; no published LODES value
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}    # no unit-conversion constant needed for power form

LOCAL_FITTABLE = {}     # Type I — no per-cluster parameters


def predict(X: np.ndarray, beta: float = 3.32, log10_K: float = 0.0) -> np.ndarray:
    """Power-law gravity: log10(T_ij + 1) for each county pair.

    X: (n, 3) — columns [m_i, m_j, d_ij_km] in USED_INPUTS order.
    """
    m_i = np.asarray(X[:, 0], dtype=float)
    m_j = np.asarray(X[:, 1], dtype=float)
    d_ij = np.asarray(X[:, 2], dtype=float)

    # log10(T_ij) ≈ log10_K + log10(m_i) + log10(m_j) - beta * log10(d_ij)
    log10_mi = np.log10(np.maximum(m_i, 1.0))
    log10_mj = np.log10(np.maximum(m_j, 1.0))
    # Guard against d_ij=0 (shouldn't occur — all county pairs have d > 0)
    log10_d = np.log10(np.maximum(d_ij, 1e-9))
    log10_T = log10_K + log10_mi + log10_mj - beta * log10_d

    # Convert back: T_ij = 10^log10_T; then log10(T_ij + 1)
    T_ij = np.power(10.0, log10_T)
    T_ij = np.clip(T_ij, 0.0, None)
    return np.log10(T_ij + 1.0)
