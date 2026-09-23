"""Lenormand et al. (2016) — gravity law with exponential distance decay (primary).

Citation: Lenormand M., Bassolas A., Ramasco J.J. (2016). "Systematic
comparison of trip distribution laws and models." Journal of Transport
Geography 51, 158–169. DOI: 10.1016/j.jtrangeo.2015.12.008. arXiv: 1506.04889.
Equations: Eq. (2) + Eq. (3), PDF p. 4. Parameter scaling: Fig. 8a, PDF p. 10.

Formula (Eqs. 2 + 3, PDF p. 4):
    T_ij = K * m_i * m_j * exp(-beta * d_ij)

where K absorbs the production-constraint normalisation that Lenormand
applies via IPF (Eqs. 9–13, PDF pp. 4–5). On log10 scale:
    log10(T_ij + 1) ≈ log10_K + log10(m_i) + log10(m_j) - beta * d_ij / ln(10)

The exponential form is highlighted as the primary distance-decay function
(PDF p. 4, §3.2), with power-law as the secondary form (lenormand_2016_gravity_pow.py).

Caveat (Lenormand §3.9, PDF p. 9): the exponential form underestimates
long-distance flows (> ~150 km for US counties); the power form handles
long-distance tails better.

LAW_CONSTANTS — paper's scientific claim
-----------------------------------------
  beta: 0.079 (km^-1)
    — exponential distance-decay parameter for US counties.
    Scaling law: β = 0.3 * <S>^(-0.17), evaluated at <S> = 2596.8 km²
    (US counties; Table 1, PDF p. 4): 0.3 * 2596.8^(-0.17) ≈ 0.079 km^-1.
    (Fig. 8a, PDF p. 10).

  log10_K: 0.0 (dimensionless)
    — log10 of the normalisation scalar K. No published LODES-specific value;
    additive neutral default. Harness fits it per dataset.

OTHER_CONSTANTS — structural
------------------------------
  ln10: ln(10) ≈ 2.302585
    — unit-conversion factor between beta (km^-1) and the log10 model
    coefficient. Structural / mathematical constant, not a scientific claim.

Type designation: Type I — each county-pair is an independent row.
LOCAL_FITTABLE = {} (no per-cluster refitting).

Column mapping (paper → released CSV):
    m_i   → m_i      (origin resident workers)
    m_j   → m_j      (destination jobs)
    d_ij  → d_ij_km  (great-circle distance in km)

Target: log10_Tij = log10(T_ij + 1).
"""

import numpy as np

USED_INPUTS = ["m_i", "m_j", "d_ij_km"]
PAPER_REF = "summary_formula_dataset_lenormand_2016.md"
EQUATION_LOC = "Eqs. (2) + (3), PDF p. 4; beta init from Fig. 8a scaling law, PDF p. 10"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "beta": 0.079,      # km^-1; 0.3 * 2596.8^(-0.17) (Fig. 8a, PDF p. 10)
    "log10_K": 0.0,     # additive default; no published LODES value
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "ln10": 2.302585093,  # ln(10) — unit conversion; mathematical constant
}

LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters

_LN10 = OTHER_CONSTANTS["ln10"]


def predict(X: np.ndarray, beta: float = 0.079, log10_K: float = 0.0) -> np.ndarray:
    """Exponential gravity: log10(T_ij + 1) for each county pair.

    X: (n, 3) — columns [m_i, m_j, d_ij_km] in USED_INPUTS order.
    """
    m_i = np.asarray(X[:, 0], dtype=float)
    m_j = np.asarray(X[:, 1], dtype=float)
    d_ij = np.asarray(X[:, 2], dtype=float)

    # log10(T_ij) ≈ log10_K + log10(m_i) + log10(m_j) - (beta/ln10) * d_ij
    log10_mi = np.log10(np.maximum(m_i, 1.0))
    log10_mj = np.log10(np.maximum(m_j, 1.0))
    log10_T = log10_K + log10_mi + log10_mj - (beta / _LN10) * d_ij

    # Convert back: T_ij = 10^log10_T; then log10(T_ij + 1)
    T_ij = np.power(10.0, log10_T)
    T_ij = np.clip(T_ij, 0.0, None)
    return np.log10(T_ij + 1.0)
