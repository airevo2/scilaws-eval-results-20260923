"""Lenormand et al. (2016) — extended radiation law for commuting flows.

Citation: Lenormand M., Bassolas A., Ramasco J.J. (2016). "Systematic
comparison of trip distribution laws and models." Journal of Transport
Geography 51, 158–169. DOI: 10.1016/j.jtrangeo.2015.12.008. arXiv: 1506.04889.
Equation: Eq. (8), PDF p. 4. Parameter scaling: Fig. 8d, PDF p. 10.

Formula (Eq. 8, PDF p. 4):
    P(1|m_i, m_j, s_ij) = [(m_i+m_j+s_ij)^α - (m_i+s_ij)^α] * (m_i^α + 1)
                           / {[(m_i+s_ij)^α + 1] * [(m_i+m_j+s_ij)^α + 1]}

Absolute flow: T_ij = Nc_over_N * m_i * P(1|m_i, m_j, s_ij)

where:
    α     = opportunity-sensitivity exponent (fit per dataset)
    m_i   = origin resident workers
    m_j   = destination job count
    s_ij  = intervening-opportunity worker count
    Nc_over_N = per-dataset commuter fraction (absorbs T_i normalisation)

LAW_CONSTANTS — paper's scientific claim
-----------------------------------------
The formula form (Eq. 8) is the law; parameters are fit per dataset.
No universal constants appear in the publication.

  alpha: 1.91 (dimensionless)
    — estimated from Fig. 8d scaling law: α = 0.02 * <S>^0.58,
      evaluated at <S> = 2596.8 km² (US counties; Table 1, PDF p. 4):
      0.02 * 2596.8^0.58 ≈ 1.91. Used as init; harness re-fits to LODES.

  Nc_over_N: 1.0 (dimensionless)
    — Per-dataset commuter fraction (T_i = Nc_over_N * m_i). LODES m_i
      is the out-sum of T_ij, so ≈ 1 by construction. No published
      universal value. Multiplicative neutral default.

OTHER_CONSTANTS — structural
------------------------------
The additive "+1" constants in (m_i^α + 1), [(m_i+s_ij)^α + 1], and
[(m_i+m_j+s_ij)^α + 1] are structural fingerprint constants of the
extended radiation form (Eq. 8, PDF p. 4) — NOT free parameters.

Type designation: Type I — each county-pair is an independent row.
LOCAL_FITTABLE = {} (no per-cluster refitting; α is a global dataset-level
parameter as in Lenormand 2016 §3, PDF p. 4).

Column mapping (paper → released CSV):
    m_i   → m_i    (origin resident workers)
    m_j   → m_j    (n_j in paper; destination jobs)
    s_ij  → s_ij   (intervening-opportunity worker count)

Target: log10_Tij = log10(T_ij + 1).
"""

import numpy as np

USED_INPUTS = ["m_i", "m_j", "s_ij"]
PAPER_REF = "summary_formula_dataset_lenormand_2016.md"
EQUATION_LOC = "Eq. (8), PDF p. 4; alpha init from Fig. 8d scaling, PDF p. 10"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "alpha": 1.91,       # dimensionless; 0.02 * 2596.8^0.58 (Fig. 8d, PDF p. 10)
    "Nc_over_N": 1.0,    # per-dataset commuter fraction; no published universal value
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}     # structural "+1" terms are inline numerals in the formula

LOCAL_FITTABLE = {}      # Type I — no per-cluster parameters


def predict(X: np.ndarray, alpha: float = 1.91, Nc_over_N: float = 1.0) -> np.ndarray:
    """Extended radiation model: log10(T_ij + 1) for each county pair.

    X: (n, 3) — columns [m_i, m_j, s_ij] in USED_INPUTS order.
    """
    m_i = np.asarray(X[:, 0], dtype=float)
    m_j = np.asarray(X[:, 1], dtype=float)
    s_ij = np.asarray(X[:, 2], dtype=float)

    A = m_i + s_ij
    B = m_i + m_j + s_ij

    # Guard against alpha=0 degeneracy (logs return 0)
    A_a = np.power(np.maximum(A, 0.0), alpha)
    B_a = np.power(np.maximum(B, 0.0), alpha)
    mi_a = np.power(np.maximum(m_i, 0.0), alpha)

    numer = (B_a - A_a) * (mi_a + 1.0)
    denom = (A_a + 1.0) * (B_a + 1.0)
    P = np.where(denom > 0.0, numer / denom, 0.0)

    T_i = Nc_over_N * m_i
    T_ij = np.clip(T_i * P, 0.0, None)
    return np.log10(T_ij + 1.0)
