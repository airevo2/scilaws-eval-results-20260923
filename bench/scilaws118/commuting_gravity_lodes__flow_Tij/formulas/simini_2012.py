"""Simini et al. (2012) — parameter-free radiation model for commuting flows.

Citation: Simini F., González M.C., Maritan A., Barabási A.-L. (2012).
"A universal model for mobility and migration patterns." Nature 484, 96–100.
DOI: 10.1038/nature10856. arXiv: 1111.0586.
Equation: Eq. (2), PDF p. 5; T_i definition at PDF p. 5 lines 184-186.

Formula (Eq. 2, PDF p. 5):
    <T_ij> = T_i * (m_i * m_j) / [(m_i + s_ij) * (m_i + m_j + s_ij)]

with T_i = m_i * (N_c / N)

where:
    m_i   = total resident workers at origin county i
    m_j   = total job opportunities at destination county j
    s_ij  = intervening-opportunity worker count (sum of m_k for all counties k
            with 0 < d_ik < d_ij, excluding i and j)
    N_c   = total commuters in the country
    N     = total country population
    N_c/N ≈ 0.12 for US Census 2000 (PDF p. 5, line 184)

LAW_CONSTANTS — paper's scientific claim
-----------------------------------------
The radiation model is structurally parameter-free (PDF p. 5, paragraph 2).
No empirical coefficients appear in the formula — ALL inputs are measurable
population quantities. LAW_CONSTANTS = {} is correct for the pure form.

The paper's reported N_c/N ≈ 0.12 for US Census 2000 is a per-dataset
observed covariate (not a universal constant; PDF p. 5 lines 184–186). For
LODES data, T_i = m_i exactly by construction (LODES m_i is the out-sum of
T_ij, so the commuter fraction is absorbed). We expose a scalar
commuter_fraction with LAW_CONSTANTS = {"commuter_fraction": 1.0} to allow
the harness to fit the LODES-specific deviation from unity.

OTHER_CONSTANTS — structural
------------------------------
None. All operators (+, *, /) are structural. The additive 1-like structure
appears as population additions, not bare numerals.

Type designation: Type I — each county-pair row is independent; no per-
cluster LOCAL_FITTABLE parameters. The radiation model is globally parameter-
free; LAW_CONSTANTS carries only the per-dataset scalar.

Column mapping (paper → released CSV):
    m_i   → m_i       (resident workers of home county)
    m_j   → m_j       (n_j in paper; jobs at destination county)
    s_ij  → s_ij      (intervening-opportunity worker count)

Target: log10_Tij = log10(T_ij + 1).
"""

import numpy as np

USED_INPUTS = ["m_i", "m_j", "s_ij"]
PAPER_REF = "summary_formula_dataset_simini_2012.md"
EQUATION_LOC = "Eq. (2), PDF p. 5; T_i = m_i * N_c/N, PDF p. 5 lines 184-186"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "commuter_fraction": 1.0,  # N_c/N; LODES m_i is out-sum so ≈1; paper US 2000: 0.12 (PDF p. 5)
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}

LOCAL_FITTABLE = {}  # Type I — no per-cluster parameters


def predict(X: np.ndarray, commuter_fraction: float = 1.0) -> np.ndarray:
    """Radiation model: log10(T_ij + 1) for each county pair.

    X: (n, 3) — columns [m_i, m_j, s_ij] in USED_INPUTS order.
    """
    m_i = np.asarray(X[:, 0], dtype=float)
    m_j = np.asarray(X[:, 1], dtype=float)
    s_ij = np.asarray(X[:, 2], dtype=float)

    T_i = commuter_fraction * m_i
    numer = m_i * m_j
    denom = (m_i + s_ij) * (m_i + m_j + s_ij)
    # m_i >= 107 by upstream filter; denom > 0 always
    denom_safe = np.where(denom > 0.0, denom, 1.0)
    T_ij = T_i * numer / denom_safe
    T_ij = np.clip(T_ij, 0.0, None)
    return np.log10(T_ij + 1.0)
