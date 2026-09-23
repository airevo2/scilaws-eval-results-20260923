"""Markidis et al. (2025) DDM#1 — best-performing PySR data-driven model.

Citation: S. Markidis, E. Tsourdi & M. Vianello, "Discovering Governing
Equations of Geomagnetic Storm Dynamics with Symbolic Regression,"
arXiv:2504.18461, ICCS 2025. DOI: 10.48550/arXiv.2504.18461.

Formula (Table 3 row "DDM#1 (C:19)", PDF p. 9):
    dDst/dt = [-0.036*(P_dyn + Dst) - max(-0.008*Dst, Ey)]
              * sqrt(P_dyn + 1.278) + 0.319

Disambiguation note (PDF p. 9 visually confirmed):
    The radical encloses (P_dyn + 1.278); the offset 0.319 is the
    additive constant outside the full expression. The existing
    v0.x baseline and the PDF Table 3 typography agree on this reading.
    The GitHub repo DSTequations_ranked.csv does not contain the exact
    published DDM#1 coefficients (the repo stores different PySR runs),
    but the PDF Table 3 is unambiguous.

LAW_CONSTANTS (Table 3 DDM#1, PDF p. 9):
    a    = -0.036   coupling of (P_dyn + Dst) in outer bracket
    b    = -0.008   coefficient of Dst inside max()
    c    =  1.278   nPa additive offset inside sqrt (radicand)
    d    =  0.319   nT/hr additive offset outside entire expression

OTHER_CONSTANTS: none (all four scalars from the published SR output table).

Type: Type I. All four constants are global PySR fits across the
1995-Mar-2021 OMNI hourly training window (Markidis 2025 §3.3, PDF p. 7).
No per-storm refit.

Column mapping:
    paper P_dyn -> P_dyn (nPa)
    paper Dst   -> Dst (nT)
    paper Ey    -> Ey (mV/m)
"""

import numpy as np

USED_INPUTS = ["Dst", "Ey", "P_dyn"]
PAPER_REF = "summary_formula+dataset_markidis_2025.md"
EQUATION_LOC = "Markidis 2025 Table 3 row 'DDM#1 (C:19)', PDF p. 9"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a":  -0.036,   # coupling (P_dyn+Dst) — Table 3 DDM#1, PDF p. 9
    "b":  -0.008,   # Dst coefficient in max() — Table 3 DDM#1, PDF p. 9
    "c":   1.278,   # nPa offset inside sqrt — Table 3 DDM#1, PDF p. 9
    "d":   0.319,   # nT/hr additive offset — Table 3 DDM#1, PDF p. 9
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}

LOCAL_FITTABLE = {}


def predict(X: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Predict dDst/dt from (Dst, Ey, P_dyn) using DDM#1.

    X.shape = (n, 3); columns in USED_INPUTS order: Dst, Ey, P_dyn.
    LAW_CONSTANTS arrive as named params via predict(X, **LAW_CONSTANTS).
    """
    Dst   = np.asarray(X[:, 0], dtype=float)
    Ey    = np.asarray(X[:, 1], dtype=float)
    P_dyn = np.asarray(X[:, 2], dtype=float)

    inner = a * (P_dyn + Dst) - np.maximum(b * Dst, Ey)
    return inner * np.sqrt(np.maximum(P_dyn + c, 0.0)) + d
