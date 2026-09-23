"""Tóth adsorption isotherm — n_ads (mmol/g).

Kim, K.H. and Kim, M.H. (2023). "Adsorption of CO₂, CO, H₂, and N₂ on
Zeolites, Activated Carbons, and Metal-Organic Frameworks with Different
Surface Nonuniformities." *Sustainability* 15(15):11574.
DOI: 10.3390/su151511574.

The Tóth isotherm (Tóth 1971, reproduced as Eq. 3 in kim_2023_supplementary.pdf
p. 4) describes CO₂ adsorption on heterogeneous microporous surfaces:

    n = n_s * K * P / [1 + (K*P)^m]^(1/m)

where
  - n_s  is the maximum (saturation) adsorption capacity (mmol/g),
  - K    is the affinity constant (1/bar), related to binding strength,
  - m    is the Tóth heterogeneity exponent (dimensionless); m=1 → Langmuir,
         m<1 → energetically heterogeneous surface.

Kim & Kim (2023) fit this equation independently to each (adsorbent, T) pair.
Fitted parameters from Table S1 of the supplementary (supp. pp. 5–6):
  n_s in 6.154–26.887 mmol/g (across 6 adsorbents × 3 temperatures)
  K   in 0.069–50.031 1/bar  (decreases monotonically with T for every adsorbent)
  m   in 0.412–2.215  (dimensionless; zeolites <1 = high heterogeneity; Z1200 >1)

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The functional form n_s*K*P / [1+(K*P)^m]^(1/m) is the scientific
invariant. All three parameters (n_s, K, m) are per-(adsorbent, T) fit
values in Kim 2023 Table S1; no universal numerical constant appears in
the formula itself.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The exponent 1/m in [...]^(1/m) and the exponent m in (KP)^m are
both functions of the LOCAL_FITTABLE parameter m; no separate numeric
literal appears.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- n_s : saturation capacity (mmol/g, > 0).
        Kim 2023 Table S1 range: 6.154–26.887 mmol/g.
- K   : affinity constant (1/bar, > 0).
        Kim 2023 Table S1 range: 0.069–50.031 1/bar.
- m   : Tóth heterogeneity exponent (dimensionless, > 0).
        Kim 2023 Table S1 range: 0.412–2.215.
        m=1 recovers the Langmuir isotherm exactly.

init = None on all three: fit() builds data-derived starting estimates.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["P_bar"]
PAPER_REF = "summary_formula_kim_2023.md"
EQUATION_LOC = (
    "Kim & Kim (2023) Supplementary Eq. (3), kim_2023_supplementary.pdf p. 4: "
    "n = n_s * K * P / [1 + (K*P)^m]^(1/m)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "n_s": {"init": None},
    "K":   {"init": None},
    "m":   {"init": None},
}


def _toth(P, n_s, K, m):
    """Tóth isotherm: n = n_s * K * P / [1 + (K*P)^m]^(1/m)."""
    KP = K * np.asarray(P, dtype=float)
    # clip KP^m to avoid overflow when m is large; physically KP is O(1–50)
    KPm = np.clip(KP, 0.0, None) ** m
    denom = (1.0 + KPm) ** (1.0 / m)
    return n_s * KP / denom


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the Tóth isotherm.

    Data-derived starting estimates:
      n_s0 = 1.5 * max(y)       — saturation is above the observed max
      K0   = 1.0                — neutral affinity (1/bar)
      m0   = 0.7                — mid-range heterogeneous value; Kim 2023
                                  Table S1 has many clusters with m~0.5–1.0

    Bounds follow Kim 2023 Table S1 ranges with physical margin.
    Single-start Levenberg-Marquardt / trust-region is sufficient for the
    Tóth form on Type-I isotherm data (smooth, convex, well-conditioned).
    """
    P = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    y_max = float(np.max(y)) if y.size > 0 else 1.0
    n_s0 = max(1.5 * y_max, 1.0)
    K0   = 1.0
    m0   = 0.7

    # Physical bounds: n_s > 0; K > 0; m in (0.05, 5]
    param_lo = [1e-3, 1e-4, 0.05]
    param_hi = [200.0, 500.0, 5.0]
    p0 = [n_s0, K0, m0]
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, param_lo, param_hi)]

    def residual(p):
        return _toth(P, p[0], p[1], p[2]) - y

    try:
        sol = least_squares(residual, p0, bounds=(param_lo, param_hi),
                            method="trf", max_nfev=5000)
        n_s, K, m = sol.x
        if not np.all(np.isfinite([n_s, K, m])):
            raise RuntimeError("non-finite fit")
        return {"n_s": float(n_s), "K": float(K), "m": float(m)}
    except Exception:                                   # noqa: BLE001
        return {"n_s": float(p0[0]), "K": float(p0[1]), "m": float(p0[2])}


def predict(X: np.ndarray, n_s: float, K: float, m: float) -> np.ndarray:
    """Tóth isotherm: n = n_s * K * P / [1 + (K*P)^m]^(1/m).

    X: (n, 1) — column [P_bar].
    Returns n_ads in mmol/g.
    """
    P = np.asarray(X[:, 0], dtype=float)
    return _toth(P, n_s, K, m)
