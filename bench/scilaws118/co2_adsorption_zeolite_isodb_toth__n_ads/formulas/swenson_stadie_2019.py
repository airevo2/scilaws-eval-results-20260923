"""Langmuir adsorption isotherm — n_ads (mmol/g); m=1 special case of Tóth.

Swenson, H. and Stadie, N.P. (2019). "Langmuir's Theory of Adsorption:
A Centennial Review." *Langmuir* 35(16):5409–5426.
DOI: 10.1021/acs.langmuir.9b00154.

Langmuir's adsorption isotherm (Eq. 1, PDF p. 2; Langmuir 1918 originally):

    theta = K * P / (1 + K * P)

In adsorbed-amount form (theta = n / n_s):

    n = n_s * K * P / (1 + K * P)

This is the m=1 special case of the Tóth isotherm (kim_2023.py): setting
m=1 in n_s*K*P / [1+(K*P)^m]^(1/m) gives [1+(K*P)]^1 = 1+KP, recovering
Langmuir exactly.  Swenson & Stadie (2019) derive this from kinetic
equilibrium of adsorption/desorption rates and discuss its validity for
homogeneous (single-site-energy) adsorbents.

In the Kim 2023 dataset, Z1200 (MOF) has m ≈ 2.2 (near-homogeneous
relative to the zeolites) while zeolites NaY and NaX have m ≈ 0.5–0.7
(highly heterogeneous).  Langmuir is included as a physics-motivated
homogeneous-surface baseline: a valid SR solution that cannot resolve
the Tóth heterogeneity will converge to this form.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The +1 in (1 + KP) is an algebraic structural constant of the
monolayer-saturation kinetic derivation (Swenson & Stadie 2019 Eq. 1,
PDF p. 2), not a fitted numerical parameter.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The literal 1 in the denominator (1 + KP) is a pure structural
integer from Langmuir's kinetic equilibrium derivation; it is not a
tuneable constant.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- n_s : monolayer saturation capacity (mmol/g, > 0).
        Analogous to Kim 2023 Table S1 n_s column (m=1 constraint applied).
- K   : Langmuir affinity constant (1/bar, > 0).
        Decreases monotonically with temperature (Swenson & Stadie 2019
        Eq. 2, PDF p. 2: K ∝ 1/sqrt(T) * 1/r_des_sat).

init = None on both: fit() builds data-derived starting estimates.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["P_bar"]
PAPER_REF = "summary_formula_swenson_stadie_2019.md"
EQUATION_LOC = (
    "Swenson & Stadie (2019) Eq. 1, PDF p. 2: theta = K*P / (1 + K*P); "
    "adsorbed-amount form: n = n_s * K * P / (1 + K * P).  "
    "Confirmed as m=1 limit of Tóth (Kim 2023 supp. Eq. 3, p. 4)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "n_s": {"init": None},
    "K":   {"init": None},
}


def _langmuir(P, n_s, K):
    """Langmuir isotherm: n = n_s * K * P / (1 + K * P)."""
    KP = K * np.asarray(P, dtype=float)
    return n_s * KP / (1.0 + KP)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the Langmuir isotherm.

    Data-derived starting estimates:
      n_s0 = 1.5 * max(y)       — saturation lies above the observed maximum
      K0   = 1.0                — neutral affinity (1/bar)

    Single-start trust-region reflective is sufficient; the Langmuir form
    is smooth, convex in P, and well-conditioned on Type-I isotherm data.
    """
    P = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    y_max = float(np.max(y)) if y.size > 0 else 1.0
    n_s0 = max(1.5 * y_max, 1.0)
    K0   = 1.0

    param_lo = [1e-3, 1e-4]
    param_hi = [200.0, 500.0]
    p0 = [n_s0, K0]
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, param_lo, param_hi)]

    def residual(p):
        return _langmuir(P, p[0], p[1]) - y

    try:
        sol = least_squares(residual, p0, bounds=(param_lo, param_hi),
                            method="trf", max_nfev=4000)
        n_s, K = sol.x
        if not np.all(np.isfinite([n_s, K])):
            raise RuntimeError("non-finite fit")
        return {"n_s": float(n_s), "K": float(K)}
    except Exception:                                   # noqa: BLE001
        return {"n_s": float(p0[0]), "K": float(p0[1])}


def predict(X: np.ndarray, n_s: float, K: float) -> np.ndarray:
    """Langmuir isotherm: n = n_s * K * P / (1 + K * P).

    X: (n, 1) — column [P_bar].
    Returns n_ads in mmol/g.
    """
    P = np.asarray(X[:, 0], dtype=float)
    return _langmuir(P, n_s, K)
