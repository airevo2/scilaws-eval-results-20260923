"""Peleg (2021) static Chick-Watson-Hom model on CT dose — LRV = k * CT^m.

Peleg, M. (2021). Modeling the dynamic kinetics of microbial disinfection
with dissipating chemical agents—a theoretical investigation. Applied
Microbiology and Biotechnology 105(2):539-549.
DOI:10.1007/s00253-020-11042-8. PMC7780086.

The static Chick-Watson-Hom (CWH) model, Eq. (2) of Peleg (2021) (PDF
p. 2, journal p. 540):

    Log_10[S(t)] = -k * C^n * t^m

and thus LRV = k * C^n * t^m.

With n = 1 (linear concentration dependence — the Chick law assumption)
and CT = C * t being the measured dose:

    LRV = k * C * t^m = k * (CT/t) * t^m = k * CT * t^(m-1)

This is NOT reducible to CT alone without t individually. However, a
useful CT-axis generalisation is the power-law dose model widely used in
disinfection engineering:

    LRV = k * CT^m

This form treats CT as the effective dose with a free exponent m
(Hom 1972 dose-exponent generalisation; Peleg 2021 PDF p. 2). It is
equivalent to CWH with n = 1 when C and t co-vary proportionally across
conditions, and is commonly used in practice (Gyurek and Finch 1998;
Haas and Kara 1984, referenced by Peleg 2021 PDF pp. 1-2). The m=1
special case recovers peleg_2021_cw.py.

Reference equation: Eq. (2) Peleg (2021) with CT-dose interpretation
per Hom (1972). Peleg 2021 PDF p. 2 frames m as the Weibull shape factor
("m(C) a power, the Weibullian shape factor") and recovers Chick-Watson
as the m=1 case ("the original Chick-Watson is just a special case of the
CWH model where m = 1, and the original Chick model where m = n = 1").

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. k, n, m are all specimen/disinfectant specific (Peleg 2021 PDF
p. 2). The CT-power-law form itself is the structural claim.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) Log base 10 is a structural convention.

LOCAL_FITTABLE — per-cluster, fitted via bounded NLS
-----------------------------------------------------
- k : rate constant per CT^m unit (> 0). Refit per cluster.
- m : Weibullian dose exponent / time-shape factor (> 0). m=1 recovers
      linear-CT model. Refit per cluster.
      init = None on both: fit() uses log-linear closed-form warm start.

Type II: two parameters (k, m) per (disinfectant x organism) cluster.
Column mapping: CT (mg/L*min) -> CT column; LRV -> log_reduction target.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["CT"]
PAPER_REF = "summary_formula_peleg_2021.md"
EQUATION_LOC = (
    "Structural form from Peleg (2021) Eq. (2), PDF p. 2, journal p. 540 — "
    "LRV = k * C^n * t^m. Implemented as the CT-dose power law LRV = k * CT^m, the "
    "standard engineering adaptation used when only CT (not C and t separately) is "
    "reported (Gyurek & Finch 1998; Haas & Kara 1984, cited Peleg 2021 PDF pp. 1-2). "
    "This is NOT literally Eq. (2) — k*C^n*t^m is not reducible to CT alone; see the "
    "module docstring derivation. Time exponent m per Hom (1972), Peleg 2021 PDF p. 2."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "k": {"init": None},
    "m": {"init": None},
}


def _lrv(CT, k, m):
    CT_safe = np.clip(CT, 1e-12, None)
    return k * (CT_safe ** m)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded NLS for LRV = k * CT^m.

    Log-linearise: ln(LRV) = ln(k) + m * ln(CT).
    Use OLS on the log-linearised form as warm start, then bounded NLS.

    X_fit: (n, 1) — column [CT].
    y_fit: (n,)   — LRV.
    """
    CT = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # Log-linear warm start (only where CT > 0 and y > 0)
    valid = (CT > 0) & (y > 0)
    if valid.sum() >= 2:
        log_CT = np.log(CT[valid])
        log_y = np.log(y[valid])
        A = np.column_stack([np.ones_like(log_CT), log_CT])
        coef, *_ = np.linalg.lstsq(A, log_y, rcond=None)
        ln_k0, m0 = float(coef[0]), float(coef[1])
        k0 = float(np.exp(ln_k0))
        k0 = max(min(k0, 1e3), 1e-9)
        m0 = max(min(m0, 10.0), 1e-3)
    else:
        k0, m0 = 1e-3, 1.0

    p0 = [k0, m0]
    param_lo = [1e-9, 1e-3]
    param_hi = [1e3,  10.0]
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, param_lo, param_hi)]

    def residual(p):
        return _lrv(CT, p[0], p[1]) - y

    try:
        sol = least_squares(residual, p0, bounds=(param_lo, param_hi),
                            method="trf", max_nfev=4000)
        k_fit, m_fit = sol.x
        if not np.all(np.isfinite([k_fit, m_fit])):
            raise RuntimeError("non-finite")
        return {"k": float(k_fit), "m": float(m_fit)}
    except Exception:  # noqa: BLE001
        return {"k": float(p0[0]), "m": float(p0[1])}


def predict(X: np.ndarray, k: float, m: float) -> np.ndarray:
    """LRV = k * CT^m.

    X: (n, 1) — column [CT].
    """
    CT = np.asarray(X[:, 0], dtype=float)
    return _lrv(CT, k, m)
