"""Rezende & Bozinovic (2019) mechanistic thermal performance curve.

Rezende & Bozinovic, *Phil. Trans. R. Soc. B* 374:20180549 (2019), Eq. 2.3
(PDF p. 2). A four-parameter TPC written as the product of a thermodynamic
(Q10) rate component (Eq. 2.1) and a piecewise quadratic high-temperature
denaturation component (Eq. 2.2):

    performance(T) = C * exp(T * ln(Q10) / 10) * b(T)
    b(T) = 1                          if T <= T_th
    b(T) = 1 - d * (T - T_th) ** 2     if T  > T_th

The exponential term is the Q10 rule: the rate multiplies by Q10 for every
10 degrees C of warming. Below the thermal threshold T_th the curve is pure
exponential rise; above T_th a quadratic denaturation penalty bends it down,
producing the peak and the steep high-temperature collapse.

Per-curve parameters are (C, Q10, T_th, d), reported per experiment in
Rezende 2019 Table 1 (PDF p. 5) with cross-trait medians: photosynthesis
Q10 ~ 1.81, lizards ~ 2.36, insects ~ 3.79; T_th medians 16-21 degrees C;
d medians 0.0013-0.0051 degrees C^-2.

The structural constants 1 (cold-side intercept of b), 10 (the Q10
denominator) and 2 (the quadratic exponent) are part of the fixed form and
are never refit (Rezende 2019 Eqs. 2.1-2.2, PDF p. 2).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The mechanistic Q10 x denaturation form is the scientific claim; its
four parameters are per-experiment fits with no universal numerical defaults
(the dataset here spans 159 distinct trait units, so C and Q10 cannot share
a universal value across clusters).

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The 1, 10 and 2 are part of the formula's fixed algebraic form.

LOCAL_FITTABLE — per-cluster, computed by fit() via bounded nonlinear LS
-------------------------------------------------------------------------
- C    : performance scale at T = 0 degrees C (units = the cluster's own).
- Q10  : the Q10 temperature coefficient (rate fold-change per 10 deg C).
- T_th : thermal threshold (degrees C) above which denaturation begins.
- d    : quadratic denaturation coefficient (degrees C^-2).

init = None on all four: fit() builds its own deterministic, data-derived
start. The exponential overflows easily, so fit() bounds all four parameters
and wraps the solve in try/except with a safe fallback.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["temperature"]
PAPER_REF = "summary_formula+dataset_rezende_2019.md"
EQUATION_LOC = (
    "Rezende & Bozinovic (2019) Eq. 2.3, PDF p. 2 (with Eqs. 2.1-2.2; "
    "Table 1, PDF p. 5 for fitted parameter ranges)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "C":    {"init": None},
    "Q10":  {"init": None},
    "T_th": {"init": None},
    "d":    {"init": None},
}

# bounds — keep the exp() argument finite over the data temperature range.
_LO = [-np.inf, 1.001,   -50.0,   0.0]
_HI = [ np.inf, 50.0,    150.0,   10.0]


def _rezende(T, C, Q10, T_th, d):
    arg = T * np.log(Q10) / 10.0
    arg = np.clip(arg, -700.0, 700.0)            # guard exp overflow
    thermo = C * np.exp(arg)
    denat = np.where(T <= T_th, 1.0, 1.0 - d * (T - T_th) ** 2)
    return thermo * denat


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the Rezende form.

    Deterministic, data-derived start:
      T_th0 = temperature at the largest observed performance (peak),
      Q10_0 = 2.0 (a generic biological rate coefficient),
      C0    = performance at the coldest measurement, back-projected to 0 C,
      d0    = a small positive denaturation coefficient.
    All four parameters are bounded; the solve is wrapped so an overflow or
    non-convergence falls back to the start.
    """
    T = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    i_peak = int(np.argmax(y))
    T_th0  = float(T[i_peak])
    Q10_0  = 2.0
    # back-project the coldest observed performance to T = 0.
    i_cold = int(np.argmin(T))
    y_cold = float(y[i_cold]) if y[i_cold] > 0 else float(np.max(y))
    C0 = y_cold / np.exp(np.clip(T[i_cold] * np.log(Q10_0) / 10.0, -700, 700))
    if not np.isfinite(C0) or C0 == 0:
        C0 = float(np.max(np.abs(y))) or 1.0
    span = float(T.max() - T.min()) or 1.0
    d0 = 1.0 / (span ** 2)

    def residual(p):
        return _rezende(T, p[0], p[1], p[2], p[3]) - y

    p0 = [C0, Q10_0, T_th0, d0]
    # clamp the start inside the bounds
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, _LO, _HI)]
    try:
        sol = least_squares(residual, p0, bounds=(_LO, _HI),
                            method="trf", max_nfev=4000)
        C, Q10, T_th, d = sol.x
        if not np.all(np.isfinite([C, Q10, T_th, d])):
            raise RuntimeError("non-finite fit")
        return {"C": float(C), "Q10": float(Q10),
                "T_th": float(T_th), "d": float(d)}
    except Exception:                              # noqa: BLE001 — fall back to the start
        return {"C": float(p0[0]), "Q10": float(p0[1]),
                "T_th": float(p0[2]), "d": float(p0[3])}


def predict(X: np.ndarray, C: float, Q10: float, T_th: float, d: float) -> np.ndarray:
    """performance = C * exp(T*ln(Q10)/10) * b(T), b piecewise about T_th.

    X: (n, 1) — column 0 is temperature (degrees C).
    """
    T = np.asarray(X[:, 0], dtype=float)
    return _rezende(T, C, Q10, T_th, d)
