"""Universal Thermal Performance Curve (UTPC) — physical-units form.

Arnoldi et al. (2025), "A universal thermal performance curve arises in
biology and ecology", PNAS, DOI:10.1073/pnas.2513099122. The paper's central
claim is that, after a per-experiment affine rescaling of temperature and a
multiplicative rescaling of performance, every thermal performance curve
collapses onto a single parameter-free attractor shape (Eq. 10, PDF p. 3):

    y(x) = exp(x) * (1 - x)

with the rescaling (Eq. 11, PDF p. 3):

    x = (T - Topt) / (Tc - Topt)
    y = performance / Pfmax

Undoing the rescaling gives the physical-units form actually fitted per
experiment (Eq. 15, PDF p. 4 — the three values fit per experiment by
L-BFGS-B in the paper):

    performance(T) = Pfmax * exp(x) * (1 - x),   x = (T - Topt) / (Tc - Topt)

Shape:
- At T = Topt, x = 0, performance = Pfmax (the peak).
- At T = Tc,   x = 1, performance = 0 (the critical temperature).
- For T > Tc the algebraic value is negative; the paper notes performance
  has collapsed beyond Tc. Predictions are left unclipped so fit metrics
  reflect the form exactly as published.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The UTPC shape `exp(x)*(1-x)` IS the scientific claim; the `1` inside
`(1 - x)` and the `e` of `exp` are structural features of the attractor, not
tunable constants. The three per-experiment quantities Pfmax, Topt, Tc are
the LOCAL_FITTABLE parameters.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The `1` and `e` are part of the formula's fixed algebraic form.

LOCAL_FITTABLE — per-cluster, computed by fit() via nonlinear least squares
---------------------------------------------------------------------------
- Pfmax : peak performance of this experiment (units = the cluster's own).
- Topt  : optimal temperature (degrees C) — where performance peaks.
- Tc    : critical temperature (degrees C) — where performance reaches zero.

init = None on all three: fit() builds its own deterministic, data-derived
start (Topt = argmax-temperature, Pfmax = max performance, Tc = Topt + a
fraction of the observed temperature span).
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["temperature"]
PAPER_REF = "summary_formula_dataset_arnoldi_2025.md"
EQUATION_LOC = (
    "Arnoldi et al. (2025) Eq. 10, PDF p. 3 — universal attractor "
    "y = exp(x)*(1-x); physical-units form Eq. 15, PDF p. 4, with "
    "x = (T - Topt)/(Tc - Topt) and performance = Pfmax * y."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "Pfmax": {"init": None},
    "Topt":  {"init": None},
    "Tc":    {"init": None},
}


def _utpc(T, Pfmax, Topt, Tc):
    denom = Tc - Topt
    if abs(denom) < 1e-9:
        denom = 1e-9 if denom >= 0 else -1e-9
    x = (T - Topt) / denom
    return Pfmax * np.exp(x) * (1.0 - x)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear least-squares fit of the UTPC physical-units form.

    Deterministic, data-derived start:
      Topt0  = temperature at the largest observed performance,
      Pfmax0 = the largest observed performance,
      Tc0    = Topt0 + half the observed temperature span (Tc lies above Topt).
    """
    T = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    Topt0  = float(T[int(np.argmax(y))])
    Pfmax0 = float(np.max(y))
    if Pfmax0 <= 0:
        Pfmax0 = float(np.max(np.abs(y))) or 1.0
    span = float(T.max() - T.min())
    Tc0 = Topt0 + max(0.5 * span, 1.0)

    def residual(p):
        return _utpc(T, p[0], p[1], p[2]) - y

    p0 = [Pfmax0, Topt0, Tc0]
    try:
        sol = least_squares(residual, p0, method="lm", max_nfev=4000)
        Pfmax, Topt, Tc = sol.x
        if not np.all(np.isfinite([Pfmax, Topt, Tc])) or abs(Tc - Topt) < 1e-9:
            raise RuntimeError("degenerate fit")
        return {"Pfmax": float(Pfmax), "Topt": float(Topt), "Tc": float(Tc)}
    except Exception:                              # noqa: BLE001 — fall back to the start
        return {"Pfmax": Pfmax0, "Topt": Topt0, "Tc": Tc0}


def predict(X: np.ndarray, Pfmax: float, Topt: float, Tc: float) -> np.ndarray:
    """performance = Pfmax * exp(x) * (1 - x), x = (T - Topt)/(Tc - Topt).

    X: (n, 1) — column 0 is temperature (degrees C).
    """
    T = np.asarray(X[:, 0], dtype=float)
    return _utpc(T, Pfmax, Topt, Tc)
