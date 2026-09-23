"""He et al. (2011) double-exponential capacity fade.

He, Williard, Osterman & Pecht (2011), J. Power Sources 196(23):
10314-10321, Eq. 1 (PDF p. 3): a two-rate phenomenological model for
Li-ion capacity fade,

    cap_dischg_mAh = a * exp(b * k) + c * exp(d * k)

with cycle_index k. The first exponential captures rapid early-cycle
fade (e.g. SEI formation, initial cyclable-Li loss); the second a
slower long-term decay. All four scalars are per-cell fits — no
universal numerical constants in the formula form.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster, computed by fit() via nonlinear LS
-----------------------------------------------------------------
- a, b : first-exponential amplitude and rate.
- c, d : second-exponential amplitude and rate.
init = None on all four: fit() builds its own data-derived start
  (asymptote ~ first/last capacity; rate ~ log decay over the window).
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["cycle_index"]
PAPER_REF = "summary_formula+dataset_he_2011.md"
EQUATION_LOC = (
    "He, Williard, Osterman & Pecht (2011) Eq. 1, PDF p. 3 — "
    "Q = a*exp(b*k) + c*exp(d*k)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "a": {"init": None}, "b": {"init": None},
    "c": {"init": None}, "d": {"init": None},
}


def _model(k, a, b, c, d):
    return a * np.exp(np.clip(b * k, -50.0, 50.0)) + c * np.exp(np.clip(d * k, -50.0, 50.0))


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear least-squares fit of a, b, c, d.

    Start: split the fade into "slow tail" (the late-cycle near-asymptote
    captured by c·exp(d·k) with d slightly negative) and "fast head"
    (the gap between first cycle and the tail, captured by a·exp(b·k)
    decaying quickly). Bounds keep rates in a sane range.
    """
    k = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    y_start = float(y[np.argmin(k)]) if len(y) else 1.0
    y_end   = float(y[np.argmax(k)]) if len(y) else 1.0
    k_max   = float(np.max(k)) if len(k) else 1.0

    c0 = y_end if y_end > 0 else max(abs(y_start) * 0.5, 1.0)
    a0 = (y_start - y_end) if y_start > y_end else max(abs(y_start) * 0.1, 0.1)
    d0 = -1e-4
    b0 = -1e-2
    p0 = [a0, b0, c0, d0]

    def residual(p):
        a, b, c, d = p
        return _model(k, a, b, c, d) - y

    try:
        sol = least_squares(residual, p0,
                            bounds=([-1e6, -1.0, -1e6, -1.0],
                                    [ 1e6,  1.0,  1e6,  1.0]),
                            method="trf", max_nfev=4000)
        return {"a": float(sol.x[0]), "b": float(sol.x[1]),
                "c": float(sol.x[2]), "d": float(sol.x[3])}
    except Exception:                              # noqa: BLE001
        return {"a": a0, "b": b0, "c": c0, "d": d0}


def predict(X: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """cap = a * exp(b * k) + c * exp(d * k).

    X: (n, 1) — column 0 is cycle_index.
    """
    return _model(np.asarray(X[:, 0], dtype=float), a, b, c, d)
