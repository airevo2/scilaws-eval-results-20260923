"""Gompertz-Makeham law — adult senescence plus a constant background.

Makeham (1860) modified Gompertz's law by adding an age-independent
constant, giving the force of mortality

    m(x) = A + B * exp(beta * x)   ==>   log_m_full = log(A + B*exp(beta*x))

- A              : the Makeham constant — background, age-independent mortality.
- B * exp(beta*x): the Gompertz senescent term.

This is the WEAK rung of the full-age reference landscape. The
Gompertz-Makeham law has only two regimes — a constant floor and an
exponential senescent rise — so on the benchmark's full 0-95 age range it
cannot represent the steep childhood-mortality decline (ages 0-10) or the
young-adult accident hump (ages 15-30). It is expected to fit those
regimes poorly; what it measures is how much of the full curve is
captured by senescence + background alone, before the Heligman-Pollard
childhood and accident-hump terms are added.

All three parameters are country-specific and fitted per cluster by
nonlinear least squares. The start is data-derived (a Gompertz OLS on
adult ages seeds B and beta), so no init seeds are needed.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The Gompertz-Makeham form is the scientific claim; A, B, beta are
per-cluster fits.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster, computed by fit() via nonlinear least squares
---------------------------------------------------------------------------
- A    : Makeham constant, age-independent mortality level (>= 0).
- B    : Gompertz senescent level.
- beta : Gompertz senescence slope (log-mortality growth per year of age).
init = None on all three: fit() builds its own data-derived start.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["age"]
PAPER_REF = "summary_formula_makeham_1860.md"
EQUATION_LOC = (
    "Makeham (1860) Part I, PDF p. 303 — modification of Gompertz's law by an "
    "additional constant; force-of-mortality form m(x) = A + B*exp(beta*age), "
    "with log_m_full = log(A + B*exp(beta*age))."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "A":    {"init": None},
    "B":    {"init": None},
    "beta": {"init": None},
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear least-squares fit of log(A + B*exp(beta*age)) to log_m_full.

    Deterministic start: a Gompertz OLS on adult ages (>=30, where the
    senescent term dominates) seeds beta and log B; the Makeham constant A
    starts at a small positive fraction of the smallest observed rate.
    """
    age = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    adult = age >= 30
    if adult.sum() >= 5:
        coef, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(adult.sum()), age[adult]]), y[adult], rcond=None)
    else:
        coef, *_ = np.linalg.lstsq(
            np.column_stack([np.ones_like(age), age]), y, rcond=None)
    logB0, beta0 = float(coef[0]), float(coef[1])
    A0 = 0.5 * float(np.exp(np.min(y)))

    def residual(p):
        A, logB, beta = p
        return np.log(A + np.exp(logB + beta * age)) - y

    p0 = [A0, logB0, beta0]
    lo = [0.0, -60.0, 0.0]
    hi = [np.inf, 20.0, 1.0]
    try:
        sol = least_squares(residual, p0, bounds=(lo, hi), method="trf", max_nfev=2000)
        A, logB, beta = sol.x
        return {"A": float(A), "B": float(np.exp(logB)), "beta": float(beta)}
    except Exception:                              # noqa: BLE001 — fall back to pure Gompertz
        return {"A": 0.0, "B": float(np.exp(logB0)), "beta": beta0}


def predict(X: np.ndarray, A: float, B: float, beta: float) -> np.ndarray:
    """log_m_full = log(A + B * exp(beta * age)).

    X: (n, 1) — column 0 is age.
    """
    age = np.asarray(X[:, 0], dtype=float)
    return np.log(A + B * np.exp(beta * age))
