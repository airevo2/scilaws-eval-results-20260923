"""Gompertz-Makeham adult mortality (log of a Gompertz term plus a constant).

Makeham (1860) observed that the logarithms of survival probabilities do
NOT form a single geometric progression as Gompertz (1825) requires — the
real curve rises faster-than-geometric through middle adult ages. His fix:
add an age-independent constant to the Gompertz term, giving the force of
mortality (Makeham 1860, PDF p. 303; force-of-mortality form in Makeham
1867):

    m(x) = A + B * exp(beta * x)   ==>   log m(x) = log(A + B * exp(beta*x))

- A             : the Makeham constant — background, age-independent mortality.
- B * exp(beta*x): the Gompertz senescent term (Gompertz 1825), unchanged.

This is NOT linear in age: log of a sum bends the curve down at younger
adult ages (where A is non-negligible) and approaches the straight Gompertz
line log B + beta*x at older ages. A -> 0 recovers pure Gompertz exactly.

All three parameters are country-specific and fitted per cluster. Unlike
the closed-form OLS baselines, the fit is nonlinear least squares; the
start is derived deterministically from the data (OLS on the log scale),
so no init seeds are needed.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The Gompertz-Makeham form is the scientific claim; A, B, beta are
country-specific fits.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster, computed by fit() via nonlinear least squares
---------------------------------------------------------------------------
- A    : Makeham constant, age-independent mortality level (>= 0).
- B    : Gompertz senescent level.
- beta : Gompertz senescence slope = ln c, log-mortality growth per year.

init = None on all three: fit() builds its own data-derived start.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["age"]
PAPER_REF = "summary_formula_makeham_1860.md"
EQUATION_LOC = (
    "Makeham (1860) Part I, PDF p. 303 — modification of Gompertz's law by an "
    "additional constant; force-of-mortality form mu(x) = A + B*c^x "
    "(c = exp(beta)), with log m(x) = log(A + B*exp(beta*age)) for adult ages."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "A":    {"init": None},
    "B":    {"init": None},
    "beta": {"init": None},
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear least-squares fit of log(A + B*exp(beta*age)) to log_m_adult.

    Deterministic start: OLS of y on age gives (beta, log B); the Makeham
    constant A starts at a small positive fraction of the smallest rate.
    """
    age = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # OLS start (the pure-Gompertz line): y ~ logB + beta*age.
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(age), age]), y, rcond=None)
    logB0, beta0 = float(coef[0]), float(coef[1])
    A0 = 0.5 * float(np.exp(np.min(y)))            # below the smallest observed m

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
    """log_m_adult = log(A + B * exp(beta * age)).

    X: (n, 1) — column 0 is age.
    """
    age = np.asarray(X[:, 0], dtype=float)
    return np.log(A + B * np.exp(beta * age))
