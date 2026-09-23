"""Benzekry 2014 — Generalized logistic tumor growth model.

Benzekry S, Lamont C, Beheshti A, Tracz A, Ebos JML, Hlatky L, Hahnfeldt P.
(2014). Classical Mathematical Models for Description and Prediction of
Experimental Tumor Growth. *PLoS Computational Biology*, 10(8): e1003800.
DOI:10.1371/journal.pcbi.1003800.

Eq. 3 (PDF p. 3):

    dV/dt = a * V * (1 - (V/K)^n),   V(t=0) = 1 mm³

Explicit solution with V0 = 1 mm³ (PDF p. 3):

    V(t) = V0 * K / [V0^n + (K^n - V0^n) * exp(-a*n*t)]^(1/n)

Special cases (paper-stated, PDF p. 3):
  - n=1  => logistic (Eq. 2)
  - n->0 => Gompertz (Eq. 4)

Shape parameter n controls the steepness of the sigmoid; fitted values are
near 0 in practice (Gompertz-like; median n ~ 1.4e-4 for LLC, Table 3).

Three free per-animal parameters: a, K, n.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
(empty.)

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
- V0: 1.0 mm³ — assumed (given) initial condition, NOT a fitted coefficient.
  Benzekry 2014 PDF p. 3 / L255 calls 1 mm³ "a reasonable approximation for
  V(t=0)" (~10^6 injected cells) — an invariant-but-GIVEN injection volume, so
  it is filed under OTHER (structural/given) rather than LAW (fitted-law).

LOCAL_FITTABLE — per-cluster, fitted by fit()
----------------------------------------------
- a : proliferation rate coefficient (day^-1, > 0).
- K : asymptotic carrying capacity (mm³, > 0).
- n : shape exponent controlling sigmoid steepness (dimensionless, > 0).
  init = None on all: fit() derives data-driven starts.

Note: n is poorly identifiable from short time series (Benzekry 2014 p. 9,
Table 3). The fitter uses bounds [1e-6, 5] for n to prevent numerical issues
in the power expressions.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["time_day"]
PAPER_REF = "summary_formula_dataset_benzekry_2014.md"
EQUATION_LOC = (
    "Benzekry 2014 Eq. 3, PDF p. 3 — generalized logistic: "
    "dV/dt = a*V*(1-(V/K)^n), V(0)=1; "
    "solution V(t) = V0*K / [V0^n + (K^n-V0^n)*exp(-a*n*t)]^(1/n)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    "V0": 1.0,   # mm^3; assumed initial condition, Benzekry 2014 PDF p. 3
}
LOCAL_FITTABLE = {
    "a": {"init": None},
    "K": {"init": None},
    "n": {"init": None},
}


def _predict_core(t: np.ndarray, V0: float, a: float, K: float, n: float) -> np.ndarray:
    # V(t) = V0*K / [V0^n + (K^n - V0^n)*exp(-a*n*t)]^(1/n)
    V0n = V0 ** n
    Kn  = K  ** n
    inner = V0n + (Kn - V0n) * np.exp(-a * n * t)
    inner = np.maximum(inner, 1e-30)
    return V0 * K / (inner ** (1.0 / n))


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear LS fit of the generalized logistic model.

    Init: a_0 from log-linear slope; K_0 = 2*max(y); n_0 = 0.5
    (intermediate between logistic and Gompertz-like).
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    K_0 = float(np.max(y)) * 2.0
    K_0 = max(K_0, 2.0)

    log_y = np.log(np.clip(y, 1e-6, None))
    if t.std() > 1e-9:
        a_0 = float(np.polyfit(t, log_y, 1)[0])
    else:
        a_0 = 0.3
    a_0 = max(1e-4, min(a_0, 5.0))
    n_0 = 0.5

    def residual(p):
        return _predict_core(t, V0, p[0], p[1], p[2]) - y

    try:
        sol = least_squares(residual, [a_0, K_0, n_0],
                            bounds=([1e-6, 1.01, 1e-6], [20.0, 1e6, 5.0]),
                            method="trf", max_nfev=6000)
        return {"a": float(sol.x[0]), "K": float(sol.x[1]),
                "n": float(sol.x[2])}
    except Exception:                               # noqa: BLE001
        return {"a": a_0, "K": K_0, "n": n_0}


def predict(X: np.ndarray, a: float, K: float, n: float) -> np.ndarray:
    """Generalized logistic tumor volume.

    X: (n_rows, 1) — column [time_day]. V0 read from OTHER_CONSTANTS.
    """
    V0 = OTHER_CONSTANTS["V0"]
    t = np.asarray(X[:, 0], dtype=float)
    return _predict_core(t, V0, a, K, n)
