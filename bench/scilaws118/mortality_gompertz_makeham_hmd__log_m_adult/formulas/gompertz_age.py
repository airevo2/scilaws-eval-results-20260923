"""Gompertz log-linear adult mortality.

For adult ages (≈30+) the central death rate m(x) is empirically well
described by the Gompertz law (Gompertz 1825; restated as the third term
of Heligman & Pollard 1980 Eq. 1, PDF p. 49 §2.1):

    m(x) ≈ G * H^x      ⇒      log m(x) = alpha + beta * x

with alpha = ln G and beta = ln H. The two parameters are country-specific
and fitted per cluster by linear OLS (closed-form, no init seeds needed).

Why only two parameters: the infant-mortality and accident-hump terms of
the full HP 1980 form are negligible in the age range 30-95 covered by
this benchmark (HP 1980 §2.3, PDF p. 50). What remains is pure Gompertz.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The Gompertz form itself is the scientific claim; its two
parameters (alpha, beta) are country-specific and fitted per cluster.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster, computed by fit() via closed-form OLS
-------------------------------------------------------------------
- alpha : ln G — log-mortality intercept at age 0 for the country.
- beta  : ln H — Gompertz senescence slope (rate of log-mortality
                  increase per year of age).

init = None on both: linear OLS is deterministic, no multi-start.
"""

import numpy as np

USED_INPUTS = ["age"]
PAPER_REF = "summary_formula_gompertz_1825.md"
EQUATION_LOC = (
    "Gompertz (1825) Arts. 1-4, PDF pp. 514-517 — survivors fall in geometric "
    "progression with age, i.e. mu(x) = B*c^x, giving log m = alpha + beta*age "
    "for the adult range (alpha = ln B, beta = ln c)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "alpha": {"init": None},   # closed-form OLS
    "beta":  {"init": None},
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS of log_m_adult vs age."""
    age = np.asarray(X_fit[:, 0], dtype=float)
    A = np.column_stack([np.ones_like(age), age])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y_fit, dtype=float), rcond=None)
    return {"alpha": float(coef[0]), "beta": float(coef[1])}


def predict(X: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """log_m_adult = alpha + beta * age.

    X: (n, 1) — column 0 is age.
    """
    age = np.asarray(X[:, 0], dtype=float)
    return alpha + beta * age
