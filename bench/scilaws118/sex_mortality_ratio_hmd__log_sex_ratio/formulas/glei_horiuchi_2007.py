"""Glei & Horiuchi (2007) Gompertz-implied linear sex-mortality ratio.

Glei & Horiuchi (2007), Population Studies 61(2):141-159, DOI
10.1080/00324720701331433. Sex-specific adult mortality follows the
Gompertz law (Eq. 2, PDF p. 3):

    mu_k(x, t) = a_k(t) * exp(b_k * x),   k in {male, female}

Taking the ratio alpha(x) = mu_M(x) / mu_F(x) and taking the log:

    ln alpha(x) = ln(a_M / a_F) + (b_M - b_F) * x  =  A + B * x

— a straight line in age, with country-specific intercept A (log-level
sex differential) and slope B (Gompertz-slope difference). Glei &
Horiuchi (2007) Table 2 (PDF p. 11) reports b_F in [0.090, 0.111] and
b_M in [0.071, 0.097] across 29 countries (1975-1979) — implying
B = b_M - b_F < 0 in all of them: ln alpha falls with age, the
familiar "male disadvantage attenuates as both sexes age."

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The linear-in-age form is the scientific claim; A and B are
per-country fits.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster, computed by fit() via closed-form OLS
-------------------------------------------------------------------
- A : ln(a_M / a_F), country-specific log-level sex differential.
- B : b_M - b_F, country-specific Gompertz-slope difference (negative
      for modern high-income populations).
init = None on both: linear OLS is deterministic.

Caveat: `year` is not in this formula — the paper analyses a single
period 1975-1979 and reports cross-sectional values. The benchmark
pools all of a cluster's years; the fit captures the time-averaged
linear age pattern.
"""

import numpy as np

USED_INPUTS = ["age"]
PAPER_REF = "summary_supporting_glei_2007.md"
EQUATION_LOC = (
    "Glei & Horiuchi (2007) Eq. 2, PDF p. 3 (Gompertz mu = a*exp(b*x)); "
    "linear form ln alpha = A + B*x in §'Why it appears in this source group'."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "A": {"init": None},
    "B": {"init": None},
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS of log_sex_ratio on (1, age)."""
    age = np.asarray(X_fit[:, 0], dtype=float)
    A = np.column_stack([np.ones_like(age), age])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y_fit, dtype=float), rcond=None)
    return {"A": float(coef[0]), "B": float(coef[1])}


def predict(X: np.ndarray, A: float, B: float) -> np.ndarray:
    """log_sex_ratio = A + B * age.

    X: (n, 1) — column 0 is age.
    """
    age = np.asarray(X[:, 0], dtype=float)
    return A + B * age
