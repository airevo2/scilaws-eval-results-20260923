"""Exponential distance-decay of community similarity.

Nekola & White (1999) formalised the *distance decay of similarity*: the
compositional similarity of two ecological communities declines as the
separation between them grows. They model it with an exponential decay,

    S = S0 * exp(-c * d)        (Nekola & White 1999, Discussion, p. 9)

which linearises in log space to  ln S = ln S0 - c * d. Graco-Roza et
al. (2022) — the source meta-analysis of this benchmark — fit the same
point-prediction shape per dataset as a log-link GLM,
`log(S) = intercept + slope * d` (PDF p. 10, §2.4). Writing
alpha = ln S0 and beta = -c the two collapse to one form:

    S = exp(alpha + beta * d)

with d the released environmental distance `env_dist`. Both parameters
are study-specific and fitted per cluster; the exponential *form* is the
scientific claim and is invariant across clusters.

LAW_CONSTANTS  — none. The exponential form is the claim; its two
                 parameters are cluster-specific.
OTHER_CONSTANTS— none.
LOCAL_FITTABLE — alpha, beta. Fitted by closed-form OLS of ln(S) on d
                 (init = None — log-space linear regression is exact and
                 deterministic, no multi-start needed).
"""

import numpy as np

USED_INPUTS = ["env_dist"]
PAPER_REF = "summary_formula_nekola_1999.md"
EQUATION_LOC = (
    "Nekola & White (1999) Discussion, PDF p. 9 — exponential decay "
    "S = S0*exp(-c*d), semi-log form ln S = ln S0 - c*d; same shape as "
    "Graco-Roza et al. (2022) log-link GLM log(S) = intercept + slope*d "
    "(PDF p. 10, §2.4). Here S = exp(alpha + beta*env_dist)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "alpha": {"init": None},   # closed-form OLS in log space
    "beta":  {"init": None},
}

# similarity floor used so ln() is finite when fitting clusters that
# contain a zero bin-mean (structural, not a tuned constant)
_SIM_FLOOR = 1e-6


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS of ln(Sim_sor) on env_dist."""
    d = np.asarray(X_fit[:, 0], dtype=float)
    y = np.clip(np.asarray(y_fit, dtype=float), _SIM_FLOOR, None)
    A = np.column_stack([np.ones_like(d), d])
    coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    return {"alpha": float(coef[0]), "beta": float(coef[1])}


def predict(X: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Sim_sor = exp(alpha + beta * env_dist).

    X: (n, 1) — column 0 is env_dist.
    """
    d = np.asarray(X[:, 0], dtype=float)
    return np.exp(alpha + beta * d)
