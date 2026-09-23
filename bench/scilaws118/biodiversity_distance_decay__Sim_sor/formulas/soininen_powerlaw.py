"""Power-law (log-log) distance-decay of community similarity.

Soininen et al. (2007) enumerate three regression forms used in the
distance-decay literature. This is the log-log form (their §3c):

    ln[Sim(d)] = beta * ln(d) + alpha
    Sim(d)     = exp(alpha) * d ** beta

a power law in the separation d (here the released environmental
distance `env_dist`). Both parameters are study-specific; the power-law
*form* is invariant across clusters and is the scientific claim. The
form is undefined at d = 0; the released bin-mean env_dist values are
strictly positive.

LAW_CONSTANTS  — none.
OTHER_CONSTANTS— none.
LOCAL_FITTABLE — alpha, beta. Fitted by closed-form OLS of ln(S) on
                 ln(d) (init = None — exact log-log linear regression).
"""

import numpy as np

USED_INPUTS = ["env_dist"]
PAPER_REF = "summary_formula_dataset_soininen_2007.md"
EQUATION_LOC = (
    "Soininen et al. (2007) Methods, PDF p. 3, log-log form: "
    "ln[Sim(d)] = beta*ln(d) + alpha  =>  Sim(d) = exp(alpha) * d**beta."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "alpha": {"init": None},   # closed-form OLS in log-log space
    "beta":  {"init": None},
}

_SIM_FLOOR = 1e-6   # similarity floor so ln(S) is finite
_D_FLOOR = 1e-6     # distance floor so ln(d) is finite


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS of ln(Sim_sor) on ln(env_dist)."""
    d = np.clip(np.asarray(X_fit[:, 0], dtype=float), _D_FLOOR, None)
    y = np.clip(np.asarray(y_fit, dtype=float), _SIM_FLOOR, None)
    A = np.column_stack([np.ones_like(d), np.log(d)])
    coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    return {"alpha": float(coef[0]), "beta": float(coef[1])}


def predict(X: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Sim_sor = exp(alpha) * env_dist ** beta.

    X: (n, 1) — column 0 is env_dist.
    """
    d = np.clip(np.asarray(X[:, 0], dtype=float), _D_FLOOR, None)
    return np.exp(alpha) * np.power(d, beta)
