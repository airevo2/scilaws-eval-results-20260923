"""Linear distance-decay of community similarity.

Soininen et al. (2007) enumerate three regression forms used in the
distance-decay literature. This is the linear-linear form (their §3a):

    Sim(d) = alpha + beta * d

with d the released environmental distance `env_dist`. Both parameters
are study-specific; the linear *form* is the scientific claim and is
invariant across clusters. The form can predict negative similarity at
large distances (Soininen et al. 2007 flag this as a drawback of the
linear-linear form versus the log-linear / log-log forms); `predict`
does not clip.

LAW_CONSTANTS  — none.
OTHER_CONSTANTS— none.
LOCAL_FITTABLE — alpha, beta. Fitted by closed-form OLS of Sim_sor on
                 env_dist (init = None — exact linear regression).
"""

import numpy as np

USED_INPUTS = ["env_dist"]
PAPER_REF = "summary_formula_dataset_soininen_2007.md"
EQUATION_LOC = (
    "Soininen et al. (2007) Methods, PDF p. 3, linear-linear form: "
    "Sim(d) = beta*d + alpha."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "alpha": {"init": None},   # closed-form OLS
    "beta":  {"init": None},
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS of Sim_sor on env_dist."""
    d = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)
    A = np.column_stack([np.ones_like(d), d])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"alpha": float(coef[0]), "beta": float(coef[1])}


def predict(X: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Sim_sor = alpha + beta * env_dist.

    X: (n, 1) — column 0 is env_dist.
    """
    d = np.asarray(X[:, 0], dtype=float)
    return alpha + beta * d
