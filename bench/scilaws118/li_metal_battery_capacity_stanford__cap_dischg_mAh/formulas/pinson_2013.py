"""Pinson-Bazant (2013) sqrt-time capacity fade.

Pinson & Bazant (2013), J. Electrochem. Soc. 160(2):A243-A250, Eq. 4
(PDF p. 7): the large-time asymptote of SEI thickness on a graphite
anode follows sqrt(t). Substituting cycle_index k for time t (with the
sqrt(tau) per-cycle time factor absorbed into alpha) and rewriting as
a capacity expression gives

    cap_dischg_mAh = nominal_capacity_mAh - alpha * sqrt(cycle_index) + beta

alpha (rate of sqrt-fade per sqrt(cycle), aggregates the paper's
sqrt(2*c*m*D*tau/rho)*A_elec) and beta (offset capturing deviation of
the first cycle from rated capacity) are per-cell fits. Closed-form
linear OLS on the design [-sqrt(k), 1] vs (cap - Q0).

Chemistry caveat: Pinson-Bazant derive sqrt(t) for SEI growth on
graphite Li-ion anodes (PDF §1-2); the Uppaluri 2025 dataset is
Li-metal / NMC-811. The summary explicitly flags this chemistry
mismatch — the formula is included as the standard sqrt-time baseline
to probe whether SR methods recover this form on Li-metal data.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster, computed by fit() via closed-form OLS
-------------------------------------------------------------------
- alpha : sqrt-fade rate per sqrt(cycle).
- beta  : offset (cap_0 - Q_0).
init = None on both: linear OLS is deterministic.
"""

import numpy as np

USED_INPUTS = ["cycle_index", "nominal_capacity_mAh"]
PAPER_REF = "summary_formula_pinson_2013.md"
EQUATION_LOC = (
    "Pinson & Bazant (2013) Eq. 4, PDF p. 7 — s(t) = sqrt(2*c*m*D*t/rho) - D/k_r "
    "(large-t SEI asymptote); cap = Q_0 - alpha*sqrt(k) + beta after substituting "
    "k for t and absorbing material/rate constants into alpha."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "alpha": {"init": None},
    "beta":  {"init": None},
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS of (cap - Q_0) on [-sqrt(k), 1]."""
    k  = np.asarray(X_fit[:, 0], dtype=float)
    q0 = np.asarray(X_fit[:, 1], dtype=float)
    y = np.asarray(y_fit, dtype=float) - q0
    A = np.column_stack([-np.sqrt(np.clip(k, 0.0, None)), np.ones_like(k)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"alpha": float(coef[0]), "beta": float(coef[1])}


def predict(X: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """cap = Q_0 - alpha * sqrt(k) + beta.

    X: (n, 2) — columns cycle_index, nominal_capacity_mAh.
    """
    k  = np.asarray(X[:, 0], dtype=float)
    q0 = np.asarray(X[:, 1], dtype=float)
    return q0 - alpha * np.sqrt(np.clip(k, 0.0, None)) + beta
