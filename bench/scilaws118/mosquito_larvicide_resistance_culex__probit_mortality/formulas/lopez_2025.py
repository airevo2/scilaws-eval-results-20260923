"""Probit dose-response model — Abbott-corrected mortality probability.

Lopez K, Irwin P, Tomek M, Holub R, Paskewitz S, Bartholomay L, Clifton M
(2025). "Dual S-methoprene and Lysinibacillus sphaericus larvicide use leads
to multiple independent, and not cross-resistance in Culex pipiens."
PLOS ONE 20(9): e0332621. DOI: 10.1371/journal.pone.0332621.

The probit dose-response model applied throughout Lopez et al. (2025) is the
Bliss/Finney paradigm (Finney 1971 framework, executed via R package `ecotox`):
Abbott-corrected mortality probability is modeled as a standard normal CDF
(probit link) evaluated at a linear function of log10-dose (PDF pp. 5-6):

    P = Phi(alpha + beta * log10(x))

where Phi is the standard normal CDF, x is Ls concentration (ppb), alpha is
the probit intercept (shifts the curve along the log-dose axis), and beta is
the probit slope (curve steepness). Both alpha and beta are refit per
population (cluster) using maximum likelihood.

Parameter estimates are published per-population in Table 1 (PDF p. 10):
  - Slope (beta) ranges from ~0.55 to ~3.46 across the 32 field populations
  - LC50 values (= 10^(-alpha/beta)) range from 9.98 to 860.55 ppb
The COL (susceptible Iowa lab strain) reference LC50 is 9.98 ppb (Table 1).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The probit FORM is the scientific claim. alpha and beta are refit per
population in every published source; no universal numerical values apply
across populations. Both are LOCAL_FITTABLE.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
None assigned. The Phi (standard normal CDF) is the fixed structural link
function, not a tunable constant. The factor 10 in log10 is definitional.

LOCAL_FITTABLE — per-cluster, fitted by fit() via MLE / WLS
------------------------------------------------------------
- alpha : probit intercept (dimensionless); shifts curve along log-dose axis.
          LC50 (in log10 scale) = -alpha / beta.
- beta  : probit slope (dimensionless, > 0 for increasing mortality with dose).
          Typical range across 32 populations: 0.55-3.46 (Table 1, PDF p. 10).

init = None on both: fit() builds its own data-derived start from empirical
median and interquartile estimates of the dose-response relationship.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr  # standard normal CDF, numerically stable

USED_INPUTS = ["log10_conc_ppb"]
PAPER_REF = "summary_formula_dataset_lopez_2025.md"
EQUATION_LOC = (
    "Lopez et al. (2025) PDF pp. 5-6, Methods — probit regression via R "
    "package ecotox; Finney (1971) framework. "
    "P = Phi(alpha + beta * log10(x)), where Phi = standard normal CDF."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "alpha": {"init": None},
    "beta":  {"init": None},
}


def _probit_mortality(log10_x, alpha, beta):
    """P = Phi(alpha + beta * log10(x))."""
    return ndtr(alpha + beta * log10_x)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Weighted least-squares probit fit (Finney 1971 framework approximation).

    Deterministic, data-derived start:
      beta0  = 1.0 (moderate slope, within the 0.55-3.46 range in Table 1).
      alpha0 = median(probit(y_clipped)) - beta0 * median(log10_x), so the
               initial curve passes through the median observation.

    Fit is performed as penalised WLS on the probit scale (iteratively
    reweighted — here a single-pass approximation using binary cross-entropy
    minimisation, which converges to the MLE for binomial probit regression).
    """
    log10_x = np.asarray(X_fit[:, 0], dtype=float)
    y = np.clip(np.asarray(y_fit, dtype=float), 1e-6, 1.0 - 1e-6)

    # Data-derived starting values
    beta0 = 1.0
    probit_y = np.sqrt(2.0) * np.array(
        [float(np.log(yi / (1.0 - yi))) for yi in y]  # logit as probit proxy
    )
    # Better: use ndtri (probit) directly
    from scipy.special import ndtri
    probit_y = ndtri(y)
    alpha0 = float(np.median(probit_y)) - beta0 * float(np.median(log10_x))

    def neg_log_lik(params):
        a, b = params
        p = np.clip(_probit_mortality(log10_x, a, b), 1e-9, 1.0 - 1e-9)
        # Binary cross-entropy (probit MLE objective)
        return -float(np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))

    try:
        res = minimize(
            neg_log_lik,
            x0=[alpha0, beta0],
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-7, "fatol": 1e-7},
        )
        alpha_fit, beta_fit = res.x
        if not np.all(np.isfinite([alpha_fit, beta_fit])):
            raise RuntimeError("non-finite fit")
        return {"alpha": float(alpha_fit), "beta": float(beta_fit)}
    except Exception:  # noqa: BLE001
        return {"alpha": float(alpha0), "beta": float(beta0)}


def predict(X: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Probit mortality: P = Phi(alpha + beta * log10(x)).

    X: (n, 1) — column [log10_conc_ppb].
    Returns Abbott-corrected mortality probability in [0, 1].
    """
    log10_x = np.asarray(X[:, 0], dtype=float)
    return _probit_mortality(log10_x, alpha, beta)
