"""Geiger-Nuttall alpha-decay law — per-Z-chain fit (Qi et al. 2014).

Citation
--------
Qi, C., Andreyev, A. N., Huyse, M., Liotta, R. J., Van Duppen, P., Wyss, R.
"On the validity of the Geiger-Nuttall alpha-decay law and its microscopic basis."
Physics Letters B 734 (2014) 203-206.
DOI: 10.1016/j.physletb.2014.05.066; arXiv:1405.5633.
PDF: reference/qi_2014.pdf, page 1.

Formula
-------
Equation (1), PDF p. 1:

    log10(T_1/2 / s) = A(Z) * Q_alpha^(-1/2) + B(Z)

where:
  - Q_alpha  : alpha-decay Q-value in MeV  (input column Q_alpha_MeV)
  - A(Z)     : per-Z-chain slope coefficient (LOCAL_FITTABLE)
  - B(Z)     : per-Z-chain intercept coefficient (LOCAL_FITTABLE)
  - exponent -1/2 is a structural literal (quantum tunnelling result)

Surrounding text on PDF p. 1: "A(Z) and B(Z) are the coefficients which are
determined by fitting experimental data for each isotopic chain."

LAW_CONSTANTS
-------------
None. The law's claim is the functional form itself (linear in Q_alpha^{-1/2});
no universal numeric constants appear in Eq. (1).

OTHER_CONSTANTS
---------------
None. The formula is dimensionless in the predicted quantity (log10 of seconds)
and takes Q_alpha in MeV directly. The exponent -0.5 is inline.

LOCAL_FITTABLE
--------------
- A_Z : per-Z-chain slope (MeV^{1/2} units). Typical range ~80-180
        for heavy elements (e.g. Z=92 Uranium has A~137 from the paper's Fig. 1).
- B_Z : per-Z-chain intercept (dimensionless). Typical range ~ -50 to -30.

Both are fit by closed-form ordinary least squares (OLS) within each cluster.
With as few as 2 data points the fit is exact; with more it is the optimal
linear fit. init=None signals closed-form fit (no iterative optimiser).

Type designation
----------------
Type II. Criterion (a): A(Z) and B(Z) are per-cluster (per-Z-chain) fitted
parameters, not universal constants. Criterion (b): each Z-chain (element) is
a natural cluster — isotopes of different elements are not interchangeable.

Column mapping
--------------
Paper Q_alpha -> released CSV column Q_alpha_MeV (MeV).
Paper log10(T_1/2/s) -> target column log10_half_life_s.

Caveats
-------
- The GN law breaks for neutron-deficient Po isotopes (A<196, Z=84): see
  paper Fig. 1b. These remain in the dataset as documented hard cases.
- For clusters with only 2 rows the OLS fit is exact (0 residual dof); the
  test window then purely tests extrapolation.
- The fit is closed-form linear: no multi-start, no tolerance issues.
"""

import numpy as np

USED_INPUTS = ["Q_alpha_MeV"]
PAPER_REF = "summary_formula_dataset_qi_2014.md"
EQUATION_LOC = "Eq. (1), PDF p. 1 — log10(T_1/2/s) = A(Z)*Q_alpha^(-1/2) + B(Z)"

LAW_CONSTANTS = {}   # the form itself is the scientific claim; no universal scalar
OTHER_CONSTANTS = {} # dimensionless formula; Q in MeV directly; -0.5 inline

LOCAL_FITTABLE = {
    "A_Z": {"init": None},   # closed-form OLS — scalar, single start
    "B_Z": {"init": None},   # closed-form OLS — scalar, single start
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS fit of log10(T_1/2/s) vs Q_alpha^(-1/2).

    Returns A_Z (slope) and B_Z (intercept) for the cluster.
    With n=2 rows the solution is exact; with n>2 it is the least-squares
    linear regression.
    """
    Q = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)
    x = Q ** (-0.5)
    # OLS: [x, 1] * [A_Z, B_Z]^T = y
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"A_Z": float(coef[0]), "B_Z": float(coef[1])}


def predict(X: np.ndarray, A_Z: float, B_Z: float) -> np.ndarray:
    """Evaluate Geiger-Nuttall law: log10(T_1/2/s) = A_Z * Q^(-1/2) + B_Z.

    X[:, 0] = Q_alpha_MeV (alpha-decay Q-value in MeV).
    """
    Q = np.asarray(X[:, 0], dtype=float)
    return A_Z * Q ** (-0.5) + B_Z
