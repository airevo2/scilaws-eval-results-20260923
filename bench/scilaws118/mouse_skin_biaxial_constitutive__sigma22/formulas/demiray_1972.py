"""Demiray (1972) isotropic exponential strain-energy — biaxial sigma22.

Demiray, H. (1972). *A note on the elasticity of soft biological tissues.*
Journal of Biomechanics 5(3):309-311. DOI:10.1016/0021-9290(72)90047-4.

Demiray Eq. 5 (PDF p. 2) proposes an isotropic, incompressible strain-energy
density for soft biological tissue,

    Sigma = (beta / (2*alpha)) * (exp(alpha*(I1 - 3)) - 1)

with alpha, beta > 0 material constants. The general incompressible Cauchy
stress is t^kl = p g^kl + Phi C^kl (Eq. 7, PDF p. 2), with
Phi = 2 dSigma/dI1 = beta * exp(alpha*(I1 - 3)).

For planar biaxial extension of a thin incompressible membrane with in-plane
principal stretches lambda11, lambda22 and (by incompressibility)
lambda33 = 1/(lambda11*lambda22):

    I1 = lambda11^2 + lambda22^2 + (lambda11*lambda22)^(-2)

The plane-stress condition sigma33 = 0 fixes the Lagrange multiplier p, and
the cranial-caudal Cauchy stress reduces to

    sigma22 = beta * (lambda22^2 - (lambda11*lambda22)^(-2)) * exp(alpha*(I1 - 3))

This is the direction-2 companion of the lateral (sigma11) form: the kinematic
factor uses lambda22^2 (the direction whose stress is sought) in place of
lambda11^2. It is the standard continuum-mechanics specialisation of Demiray
Eq. 7 (the paper works only the uniaxial example, Eq. 14, PDF p. 2).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The exponential strain-energy FORM is the scientific claim. alpha and
beta are "material constants to be determined through experimental studies"
(Demiray 1972, PDF p. 2, below Eq. 5) — no universal numerical values are
published; both are per-specimen fits.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The 3 in (I1 - 3) and the exponent 2 are part of the fixed
algebraic form, not tunable constants.

LOCAL_FITTABLE — per-cluster, fitted by fit() via nonlinear least squares
--------------------------------------------------------------------------
- alpha : exponential stiffening rate (dimensionless, > 0).
- beta  : stress-scale modulus (MPa, > 0).

init = None on both: fit() builds its own deterministic, data-derived start.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["lambda11", "lambda22"]
PAPER_REF = "summary_formula_demiray_1972.md"
EQUATION_LOC = (
    "Demiray (1972) Eq. 5 (strain energy, PDF p. 2) + Eq. 7 (Cauchy stress, "
    "PDF p. 2); biaxial sigma22 by the plane-stress condition sigma33 = 0."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "alpha": {"init": None},
    "beta":  {"init": None},
}


def _sigma22(lam1, lam2, alpha, beta):
    lam3_sq = (lam1 * lam2) ** (-2.0)
    I1 = lam1 ** 2 + lam2 ** 2 + lam3_sq
    arg = np.clip(alpha * (I1 - 3.0), -700.0, 700.0)
    return beta * (lam2 ** 2 - lam3_sq) * np.exp(arg)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the Demiray biaxial form.

    Deterministic, data-derived start:
      alpha0 = 1.0 — a generic mild exponential stiffening rate;
      beta0  = a scale set so the linear-kinematic term matches the median
               observed stress over the fit data.
    The solve is wrapped so an overflow or non-convergence falls back to the
    start.
    """
    lam1 = np.asarray(X_fit[:, 0], dtype=float)
    lam2 = np.asarray(X_fit[:, 1], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    alpha0 = 1.0
    lam3_sq = (lam1 * lam2) ** (-2.0)
    kin = lam2 ** 2 - lam3_sq
    denom = float(np.median(np.abs(kin)))
    scale = float(np.median(np.abs(y)))
    beta0 = (scale / denom) if denom > 0 else 1e-3
    if not np.isfinite(beta0) or beta0 <= 0:
        beta0 = 1e-3

    # optimiser bounds (fit-procedure config, not formula constants)
    param_lo = [1e-6,   1e-9]
    param_hi = [50.0,   1e4]

    p0 = [alpha0, beta0]
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, param_lo, param_hi)]

    def residual(p):
        return _sigma22(lam1, lam2, p[0], p[1]) - y

    try:
        sol = least_squares(residual, p0, bounds=(param_lo, param_hi),
                            method="trf", max_nfev=4000)
        alpha, beta = sol.x
        if not np.all(np.isfinite([alpha, beta])):
            raise RuntimeError("non-finite fit")
        return {"alpha": float(alpha), "beta": float(beta)}
    except Exception:                              # noqa: BLE001
        return {"alpha": float(p0[0]), "beta": float(p0[1])}


def predict(X: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """sigma22 = beta*(lambda22^2 - lambda33^2)*exp(alpha*(I1 - 3)).

    X: (n, 2) — columns [lambda11, lambda22]. lambda33 = 1/(lambda11*lambda22).
    """
    lam1 = np.asarray(X[:, 0], dtype=float)
    lam2 = np.asarray(X[:, 1], dtype=float)
    return _sigma22(lam1, lam2, alpha, beta)
