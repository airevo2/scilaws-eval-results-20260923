"""CANN-discovered two-term quadratic-exponential constitutive model — sigma22.

Linka, K. *et al.* (2023). *Automated model discovery for skin: Discovering
the best model, data, and experiment.* Computer Methods in Applied Mechanics
and Engineering 410:116007.

Linka et al. report that a Constitutive Artificial Neural Network (CANN)
robustly discovers a two-term free energy for skin (Eqs. 22-23, PDF p. 11):

    psi(I1, I4) = (a1/(2*b1)) * (exp(b1*(I1-3)^2) - 1)
                + (a4/(2*b4)) * (exp(b4*(I4-1)^2) - 1)

with I1 the first isotropic invariant and I4 the anisotropic fiber-stretch
invariant. For incompressible planar biaxial loading with principal stretches
lambda11, lambda22 and lambda33 = 1/(lambda11*lambda22) and a fiber direction
M0 = (cos(alpha), sin(alpha), 0):

    I1 = lambda11^2 + lambda22^2 + (lambda11*lambda22)^(-2)
    I4 = lambda11^2*cos^2(alpha) + lambda22^2*sin^2(alpha)

The plane-stress condition sigma33 = 0 (Eq. 19, PDF p. 6) gives the
cranial-caudal Cauchy stress:

    sigma22 = 2*(lambda22^2 - (lambda11*lambda22)^(-2))
              * a1*(I1-3)*exp(b1*(I1-3)^2)
            + 2*lambda22^2*sin^2(alpha)
              * a4*(I4-1)*exp(b4*(I4-1)^2)

This is the direction-2 companion of the lateral (sigma11) form: the kinematic
factor uses lambda22^2 in place of lambda11^2, and the fiber term picks up the
direction-2 fiber projection sin^2(alpha).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The two-term quadratic-exponential FORM is the scientific claim. Linka
2023 publishes group-level CANN weights for pig skin (PDF p. 11: a1 ~ 1.329
MPa, b1 ~ 0.8207, a4 ~ 0.266 MPa, b4 ~ 0.392) and rabbit skin, but the
benchmark data is mouse skin — no published mouse-skin coefficients exist, and
the CANN is retrained per dataset. The pig values are recorded in this
docstring as a magnitude reference only; all five parameters are per-specimen
fits for this benchmark.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The 3 in (I1-3), the 1 in (I4-1), the 2 prefactors and the squaring
of the brackets are part of the fixed algebraic form.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- a1    : isotropic stiffness coefficient (MPa, > 0).
- b1    : isotropic exponential shape parameter (dimensionless, > 0).
- a4    : anisotropic (fiber) stiffness coefficient (MPa, > 0).
- b4    : anisotropic exponential shape parameter (dimensionless, > 0).
- alpha : collagen fiber angle from the x-axis (radians).

init: a1, b1, a4, b4 carry init = None (fit() builds its own deterministic,
data-derived start); alpha carries a 4-element init list — the deterministic
multi-start sweep over the fiber angle (the loss is non-convex in that angle).
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["lambda11", "lambda22"]
PAPER_REF = "summary_formula_linka_2023.md"
EQUATION_LOC = (
    "Linka et al. (2023) Eqs. 22-23, PDF p. 11 (two-term CANN free energy); "
    "biaxial kinematics Eqs. 15-19, PDF p. 6; plane-stress sigma22 from "
    "sigma33 = 0."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "a1":    {"init": None},
    "b1":    {"init": None},
    "a4":    {"init": None},
    "b4":    {"init": None},
    "alpha": {"init": [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]},
}


def _sigma22(lam1, lam2, a1, b1, a4, b4, alpha):
    lam3_sq = (lam1 * lam2) ** (-2.0)
    I1 = lam1 ** 2 + lam2 ** 2 + lam3_sq
    cos2 = np.cos(alpha) ** 2
    sin2 = np.sin(alpha) ** 2
    I4 = lam1 ** 2 * cos2 + lam2 ** 2 * sin2
    dI1 = I1 - 3.0
    dI4 = I4 - 1.0
    dpsi_dI1 = a1 * dI1 * np.exp(np.clip(b1 * dI1 ** 2, -700.0, 700.0))
    dpsi_dI4 = a4 * dI4 * np.exp(np.clip(b4 * dI4 ** 2, -700.0, 700.0))
    return (2.0 * (lam2 ** 2 - lam3_sq) * dpsi_dI1
            + 2.0 * lam2 ** 2 * sin2 * dpsi_dI4)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the Linka two-term form.

    Deterministic, data-derived start: a1 from the median observed
    stress / linear-kinematic term, b1/b4 mild defaults, a4 = a1, with a
    multi-start sweep over alpha. The best (lowest-cost) solve is returned;
    each solve is wrapped so overflow / non-convergence is skipped.
    """
    lam1 = np.asarray(X_fit[:, 0], dtype=float)
    lam2 = np.asarray(X_fit[:, 1], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    lam3_sq = (lam1 * lam2) ** (-2.0)
    kin = 2.0 * (lam2 ** 2 - lam3_sq)
    denom = float(np.median(np.abs(kin)))
    scale = float(np.median(np.abs(y)))
    a1_0 = (scale / denom) if denom > 0 else 1e-3
    if not np.isfinite(a1_0) or a1_0 <= 0:
        a1_0 = 1e-3
    b1_0, a4_0, b4_0 = 0.5, a1_0, 0.5

    # optimiser bounds (fit-procedure config, not formula constants)
    #            a1     b1      a4     b4      alpha
    param_lo = [0.0,   1e-9,   0.0,   1e-9,   0.0]
    param_hi = [1e4,   500.0,  1e4,   500.0,  np.pi]
    angle_starts = LOCAL_FITTABLE["alpha"]["init"]

    def residual(p):
        return _sigma22(lam1, lam2, p[0], p[1], p[2], p[3], p[4]) - y

    best = None
    best_cost = np.inf
    for alpha0 in angle_starts:
        p0 = [a1_0, b1_0, a4_0, b4_0, alpha0]
        p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, param_lo, param_hi)]
        try:
            sol = least_squares(residual, p0, bounds=(param_lo, param_hi),
                                method="trf", max_nfev=4000)
            if sol.cost < best_cost and np.all(np.isfinite(sol.x)):
                best_cost = sol.cost
                best = sol.x
        except Exception:                          # noqa: BLE001
            continue

    if best is None:
        best = [min(max(v, lo), hi)
                for v, lo, hi in zip([a1_0, b1_0, a4_0, b4_0, angle_starts[2]],
                                     param_lo, param_hi)]
    a1, b1, a4, b4, alpha = best
    return {"a1": float(a1), "b1": float(b1), "a4": float(a4),
            "b4": float(b4), "alpha": float(alpha)}


def predict(X: np.ndarray, a1: float, b1: float, a4: float, b4: float,
            alpha: float) -> np.ndarray:
    """Linka two-term CANN biaxial cranial-caudal Cauchy stress sigma22.

    X: (n, 2) — columns [lambda11, lambda22].
    """
    lam1 = np.asarray(X[:, 0], dtype=float)
    lam2 = np.asarray(X[:, 1], dtype=float)
    return _sigma22(lam1, lam2, a1, b1, a4, b4, alpha)
