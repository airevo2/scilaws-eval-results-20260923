"""Gasser-Ogden-Holzapfel (GOH) anisotropic constitutive model — biaxial sigma22.

Tac, V. *et al.* (2022). *Data-driven modeling of the mechanical behavior of
anisotropic soft biological tissue.* Engineering with Computers 38:4167-4182.
DOI:10.1007/s00366-022-01733-3.

Tac et al. §2.2.1 (Eqs. 23-26, PDF pp. 6-7) instantiate the GOH isochoric
strain energy for a single fiber family (this is the same GOH form adopted by
Meador et al. 2020 Eq. 1 for the murine dermis data used here):

    Psi_iso(C)      = mu*(I1 - 3)                               (Eq. 24)
    Psi_aniso(C,a0) = (k1/(4*k2)) * (exp(k2*E^2) - 1)           (Eq. 25)
    E               = kappa*I1 + (1 - 3*kappa)*I4 - 1           (Eq. 26)

with a0 = (sin(theta), cos(theta), 0) the mean fiber direction (Eq. 23). This
file is the direction-2 companion of the sister sigma11 task's tac_2022.py:
the SAME GOH form, parameters and theta convention — only the stress
component projected out differs.

For an incompressible material under planar biaxial loading with principal
stretches lambda11, lambda22 and lambda33 = 1/(lambda11*lambda22):

    I1 = lambda11^2 + lambda22^2 + (lambda11*lambda22)^(-2)
    I4 = lambda11^2*sin^2(theta) + lambda22^2*cos^2(theta)

The plane-stress condition sigma33 = 0 fixes the Lagrange multiplier; the
cranial-caudal Cauchy stress (push-forward sigma = F S F^T / J) is

    sigma22 = 2*(lambda22^2 - (lambda11*lambda22)^(-2)) * dPsi/dI1
            + 2*lambda22^2*cos^2(theta) * dPsi/dI4

    dPsi/dI1 = mu + k1*kappa*E*exp(k2*E^2)
    dPsi/dI4 = k1*(1 - 3*kappa)*E*exp(k2*E^2)

Relative to the sigma11 form: the kinematic factor uses lambda22^2 in place
of lambda11^2, and the fiber term picks up the direction-2 fiber projection
cos^2(theta) (a0's second component) in place of sin^2(theta). I4 is an
invariant — unchanged. The generalised fiber strain E is clipped at zero
(collagen fibers buckle rather than carry load in compression; Tac 2022 §3
convention, PDF p. 7).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The GOH strain-energy FORM is the scientific claim. All five material
parameters are refit per specimen: Meador 2020 §2.6 (PDF p. 3) "we identified
the five unknown material parameters mu, k1, k2, alpha, kappa for each
sample"; Tac 2022 §2.3 fits the GOH parameters per dataset. No universal
coefficient across specimens is published.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The 3 in (I1 - 3), the 1 in (E + 1), the 3 in (1 - 3*kappa), the 2
prefactors and the exponent 2 are part of the fixed algebraic form.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- mu    : isotropic neo-Hookean modulus (MPa, > 0).
- k1    : fiber stiffness prefactor (MPa, >= 0).
- k2    : fiber nonlinearity exponent (dimensionless, >= 0).
- kappa : fiber dispersion (dimensionless, constrained to [0, 1/3] by the
          GOH model physics).
- theta : mean fiber angle to the x-axis (radians).

init: mu, k1, k2, kappa carry init = None (fit() builds its own
deterministic, data-derived start); theta carries a 4-element init list —
the deterministic multi-start sweep over the fiber angle (the loss is
non-convex in that angle).
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["lambda11", "lambda22"]
PAPER_REF = "summary_formula_dataset_tac_2022.md"
EQUATION_LOC = (
    "Tac et al. (2022) Eqs. 23-26, PDF pp. 6-7 (GOH strain energy); biaxial "
    "sigma22 Cauchy stress from the plane-stress reduction Eqs. 5-9, PDF pp. 2-3."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "mu":    {"init": None},
    "k1":    {"init": None},
    "k2":    {"init": None},
    "kappa": {"init": None},
    "theta": {"init": [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]},
}


def _sigma22(lam1, lam2, mu, k1, k2, kappa, theta):
    lam3_sq = (lam1 * lam2) ** (-2.0)
    I1 = lam1 ** 2 + lam2 ** 2 + lam3_sq
    sin2 = np.sin(theta) ** 2
    cos2 = np.cos(theta) ** 2
    I4 = lam1 ** 2 * sin2 + lam2 ** 2 * cos2
    E = kappa * I1 + (1.0 - 3.0 * kappa) * I4 - 1.0
    E_pos = np.maximum(E, 0.0)
    exp_term = np.exp(np.clip(k2 * E_pos ** 2, -700.0, 700.0))
    dPsi_dI1 = mu + k1 * kappa * E_pos * exp_term
    dPsi_dI4 = k1 * (1.0 - 3.0 * kappa) * E_pos * exp_term
    return (2.0 * (lam2 ** 2 - lam3_sq) * dPsi_dI1
            + 2.0 * lam2 ** 2 * cos2 * dPsi_dI4)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the GOH biaxial form.

    Deterministic, data-derived start: mu0 from the small-strain slope of
    the observed stress, k1/k2 mild defaults, kappa0 = 0.2 (a typical dermal
    dispersion), and a multi-start sweep over theta (the loss is non-convex
    in the fiber angle). The best (lowest-cost) solve over the theta grid is
    returned; each solve is wrapped so overflow / non-convergence is skipped.
    """
    lam1 = np.asarray(X_fit[:, 0], dtype=float)
    lam2 = np.asarray(X_fit[:, 1], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    lam3_sq = (lam1 * lam2) ** (-2.0)
    kin = 2.0 * (lam2 ** 2 - lam3_sq)
    denom = float(np.median(np.abs(kin)))
    scale = float(np.median(np.abs(y)))
    mu0 = (scale / denom) if denom > 0 else 1e-3
    if not np.isfinite(mu0) or mu0 <= 0:
        mu0 = 1e-3
    k1_0, k2_0, kappa0 = mu0, 5.0, 0.2

    # optimiser bounds (fit-procedure config, not formula constants)
    #            mu     k1      k2     kappa   theta
    param_lo = [1e-9,  0.0,    0.0,   0.0,    0.0]
    param_hi = [1e4,   1e4,    500.0, 1.0 / 3.0, np.pi]
    angle_starts = LOCAL_FITTABLE["theta"]["init"]

    def residual(p):
        return _sigma22(lam1, lam2, p[0], p[1], p[2], p[3], p[4]) - y

    best = None
    best_cost = np.inf
    for theta0 in angle_starts:
        p0 = [mu0, k1_0, k2_0, kappa0, theta0]
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
                for v, lo, hi in zip([mu0, k1_0, k2_0, kappa0, angle_starts[2]],
                                     param_lo, param_hi)]
    mu, k1, k2, kappa, theta = best
    return {"mu": float(mu), "k1": float(k1), "k2": float(k2),
            "kappa": float(kappa), "theta": float(theta)}


def predict(X: np.ndarray, mu: float, k1: float, k2: float,
            kappa: float, theta: float) -> np.ndarray:
    """GOH biaxial cranial-caudal Cauchy stress sigma22.

    X: (n, 2) — columns [lambda11, lambda22].
    """
    lam1 = np.asarray(X[:, 0], dtype=float)
    lam2 = np.asarray(X[:, 1], dtype=float)
    return _sigma22(lam1, lam2, mu, k1, k2, kappa, theta)
