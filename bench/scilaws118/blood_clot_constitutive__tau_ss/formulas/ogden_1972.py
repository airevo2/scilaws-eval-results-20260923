"""One-term Ogden simple-shear stress — tau_ss.

Ogden, R. W. (1972). *Large deformation isotropic elasticity — on the
correlation of theory and experiment for incompressible rubberlike solids.*
Proc. R. Soc. Lond. A 326(1567):565-584. DOI:10.1098/rspa.1972.0026.

For an incompressible isotropic hyperelastic solid in homogeneous simple
shear with amount-of-shear gamma, the larger principal stretch a satisfies
a - a^(-1) = gamma (Ogden 1972 Eq. 19, PDF p. 573), so

    a = (gamma + sqrt(gamma^2 + 4)) / 2.

The one-term Ogden Cauchy shear stress is

    tau_ss = mu * (a^alpha - a^(-alpha)) / (a + a^(-1)).

Kakaletsis et al. (2023) Table 2 (PDF p. 12) reports per-sample (a, b) fits
of this one-term Ogden form to the 27 Sugerman 2021 blood-clot specimens
with a == mu (in Pa) in 530-847 Pa and b == alpha (dimensionless) in
15.14-16.32 across the best/median/worst NMSE samples. Sugerman et al.
(2021, Table 1 PDF p. 7) gives the equivalent strain-energy form
W = (c1/c2^2)[lam1^c2 + lam2^c2 + lam3^c2 - 3] with c1 in 515-1401 Pa and
c2 in 11.2-16.4 across the same 27 specimens; that form collapses to the
present formula with mu = c1/c2 and alpha = c2.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The one-term Ogden FORM is the scientific claim. Both mu and alpha
are refit per blood-clot specimen in every published source (Sugerman 2021
Table 1 per-sample; Kakaletsis 2023 Table 2 per-sample); no universal
numerical value across specimens is published — both are LOCAL_FITTABLE.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The 4 in sqrt(gamma^2 + 4) and the 2 in the half-stretch formula
come from the algebraic incompressible-simple-shear kinematics a*a^(-1)*1=1,
not tunable constants.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- mu    : Ogden modulus (kPa, > 0). Small-strain shear modulus = mu*alpha/2.
- alpha : strain-hardening exponent (dimensionless). Per-sample values for
          this dataset are tightly clustered around ~12-16.

init = None on both: fit() builds its own deterministic, data-derived
start (single-start Levenberg-Marquardt / trust-region — Ogden is
well-conditioned on +-50% simple-shear data, so multi-start is not needed).
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["gamma"]
PAPER_REF = "summary_formula_ogden_1972.md"
EQUATION_LOC = (
    "Ogden 1972 Eq. 19, PDF p. 573 (one-term simple-shear Cauchy stress); "
    "kinematic relation a - a^(-1) = gamma also PDF p. 573."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "mu":    {"init": None},
    "alpha": {"init": None},
}


def _tau_ss(gamma, mu, alpha):
    a = 0.5 * (gamma + np.sqrt(gamma * gamma + 4.0))
    inv_a = 1.0 / a
    # alpha is bounded < 50 by fit(); a^alpha won't overflow on |gamma|<=0.52.
    return mu * (a ** alpha - a ** (-alpha)) / (a + inv_a)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the one-term Ogden form.

    Deterministic, data-derived start:
      alpha0 = 12.0 — mid-range value of the per-sample alpha distribution
                       reported in Sugerman 2021 Table 1 / Kakaletsis 2023
                       Table 2 (11.2-16.4).
      mu0    = scale so the small-strain limit tau_ss ~ (mu*alpha/2)*gamma
                       matches the median observed |tau_ss| / |gamma|.
    Single-start is sufficient for the one-term Ogden form on simple-shear
    data; the loss is smooth and well-conditioned.
    """
    gamma = np.asarray(X_fit[:, 0], dtype=float)
    y     = np.asarray(y_fit, dtype=float)

    alpha0 = 12.0
    nz = np.abs(gamma) > 1e-6
    if nz.any():
        slope_est = float(np.median(np.abs(y[nz]) / np.abs(gamma[nz])))
    else:
        slope_est = 1.0
    mu0 = (2.0 * slope_est / alpha0) if slope_est > 0 else 1e-3
    if not np.isfinite(mu0) or mu0 <= 0:
        mu0 = 1e-3

    # optimiser bounds (fit-procedure config, not formula constants)
    #            mu     alpha
    param_lo = [1e-6,  0.1]
    param_hi = [1e4,   50.0]

    p0 = [mu0, alpha0]
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, param_lo, param_hi)]

    def residual(p):
        return _tau_ss(gamma, p[0], p[1]) - y

    try:
        sol = least_squares(residual, p0, bounds=(param_lo, param_hi),
                            method="trf", max_nfev=4000)
        mu, alpha = sol.x
        if not np.all(np.isfinite([mu, alpha])):
            raise RuntimeError("non-finite fit")
        return {"mu": float(mu), "alpha": float(alpha)}
    except Exception:                                  # noqa: BLE001
        return {"mu": float(p0[0]), "alpha": float(p0[1])}


def predict(X: np.ndarray, mu: float, alpha: float) -> np.ndarray:
    """One-term Ogden simple-shear Cauchy stress.

    X: (n, 1) — column [gamma].
    """
    gamma = np.asarray(X[:, 0], dtype=float)
    return _tau_ss(gamma, mu, alpha)
