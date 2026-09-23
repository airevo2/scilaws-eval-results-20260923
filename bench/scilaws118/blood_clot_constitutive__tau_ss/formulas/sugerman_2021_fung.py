"""Isotropic Fung-type simple-shear stress — tau_ss.

Sugerman, G. P. *et al.* (2021). *A whole blood thrombus mimic: Constitutive
behavior under simple shear.* J. Mech. Behav. Biomed. Mater. 115:104216.
DOI:10.1016/j.jmbbm.2020.104216.

Sugerman 2021 §2.3 (PDF p. 3) defines the isotropic Fung-type strain-energy
density as

    W(C) = (c1 / 2c2) * [exp(c2 * (I1_tilde - 3)) - 1] + U(J),

with U(J) the FEBio decoupled volumetric penalty. For an incompressible,
isotropic, homogeneous hyperelastic solid in homogeneous simple shear of
amount gamma the deviatoric first invariant satisfies I1_tilde - 3 = gamma^2,
and the standard expression tau = 2*gamma*(dW/dI1 + dW/dI2) reduces — since
this Fung form depends only on I1 — to the closed-form simple-shear Cauchy
shear stress

    tau_ss = c1 * gamma * exp(c2 * gamma^2).

Sugerman 2021 Table 1 (PDF p. 7) reports per-sample fits across the 27
specimens: c1 in 267-581 Pa, c2 in 6.75-9.14 (dimensionless). Both parameters
are refit per sample; no universal scalar appears in the functional form.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The Fung-type exponential FORM is the scientific claim. c1 and c2 are
per-specimen material parameters (Sugerman 2021 Table 1).

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The 3 in (I1_tilde - 3), the 2 in (1/(2c2)), and the factor 2 in
tau = 2*gamma*dW/dI1 are part of the fixed algebraic form / continuum-
mechanics kinematics — not tunable constants.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- c1 : Fung scale modulus (kPa, > 0). Small-strain shear modulus = c1.
- c2 : Fung exponential stiffening exponent (dimensionless, >= 0).

init = None on both: fit() builds its own deterministic, data-derived
start. Single-start is sufficient; the loss is smooth in (c1, c2) on
+-50% simple-shear data.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["gamma"]
PAPER_REF = "summary_formula_dataset_sugerman_2021.md"
EQUATION_LOC = (
    "Sugerman 2021 §2.3 model 2 (PDF p. 3): W = (c1/2c2)[exp(c2*(I1~-3))-1] "
    "+ U(J); per-sample fits Table 1, PDF p. 7."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "c1": {"init": None},
    "c2": {"init": None},
}


def _tau_ss(gamma, c1, c2):
    arg = np.clip(c2 * gamma * gamma, -700.0, 700.0)
    return c1 * gamma * np.exp(arg)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the Fung-type simple-shear form.

    Deterministic, data-derived start:
      c2_0 = 8.0 — middle of the per-sample c2 distribution reported in
                    Sugerman 2021 Table 1 (6.75-9.14).
      c1_0 = small-strain slope estimate: median(|tau_ss| / |gamma|) over
              the fit data, since tau_ss ~ c1*gamma at small gamma.
    """
    gamma = np.asarray(X_fit[:, 0], dtype=float)
    y     = np.asarray(y_fit, dtype=float)

    nz = np.abs(gamma) > 1e-6
    if nz.any():
        c1_0 = float(np.median(np.abs(y[nz]) / np.abs(gamma[nz])))
    else:
        c1_0 = 1e-3
    if not np.isfinite(c1_0) or c1_0 <= 0:
        c1_0 = 1e-3
    c2_0 = 8.0

    # optimiser bounds (fit-procedure config, not formula constants)
    #            c1     c2
    param_lo = [1e-6,  0.0]
    param_hi = [1e4,   200.0]

    p0 = [c1_0, c2_0]
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, param_lo, param_hi)]

    def residual(p):
        return _tau_ss(gamma, p[0], p[1]) - y

    try:
        sol = least_squares(residual, p0, bounds=(param_lo, param_hi),
                            method="trf", max_nfev=4000)
        c1, c2 = sol.x
        if not np.all(np.isfinite([c1, c2])):
            raise RuntimeError("non-finite fit")
        return {"c1": float(c1), "c2": float(c2)}
    except Exception:                                  # noqa: BLE001
        return {"c1": float(p0[0]), "c2": float(p0[1])}


def predict(X: np.ndarray, c1: float, c2: float) -> np.ndarray:
    """Isotropic Fung-type simple-shear Cauchy stress.

    X: (n, 1) — column [gamma].
    """
    gamma = np.asarray(X[:, 0], dtype=float)
    return _tau_ss(gamma, c1, c2)
