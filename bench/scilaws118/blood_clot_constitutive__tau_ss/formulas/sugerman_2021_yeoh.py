"""Yeoh-3 simple-shear stress — tau_ss.

Sugerman, G. P. *et al.* (2021). *A whole blood thrombus mimic: Constitutive
behavior under simple shear.* J. Mech. Behav. Biomed. Mater. 115:104216.
DOI:10.1016/j.jmbbm.2020.104216.

Sugerman 2021 §2.3 (PDF p. 3) defines the Yeoh strain-energy density as

    W(C) = sum_{i=1..3} c_i (I1_tilde - 3)^i + U(J).

For an incompressible, isotropic, homogeneous hyperelastic solid in
homogeneous simple shear of amount gamma we have I1_tilde - 3 = gamma^2, so
dW/dI1 = c1 + 2*c2*gamma^2 + 3*c3*gamma^4, and tau = 2*gamma*dW/dI1 gives the
closed-form simple-shear Cauchy shear stress

    tau_ss = 2*c1*gamma + 4*c2*gamma^3 + 6*c3*gamma^5.

Sugerman 2021 Table 1 (PDF p. 7) reports highly variable per-sample fits
across the 27 specimens: c1 in 0.01-101 Pa, c2 in 1133-2742 Pa, c3 in
0.04-3390 Pa. All three are refit per sample; no universal scalar appears
in the functional form.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The Yeoh-3 polynomial-strain-energy FORM is the scientific claim. The
three Yeoh coefficients c1, c2, c3 are per-specimen material parameters
(Sugerman 2021 Table 1).

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The 3 in (I1_tilde - 3), and the polynomial-degree prefactors
2, 4, 6 (from tau = 2*gamma*dW/dI1 with d/dI1 of the i-th term yielding
i*(I1_tilde - 3)^(i-1)) are part of the fixed algebraic form — not tunable.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded linear LS
--------------------------------------------------------------------
- c1 : Yeoh leading coefficient (kPa, >= 0). Small-strain shear modulus = 2*c1.
- c2 : Yeoh quadratic-stiffening coefficient (kPa). The Sugerman 2021
       per-sample fits report this as the dominant nonlinear term for this
       dataset (range 1133-2742 Pa).
- c3 : Yeoh quartic-stiffening coefficient (kPa).

init = None on all three: the model is LINEAR in (c1, c2, c3) for fixed
gamma, so fit() solves it as a bounded linear least-squares problem with
a single closed-form / deterministic call (no multi-start needed).
"""

import numpy as np
from scipy.optimize import lsq_linear

USED_INPUTS = ["gamma"]
PAPER_REF = "summary_formula_dataset_sugerman_2021.md"
EQUATION_LOC = (
    "Sugerman 2021 §2.3 model 3 (PDF p. 3): W = sum_{i=1..3} c_i (I1~ - 3)^i "
    "+ U(J); per-sample fits Table 1, PDF p. 7."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "c1": {"init": None},
    "c2": {"init": None},
    "c3": {"init": None},
}


def _tau_ss(gamma, c1, c2, c3):
    g2 = gamma * gamma
    g3 = g2 * gamma
    g5 = g3 * g2
    return 2.0 * c1 * gamma + 4.0 * c2 * g3 + 6.0 * c3 * g5


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded linear least-squares fit of the Yeoh-3 simple-shear form.

    The model is linear in (c1, c2, c3) at fixed gamma, so we assemble the
    design matrix [2*gamma, 4*gamma^3, 6*gamma^5] and solve directly with
    scipy.optimize.lsq_linear. Single deterministic call.
    """
    gamma = np.asarray(X_fit[:, 0], dtype=float)
    y     = np.asarray(y_fit, dtype=float)
    g2 = gamma * gamma
    g3 = g2 * gamma
    g5 = g3 * g2
    A = np.column_stack([2.0 * gamma, 4.0 * g3, 6.0 * g5])

    # optimiser bounds (fit-procedure config, not formula constants)
    #            c1     c2     c3
    param_lo = [-1e4, -1e4,  -1e4]
    param_hi = [ 1e4,  1e4,   1e4]

    try:
        sol = lsq_linear(A, y, bounds=(param_lo, param_hi))
        c1, c2, c3 = sol.x
        if not np.all(np.isfinite([c1, c2, c3])):
            raise RuntimeError("non-finite fit")
        return {"c1": float(c1), "c2": float(c2), "c3": float(c3)}
    except Exception:                                  # noqa: BLE001
        return {"c1": 0.0, "c2": 0.0, "c3": 0.0}


def predict(X: np.ndarray, c1: float, c2: float, c3: float) -> np.ndarray:
    """Yeoh-3 polynomial simple-shear Cauchy stress.

    X: (n, 1) — column [gamma].
    """
    gamma = np.asarray(X[:, 0], dtype=float)
    return _tau_ss(gamma, c1, c2, c3)
