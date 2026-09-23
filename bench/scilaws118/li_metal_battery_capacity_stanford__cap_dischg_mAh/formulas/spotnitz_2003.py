"""Spotnitz (2003) SEI-growth capacity fade — explicit sqrt expression.

Spotnitz (2003), J. Power Sources 113(1):72-80, Eq. 6b (PDF p. 6): the
explicit Ozawa-Spotnitz SEI-growth solution for SEI thickness,

    N_SEI = (1/D) * (sqrt(1 + 2*D*Rf0*t) - 1)

with Rf0 the initial fade-rate parameter and D a per-cycle attenuation.
Capacity loss is proportional to N_SEI; substituting cycle_index k for
time t (per-cycle factor absorbed into Rf0):

    cap_dischg_mAh = Q_0 * (1 - frac_loss),  frac_loss = N_SEI(k; Rf0, D)

Unlike Pinson's asymptotic large-t form (cap ~ sqrt(k)), Spotnitz's
expression keeps the finite-time correction, so it interpolates between
linear-in-k (small k) and sqrt-in-k (large k).

Chemistry caveat: same as pinson_2013 — Spotnitz derived this for
Li-ion / graphite cells; the Uppaluri dataset is Li-metal / NMC-811.
Included as a physics-motivated baseline.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster, computed by fit() via nonlinear LS
-----------------------------------------------------------------
- Rf0 : initial fractional fade rate per cycle.
- D   : per-cycle attenuation of the fade rate.
init = None on both: fit() builds its own data-derived start.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["cycle_index", "nominal_capacity_mAh"]
PAPER_REF = "summary_formula_spotnitz_2003.md"
EQUATION_LOC = (
    "Spotnitz (2003) Eq. 6b, PDF p. 6 — N_SEI = (1/D)*(sqrt(1+2*D*Rf0*t)-1); "
    "cap = Q_0 * (1 - N_SEI(k; Rf0, D))."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "Rf0": {"init": None},
    "D":   {"init": None},
}


def _frac_loss(k, Rf0, D):
    arg = np.clip(1.0 + 2.0 * D * Rf0 * k, 0.0, None)
    return (np.sqrt(arg) - 1.0) / D


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    k  = np.asarray(X_fit[:, 0], dtype=float)
    q0 = np.asarray(X_fit[:, 1], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # Init: small initial fade rate, small per-cycle attenuation.
    p0 = [1e-4, 1e-3]

    def residual(p):
        Rf0, D = p
        return q0 * (1.0 - _frac_loss(k, Rf0, D)) - y

    # Tighter bounds keep the (sqrt(1+2*D*Rf0*k) - 1)/D expression numerically
    # well-conditioned across the cycle range: D pushed to ~0 makes the
    # formula diverge under modest extrapolation (the form blows up to
    # multiples of Q_0 on cells with rapid fade), so we floor D at 1e-4.
    try:
        sol = least_squares(residual, p0,
                            bounds=([1e-7, 1e-4], [0.1, 0.5]),
                            method="trf", max_nfev=4000)
        return {"Rf0": float(sol.x[0]), "D": float(sol.x[1])}
    except Exception:                              # noqa: BLE001
        return {"Rf0": p0[0], "D": p0[1]}


def predict(X: np.ndarray, Rf0: float, D: float) -> np.ndarray:
    """cap = Q_0 * (1 - (sqrt(1 + 2*D*Rf0*k) - 1) / D).

    X: (n, 2) — columns cycle_index, nominal_capacity_mAh.
    """
    k  = np.asarray(X[:, 0], dtype=float)
    q0 = np.asarray(X[:, 1], dtype=float)
    return q0 * (1.0 - _frac_loss(k, Rf0, D))
