"""Von Bertalanffy Growth Function (VBGF) for fish length at age — length_mm.

Ricker, W. E. (1975). Computation and interpretation of biological statistics
of fish populations. Bulletin 191, Department of the Environment, Fisheries
and Marine Service, Canada.
URL: https://waves-vagues.dfo-mpo.gc.ca/Library/1485.pdf

The Von Bertalanffy Growth Function (VBGF) gives mean total length at age t:

    Lt = L_inf * (1 - exp(-K * (t - t0)))        Ricker (1975) Eq. 9.9, PDF p. 221

where:
- L_inf is the asymptotic mean length (length the average fish would reach
  if it lived and grew indefinitely; Ricker 1975 section 9.6.1, PDF p. 220-221).
- K is the Brody growth coefficient (year^-1); controls rate of approach to L_inf.
  "It is misleading to refer to K as a growth rate; a better name is the Brody
  growth coefficient" (Ricker 1975 section 9.6.1, PDF p. 220).
- t0 is the hypothetical age (years) at which length would be zero if the fish had
  always grown according to this equation; can be negative (Ricker 1975 section
  9.6.2, PDF p. 221).

This is Morat et al. (2020) Eq. (4), pdf p. 5:
    Lt = L_inf * (1 - e^{-K*(t - t0)})
fitted in a Bayesian hierarchical framework per species
(target: Li_sp_m, mean back-calculated total length in mm).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The VBGF form (Eq. 9.9) is the scientific claim. L_inf, K, and t0 are
all per-species fit parameters. No universal numerical values are published
for these parameters across coral reef fish species.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
None. The 1 in (1 - exp(...)) and the negative sign in the exponent -K*(t-t0)
are algebraic invariants of the VBGF form derived from Brody's decreasing-
exponential formulation (Ricker 1975 Eq. 9.7, PDF p. 220); they are not
tunable constants.

LOCAL_FITTABLE — per-cluster (per-species), fitted by fit() via nonlinear LS
-----------------------------------------------------------------------------
- L_inf : asymptotic mean length (mm, > 0). Upper bound on predicted length.
- K     : Brody growth coefficient (year^-1, > 0). Typical reef fish: 0.1-2.0.
- t0    : hypothetical age at zero length (years). Often slightly negative.

init = None on all three: fit() builds deterministic, data-derived starts
using 1.1 * max(observed length) for L_inf, 0.3 for K, 0.0 for t0.
Single-start Levenberg-Marquardt (TRF) is sufficient; the VBGF loss is
smooth and well-conditioned on per-species length-at-age data.

S2-P12 C10 classification sweep:
  - LAW_CONSTANTS = {} (no paper-fixed universal scalars)
  - The invariant 1 and -1 in predict() body are structural, not in LAW.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["Agei"]
PAPER_REF = "summary_formula_ricker_1975.md"
EQUATION_LOC = (
    "Ricker (1975) Eq. 9.9, PDF p. 221 — Lt = L_inf * (1 - exp(-K*(t-t0))); "
    "reproduced as Morat et al. (2020) Eq. (4), PDF p. 5."
)

LAW_CONSTANTS   = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE  = {
    "L_inf": {"init": None},
    "K":     {"init": None},
    "t0":    {"init": None},
}


def _vbgf(t: np.ndarray, L_inf: float, K: float, t0: float) -> np.ndarray:
    """VBGF: Lt = L_inf * (1 - exp(-K*(t - t0)))."""
    return L_inf * (1.0 - np.exp(-K * (t - t0)))


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of VBGF per species.

    Deterministic, data-derived initial guesses:
      L_inf0 = 1.1 * max(observed length) — just above observed maximum,
               since L_inf is the asymptotic upper bound (Ricker 1975 §9.6.1, p. 221).
      K0     = 0.3 — mid-range of typical reef-fish Brody coefficient (0.1-2.0).
      t0_0   = 0.0 — neutral start; t0 is often slightly negative for reef fish.

    Single-start TRF is sufficient for the VBGF on typical per-species
    length-at-age data (smooth, well-conditioned loss surface).
    """
    t = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # Data-derived L_inf start
    y_max = float(np.nanmax(y)) if np.any(np.isfinite(y)) else 100.0
    L_inf0 = max(1.1 * y_max, 1.0)
    K0     = 0.3
    t0_0   = 0.0

    # Bounds: L_inf > 0, K > 0, t0 in (-20, 10)
    lo = [1e-3, 1e-4, -20.0]
    hi = [1e5,  10.0,  10.0]
    p0 = [min(max(v, l), h) for v, l, h in zip([L_inf0, K0, t0_0], lo, hi)]

    def residual(p: list) -> np.ndarray:
        return _vbgf(t, p[0], p[1], p[2]) - y

    try:
        sol = least_squares(residual, p0, bounds=(lo, hi),
                            method="trf", max_nfev=6000)
        L_inf, K, t0 = sol.x
        if not np.all(np.isfinite([L_inf, K, t0])):
            raise RuntimeError("non-finite fit result")
        return {"L_inf": float(L_inf), "K": float(K), "t0": float(t0)}
    except Exception:  # noqa: BLE001
        # Return initial guesses on convergence failure
        return {"L_inf": float(p0[0]), "K": float(p0[1]), "t0": float(p0[2])}


def predict(X: np.ndarray, L_inf: float, K: float, t0: float) -> np.ndarray:
    """Von Bertalanffy length at age.

    X: (n, 1) array — column [Agei] (age in years, from USED_INPUTS).
    L_inf, K, t0: per-species LOCAL_FITTABLE values from fit().
    Returns predicted mean total length in mm.
    """
    t = np.asarray(X[:, 0], dtype=float)
    return _vbgf(t, L_inf, K, t0)
