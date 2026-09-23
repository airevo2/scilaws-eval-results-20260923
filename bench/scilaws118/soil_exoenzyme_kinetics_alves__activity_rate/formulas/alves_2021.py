"""Michaelis-Menten enzyme kinetics — soil exoenzyme activity rate.

Alves R J E, Callejas I A, Marschmann G L, Mooshammer M, Singh H W,
Whitney B, Torn M S, Brodie E L (2021). Kinetic and temperature
sensitivity properties of soil exoenzymes through the soil profile down
to one-meter depth at a temperate coniferous forest (Blodgett, CA).
Frontiers in Microbiology 12:735282. DOI 10.3389/fmicb.2021.735282.

The Michaelis-Menten rate equation (PDF p. 5, Data Analyses section):

    v = V_max * S / (K_m + S)

where v is the enzyme activity rate (nmol g⁻¹ h⁻¹ dry soil), S is the
substrate concentration (µM), V_max is the maximum reaction velocity at
substrate saturation (nmol g⁻¹ h⁻¹), and K_m is the half-saturation
(Michaelis) constant (µM). Alves et al. (2021) fit this form using the
drm() MM.2 self-starter in the R drc package with Box-Cox variance
stabilisation across 324 combinations of enzyme × depth × temperature ×
soil core replicate (PDF p. 5). The mechanistic derivation of this form
under the quasi-steady-state assumption is Briggs & Haldane (1925).

The formula is the invariant structural claim; V_max and K_m vary per
cluster (enzyme type × depth interval × temperature × soil core).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The MM structural exponents on S are 1 (linear in numerator,
additive in denominator) — these are universal algebraic facts of the
MM derivation, not numerical constants that require citation.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The 1 exponents on S are structural, not data-derived constants.

LOCAL_FITTABLE — per-cluster, fitted by fit() via nonlinear LS
--------------------------------------------------------------
- V_max : Maximum reaction velocity at substrate saturation
          (nmol g⁻¹ h⁻¹ dry soil). Must be > 0. Declines ~96% from
          surface (0–10 cm, ~300–3000 nmol g⁻¹ h⁻¹) to deep soil
          (80–90 cm, ~10–100 nmol g⁻¹ h⁻¹).
- K_m   : Half-saturation (Michaelis) constant (µM). Must be > 0.
          Declines ~86% with depth (surface ~10–200 µM; deep < 50 µM).
init = None on both: fit() derives data-driven starting values.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["substrate_conc"]
PAPER_REF = "summary_formula_dataset_alves_2021.md"
EQUATION_LOC = (
    "Alves et al. (2021) PDF p. 5, Data Analyses section — "
    "v = V_max * S / (K_m + S); MM.2 model in R drc package."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "V_max": {"init": None},
    "K_m":   {"init": None},
}


def _mm_rate(S, V_max, K_m):
    """Michaelis-Menten rate: v = V_max * S / (K_m + S)."""
    return V_max * S / (K_m + S)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear least-squares fit of the 2-parameter MM rate law.

    Data-derived starting values:
      V_max0 = max observed activity rate in the cluster. This is a
               valid upper bound since v < V_max at all finite S.
      K_m0   = substrate concentration at which observed activity is
               nearest to V_max0 / 2; if that heuristic is not
               resolvable, default to the median substrate concentration.
    Single-start Levenberg-Marquardt / trust-region is sufficient: the
    MM landscape is smooth, unimodal in (V_max, K_m) on positive data,
    and well-conditioned when the substrate range spans the inflection
    region (which the 8-level design of Alves et al. ensures).
    """
    S = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # Data-derived start
    V_max0 = float(np.max(y)) if y.size > 0 else 1.0
    if V_max0 <= 0:
        V_max0 = 1.0
    half = V_max0 / 2.0
    diff = np.abs(y - half)
    idx_half = int(np.argmin(diff))
    K_m0 = float(S[idx_half]) if S.size > 0 else float(np.median(S))
    if not np.isfinite(K_m0) or K_m0 <= 0:
        K_m0 = float(np.median(S[S > 0])) if np.any(S > 0) else 50.0

    p0 = [V_max0, K_m0]

    # Bounds: V_max > 0; K_m > 0; upper bounds are generous but keep
    # the optimiser away from numerical degenerate regions.
    param_lo = [1e-6, 1e-3]
    param_hi = [1e8,  1e6]

    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, param_lo, param_hi)]

    def residual(p):
        return _mm_rate(S, p[0], p[1]) - y

    try:
        sol = least_squares(residual, p0, bounds=(param_lo, param_hi),
                            method="trf", max_nfev=4000)
        V_max, K_m = sol.x
        if not np.all(np.isfinite([V_max, K_m])):
            raise RuntimeError("non-finite fit")
        return {"V_max": float(V_max), "K_m": float(K_m)}
    except Exception:                                   # noqa: BLE001
        return {"V_max": float(p0[0]), "K_m": float(p0[1])}


def predict(X: np.ndarray, V_max: float, K_m: float) -> np.ndarray:
    """Michaelis-Menten enzyme activity rate.

    X: (n, 1) — column [substrate_conc] in µM.
    Returns activity rate in nmol g⁻¹ h⁻¹ dry soil.
    """
    S = np.asarray(X[:, 0], dtype=float)
    return _mm_rate(S, V_max, K_m)
