"""Hill kinetics — generalised cooperative enzyme activity rate.

Hill A. V. (1910). The possible effects of the aggregation of the molecules
of haemoglobin on its dissociation curves. Journal of Physiology 40 (Suppl.
iv-vii). DOI 10.1113/jphysiol.1910.sp001386.

The Hill equation for enzyme kinetics:

    v = V_max * S^n / (K_h^n + S^n)

where v is the enzyme activity rate (nmol g^-1 h^-1 dry soil), S is the
substrate concentration (uM), V_max is the maximum velocity (nmol g^-1 h^-1),
K_h is the half-saturation constant at which v = V_max / 2 (uM), and n is
the Hill coefficient (cooperativity exponent, n = 1 recovers Michaelis-
Menten exactly).

Relationship to Michaelis-Menten: for n = 1 the Hill equation is identical
to the 2-parameter MM formula v = V_max * S / (K_m + S). A 3-parameter Hill
fit generalises MM to allow cooperative (n > 1) or anti-cooperative (n < 1)
substrate binding. Alves et al. (2021) used the standard 2-parameter MM
(n fixed at 1), so the Hill baseline is a structural alternative that nests
the MM baseline as a special case.

The Hill equation is applied to extracellular soil enzyme kinetics in the
ecology literature (Sinsabaugh & Follstad Shah 2012, Ecology Letters 15:
1305-1317; DOI 10.1111/j.1461-0248.2012.01861.x) as a more flexible
functional form when cooperativity is suspected, particularly for leucine
aminopeptidase (LAP) and acid phosphatase (AP) at some depth-temperature
combinations where substrate saturation curves deviate from strict
hyperbolic form. This is consistent with observed n values slightly
different from 1 in some clusters of the Alves et al. dataset.

LAW_CONSTANTS -- cross-cluster invariants
-----------------------------------------
None. The functional form (S^n in both numerator and denominator) is
structural; the structural exponents 'n' appear identically in both
terms — this is algebraic self-similarity, not a numerically fixed constant.

OTHER_CONSTANTS -- universal factors
-------------------------------------
(empty.)

LOCAL_FITTABLE -- per-cluster, fitted by fit()
-----------------------------------------------
- V_max : Maximum reaction velocity at substrate saturation
          (nmol g^-1 h^-1 dry soil). Must be > 0.
- K_h   : Hill half-saturation constant (uM). Must be > 0.
          Equal to K_m when n = 1.
- n     : Hill cooperativity exponent (dimensionless). Must be > 0.
          n = 1 -> Michaelis-Menten (hyperbolic); n > 1 -> positive
          cooperativity (sigmoidal); n < 1 -> negative cooperativity.
          Soil hydrolase exponents are typically near 1 (Sinsabaugh 2012).
init = None on all three: fit() uses data-derived starting values.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["substrate_conc"]
PAPER_REF = "summary_formula_dataset_alves_2021.md"
EQUATION_LOC = (
    "Hill (1910) Journal of Physiology 40(Suppl.) p. vi — "
    "v = V_max * S^n / (K_h^n + S^n); n=1 recovers MM.2 of Alves et al. 2021."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "V_max": {"init": None},
    "K_h":   {"init": None},
    "n":     {"init": None},
}


def _hill_rate(S, V_max, K_h, n):
    """Hill enzyme rate: v = V_max * S^n / (K_h^n + S^n)."""
    Sn = np.power(np.abs(S), n)
    Kn = np.power(np.abs(K_h), n)
    return V_max * Sn / (Kn + Sn)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear least-squares fit of the 3-parameter Hill equation.

    Data-derived starting values:
      V_max0 = max observed activity.
      K_h0   = substrate concentration nearest to V_max0 / 2.
      n0     = 1.0 (Michaelis-Menten start; LM will move it if warranted).
    """
    S = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    V0 = float(np.max(y)) if y.size > 0 else 1.0
    if V0 <= 0:
        V0 = 1.0
    half = V0 / 2.0
    diff = np.abs(y - half)
    idx_half = int(np.argmin(diff))
    K0 = float(S[idx_half]) if S.size > 0 else float(np.median(S))
    if not np.isfinite(K0) or K0 <= 0:
        K0 = float(np.median(S[S > 0])) if np.any(S > 0) else 50.0
    n0 = 1.0

    p0 = [V0, K0, n0]
    param_lo = [1e-6, 1e-3, 0.05]
    param_hi = [1e8,  1e6,  10.0]
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, param_lo, param_hi)]

    def residual(p):
        return _hill_rate(S, p[0], p[1], p[2]) - y

    try:
        sol = least_squares(residual, p0, bounds=(param_lo, param_hi),
                            method="trf", max_nfev=6000)
        V_max, K_h, n = sol.x
        if not np.all(np.isfinite([V_max, K_h, n])):
            raise RuntimeError("non-finite fit")
        return {"V_max": float(V_max), "K_h": float(K_h), "n": float(n)}
    except Exception:                                   # noqa: BLE001
        return {"V_max": float(p0[0]), "K_h": float(p0[1]), "n": float(p0[2])}


def predict(X: np.ndarray, V_max: float, K_h: float, n: float) -> np.ndarray:
    """Hill enzyme activity rate.

    X: (m, 1) — column [substrate_conc] in uM.
    Returns activity rate in nmol g^-1 h^-1 dry soil.
    """
    S = np.asarray(X[:, 0], dtype=float)
    return _hill_rate(S, V_max, K_h, n)
