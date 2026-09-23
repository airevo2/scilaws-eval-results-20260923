"""Modified Briere thermal performance curve — OD600 at 24 h vs temperature.

Cruz-Loya, M., Tekin, E., Kang, T.M., Cardona, N., Lozano-Huntelman, N.,
Rodriguez-Verdugo, A., Savage, V.M., & Yeh, P.J. (2021). Antibiotics Shift
the Temperature Response Curve of Escherichia coli Growth. *mSystems*, 6(4),
e00228-21. DOI:10.1128/mSystems.00228-21.

Reparametrised form used for fitting (PDF p. 14, Materials and Methods):

    g(T) = g_max * [((T - T_min) / (alpha * (T_max - T_min)))^alpha
                  * ((T_max - T) / ((1 - alpha) * (T_max - T_min)))^(1 - alpha)]^s

    g(T) = 0   for T <= T_min  or  T >= T_max

The power-law form (PDF p. 13):
    g(T) = c * (T - T_min)^a * (T_max - T)^b
with a = alpha * s, b = (1 - alpha) * s, c absorbed into g_max.

The optimal temperature (PDF p. 13):
    T_opt = alpha * T_max + (1 - alpha) * T_min

LAW_CONSTANTS — cross-group invariant constants (held fixed at eval)
-------------------------------------------------------------------
None — legitimately empty.  This law has NO constant that stays invariant
across the 79 antibiotic backgrounds: all five shape parameters (g_max,
T_min, T_max, alpha, s) vary per antibiotic background — the paper's central
finding is that antibiotics SHIFT the temperature response curve — so they are
LOCAL_FITTABLE (re-fit per cluster by fit()), not fixed coefficients.  Reported
as Bayesian posterior means per condition in Cruz-Loya 2021 (PDF p. 14).  (Empty
LAW here means "no cross-group invariant", not a hidden defining coefficient.)

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)  The exponents are determined by alpha and s; the normalisation
(T_max - T_min) is the thermal range. No extra tunable constant.

LOCAL_FITTABLE — per-cluster, fitted by fit() via multi-start nonlinear LS
---------------------------------------------------------------------------
- g_max : peak OD600 at T_opt (positive); OD600 at 24 h.
- T_min : minimum temperature supporting growth (degC);
          constrained below the lowest observed temperature.
- T_max : maximum temperature supporting growth (degC);
          constrained above the highest observed temperature.
- alpha : relative position of T_opt between T_min and T_max
          (dimensionless, 0 < alpha < 1); alpha = a / (a + b).
- s     : steepness / sharpness exponent (dimensionless, s > 0); s = a + b.
          Upper bound 200.0: empirical scan of all 79 antibiotic backgrounds
          in Cruz-Loya 2021 shows s up to ~150 for conditions with very sharp
          thermal transitions near 46 degC.  The published Bayesian framework
          uses a HalfNormal prior (not a hard upper bound) that allows large s.

  init = [1.0, 3.0, 8.0, 20.0, 50.0] for s (multi-start; others init = None).
  Multi-start over s is necessary because the Briere loss landscape has a wide
  shallow plateau in s above ~20 for sharply-transitioning conditions, and a
  single starting value can prematurely terminate at a local minimum.

USED_INPUTS: ["T"]  (temperature in degC).
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["T"]
PAPER_REF   = "summary_formula_dataset_cruz_loya_2021.md"
EQUATION_LOC = (
    "Cruz-Loya et al. (2021) mSystems 6(4) e00228-21 — "
    "PDF p. 14 (Materials and Methods): reparametrised modified Briere "
    "g(T) = g_max * [((T-T_min)/(alpha*(T_max-T_min)))^alpha "
    "* ((T_max-T)/((1-alpha)*(T_max-T_min)))^(1-alpha)]^s; "
    "zero outside [T_min, T_max].  "
    "Power-law form PDF p. 13: g(T) = c*(T-T_min)^a*(T_max-T)^b."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "g_max": {"init": None},
    "T_min": {"init": None},
    "T_max": {"init": None},
    "alpha": {"init": None},
    "s":     {"init": [1.0, 3.0, 8.0, 20.0, 50.0]},
}


def _modified_briere(
    T: np.ndarray,
    g_max: float,
    T_min: float,
    T_max: float,
    alpha: float,
    s: float,
) -> np.ndarray:
    """Vectorised modified Briere reparametrised form with zero clamping."""
    T = np.asarray(T, dtype=float)
    result = np.zeros_like(T)
    mask = (T > T_min) & (T < T_max)
    if not mask.any():
        return result
    Tm = T[mask]
    drange = T_max - T_min
    if drange <= 0 or alpha <= 0 or alpha >= 1 or s <= 0:
        return result
    # Normalised factors — strictly positive inside mask
    left  = (Tm - T_min)  / (alpha         * drange)
    right = (T_max - Tm)  / ((1.0 - alpha) * drange)
    val = g_max * (left ** alpha * right ** (1.0 - alpha)) ** s
    result[mask] = val
    return result


def _fit_single_start(
    T: np.ndarray,
    y: np.ndarray,
    s0: float,
    g_max0: float,
    T_min0: float,
    T_max0: float,
    alpha0: float,
    lo: list,
    hi: list,
) -> tuple[np.ndarray, float]:
    """Single-start TRF fit; returns (params, cost)."""
    p0_s = [g_max0, T_min0, T_max0, alpha0, s0]
    p0_s = [float(np.clip(v, l, h)) for v, l, h in zip(p0_s, lo, hi)]

    def residual(p):
        g_max, T_min, T_max, alpha, s = p
        return _modified_briere(T, g_max, T_min, T_max, alpha, s) - y

    sol = least_squares(residual, p0_s, bounds=(lo, hi),
                        method="trf", max_nfev=5000)
    return sol.x, sol.cost


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Multi-start bounded nonlinear LS fit of the modified Briere curve.

    Data-derived fixed starting values (shared across all s starts):
      T_min0 = min(T) - 3
      T_max0 = max(T) + 3
      alpha0 = 0.70     (close to WT estimate from Cruz-Loya 2021)
      g_max0 = max(y_fit)

    s starting values: [1.0, 3.0, 8.0, 20.0, 50.0] — spans low to high
    sharpness. The best-cost solution across all starts is returned.

    Bounds:
      g_max: (1e-6,  10.0)
      T_min: (-10,   min(T) - 0.5)
      T_max: (max(T) + 0.5, 80.0)
      alpha: (0.01,  0.99)
      s:     (0.1,  200.0)   — upper bound raised from 20 to 200 based on
             empirical analysis of all 79 Cruz-Loya conditions; see docstring.

    Note on T_min identifiability: when OD at the lowest measured T is near zero
    (e.g., 0.003 at 22 degC), T_min is weakly identified — any value in [-10, 21]
    degC produces similar residuals.  The lower bound -10 degC is physically
    motivated (liquid water still present; E. coli cannot grow below 0 degC, and
    all experiments are in liquid LB).  Some clusters will still push T_min near
    -10; this is a data-constrained property, not a fit degeneration — prediction
    performance at the observed T levels is unaffected by T_min uncertainty when
    all observed T > T_min.
    """
    T = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    T_obs_min = float(np.min(T))
    T_obs_max = float(np.max(T))
    g_max0    = float(np.max(y)) if float(np.max(y)) > 0 else 1.0
    T_min0    = T_obs_min - 3.0
    T_max0    = T_obs_max + 3.0
    alpha0    = 0.70

    # Bounds: [g_max, T_min, T_max, alpha, s]
    # T_min lower bound: -10 degC (physical: liquid water, no growth below 0 degC)
    lo = [1e-6, -10.0,           T_obs_max + 0.5,  0.01,   0.1]
    hi = [10.0,  T_obs_min - 0.5,  80.0,            0.99, 200.0]

    s_starts = [1.0, 3.0, 8.0, 20.0, 50.0]
    best_params = None
    best_cost   = np.inf

    for s0 in s_starts:
        try:
            params, cost = _fit_single_start(
                T, y, s0, g_max0, T_min0, T_max0, alpha0, lo, hi)
            if np.all(np.isfinite(params)) and cost < best_cost:
                best_cost   = cost
                best_params = params.copy()
        except Exception:                   # noqa: BLE001
            pass

    if best_params is None or not np.all(np.isfinite(best_params)):
        # Fallback: return data-derived initial values
        p0_fb = [
            g_max0,
            float(np.clip(T_min0, lo[1], hi[1])),
            float(np.clip(T_max0, lo[2], hi[2])),
            float(np.clip(alpha0,  lo[3], hi[3])),
            float(np.clip(8.0,     lo[4], hi[4])),
        ]
        return {
            "g_max": p0_fb[0],
            "T_min": p0_fb[1],
            "T_max": p0_fb[2],
            "alpha": p0_fb[3],
            "s":     p0_fb[4],
        }

    g_max, T_min, T_max, alpha, s = best_params
    return {
        "g_max": float(g_max),
        "T_min": float(T_min),
        "T_max": float(T_max),
        "alpha": float(alpha),
        "s":     float(s),
    }


def predict(
    X: np.ndarray,
    g_max: float,
    T_min: float,
    T_max: float,
    alpha: float,
    s: float,
) -> np.ndarray:
    """Modified Briere thermal performance curve for OD24h.

    X: (n, 1) ndarray — column [T] (temperature in degC).
    Returns predicted OD600 at 24 h; zero outside [T_min, T_max].
    """
    T = np.asarray(X[:, 0], dtype=float)
    return _modified_briere(T, g_max, T_min, T_max, alpha, s)
