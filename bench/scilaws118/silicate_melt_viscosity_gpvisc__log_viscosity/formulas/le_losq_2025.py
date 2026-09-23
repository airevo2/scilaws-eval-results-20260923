"""Vogel-Tammann-Fulcher (VFT) viscosity equation per Le Losq et al. (2025).

Le Losq C., Ferraina C., Sossi P. A., Boukaré C.-E. (2025).
"A general machine learning model of aluminosilicate melt viscosity and its
application to the surface properties of dry lava planets."
Earth and Planetary Science Letters, 656, 119287.
DOI: 10.1016/j.epsl.2025.119287 | arXiv:2409.20235v2, 14 Oct 2024.

The formula (Eq. 2, PDF page 4 of arXiv preprint v2):

    log η = A + B / (T - C)

where:
  log η  — log10 dynamic viscosity in Pa·s (the SR target)
  T      — temperature in K
  A      — "a common adjustable parameter" (Le Losq 2025, p. 4); weakly
            composition-dependent, but treated here as per-cluster fittable
            alongside B and C.
  B      — adjustable parameter (units K) that "depends on melt composition
            X and pressure P" (p. 4); related to activation energy.
  C      — adjustable parameter (units K) that "depends on melt composition
            X and pressure P" (p. 4); the ideal glass transition temperature
            (Kauzmann temperature).

LAW_CONSTANTS
-------------
None. The VFT form (Eq. 2) is itself the scientific claim; A, B, C are all
per-cluster fittable. The paper does not publish a single universal set of
(A, B, C) applicable across compositions — it explicitly states they depend
on composition X. Therefore LAW_CONSTANTS = {}.

OTHER_CONSTANTS
---------------
None. The formula is dimensionally clean and uses only SI inputs (T in K,
log10 for viscosity). No unit-conversion or structural constants are needed.

LOCAL_FITTABLE
--------------
Per cluster (= per melt composition), fit:
  A — intercept (dimensionless log10 Pa·s at T -> inf). Init: -4.0
      (physically A ~ -4 to -5 for silicate melts; see Le Losq 2025 Fig. 1
      discussion and VFT literature [10] cited in the paper).
  B — pseudo-activation temperature (K). Init: 5000.0
      (typical silicate melt values reported in viscosity literature).
  C — Kauzmann / ideal glass transition temperature (K). Init: 200.0
      (typical range 0-700 K for silicate melts; Le Losq 2025 p. 4 notes
      C "depends on melt composition X and pressure P").

Type II task
------------
Criterion (a) fires: this is a per-cluster-fit formula. Each melt
composition (cluster) has its own (A, B, C). The SR challenge is to
recover the functional form A + B/(T-C) from composition-labelled data.

Column mapping (paper notation -> released CSV column)
------------------------------------------------------
  log η   -> log_viscosity (target, col 0)
  T (K)   -> T (col 1)
  X (mol%) -> sio2, tio2, al2o3, feo, fe2o3, mno, mgo, cao, na2o, k2o,
              p2o5, h2o (cols 2-13; not used in this formula — VFT uses T only
              after A, B, C are cluster-fit)

Caveats
-------
1. The fit is per-cluster (per composition). The formula uses only T as input
   to predict() after A, B, C have been fitted; composition columns are not
   consumed by predict() — they vary within a cluster only if multiple named
   compositions are aggregated, which they are not here.
2. Numerical stability: C must satisfy T > C for all T in the cluster. The
   optimizer is constrained to C < min(T_cluster) - 10. A constraint on C
   bounds is implemented in fit() via penalty.
3. Multi-start: three initial guesses cover different viscosity regimes
   (high-silica fragile, low-silica strong, intermediate glasses).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

USED_INPUTS  = ["T"]
PAPER_REF    = "summary_formula+dataset_le_losq_2025.md"
EQUATION_LOC = "Eq. (2), PDF p. 4 — arXiv:2409.20235v2 (Le Losq 2025)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS   = {}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE  = {
    "A": {"init": [-4.0, -3.0, -5.0]},   # multi-start
    "B": {"init": [5000.0, 3000.0, 8000.0]},
    "C": {"init": [200.0, 50.0, 400.0]},
}


def _vft(T: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    """Core VFT: log η = A + B / (T - C)."""
    denom = T - C
    # Guard against division by zero or negative denom
    safe = np.where(denom > 1e-3, denom, 1e-3)
    return A + B / safe


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Fit (A, B, C) per cluster by minimising squared-log-viscosity residual.

    X_fit : (n, 1) — column 0 is T (K)
    y_fit : (n,)   — log10 viscosity
    Returns dict with keys matching LOCAL_FITTABLE.
    """
    T_arr = np.asarray(X_fit[:, 0], dtype=float)
    y_obs = np.asarray(y_fit, dtype=float)
    T_min = float(T_arr.min())

    # C must be < T_min to avoid singularity; add soft constraint as penalty
    def loss(params: np.ndarray) -> float:
        A_v, B_v, C_v = float(params[0]), float(params[1]), float(params[2])
        penalty = 0.0
        if B_v < 0:
            penalty += 1e6 * (B_v ** 2)
            B_v = 1e-3
        if C_v >= T_min - 1.0:
            penalty += 1e6 * ((C_v - (T_min - 1.0)) ** 2)
            C_v = T_min - 1.0
        y_pred = _vft(T_arr, A_v, B_v, C_v)
        return float(np.sum((y_pred - y_obs) ** 2)) + penalty

    # Multi-start
    best_result = None
    best_loss = np.inf
    starts_A = LOCAL_FITTABLE["A"]["init"]
    starts_B = LOCAL_FITTABLE["B"]["init"]
    starts_C = LOCAL_FITTABLE["C"]["init"]
    for A0, B0, C0 in zip(starts_A, starts_B, starts_C):
        # Clip C0 to be safely below T_min
        C0_safe = min(float(C0), T_min - 10.0)
        x0 = np.array([A0, B0, C0_safe], dtype=float)
        try:
            res = minimize(
                loss,
                x0=x0,
                method="Nelder-Mead",
                options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 5000},
            )
            if res.fun < best_loss:
                best_loss = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None:
        # Fallback: return plausible defaults
        return {"A": -4.0, "B": 5000.0, "C": min(200.0, T_min - 10.0)}

    A_fit, B_fit, C_fit = float(best_result.x[0]), float(best_result.x[1]), float(best_result.x[2])
    # Clip C to be physically valid
    C_fit = min(C_fit, T_min - 1.0)
    return {"A": A_fit, "B": B_fit, "C": C_fit}


def predict(X: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    """Apply VFT to temperature column.

    X : (n, 1) — column 0 is T (K)
    A, B, C : per-cluster fitted parameters (from LOCAL_FITTABLE)
    Returns log10 viscosity (Pa·s).
    """
    T_arr = np.asarray(X[:, 0], dtype=float)
    return _vft(T_arr, A, B, C)
