"""Bleasdale–Nelder (1960) generalised reciprocal yield law — mean_plant_mass.

Bleasdale, J. K. A. and Nelder, J. A. (1960). *Plant population and crop
yield.* Nature 188:342. DOI:10.1038/188342a0.

The paper generalises the Shinozaki–Kira (1956) linear reciprocal yield
law 1/w = Ap + B (their Eq. 1) to a nonlinear form derived from
Richards's (1959) generalised growth equation (PDF p. 1, Eq. 3):

    1 / w^theta = A * p^theta + B

Rearranging for mean plant mass w (the benchmark target):

    w = (A * p**theta + B) ** (-1.0 / theta)

where:
  - p is plant density (plants per unit area, the benchmark input)
  - theta (θ) is a positive dimensionless exponent, "usually less than
    unity" (PDF p. 1); theta = 1 reduces to the Shinozaki–Kira special
    case 1/w = Ap + B (Eq. 1 of this paper).
  - A > 0 is the competitive suppression coefficient.
  - B > 0 is the competition-free baseline inverse-mass term; the
    isolated-plant mass (p→0) is B^(-1/theta).

Note on notation: The 1960 paper uses a single shared exponent theta on
both w^theta and p^theta. Weiner & Freckleton (2010) re-express the
general form with two independent exponents phi and theta (their Eq. 5,
1/w^theta = A + B*N^phi); that two-exponent generalisation is implemented
separately in `weiner_2010_bleasdale_nelder.py`.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The 1960 paper proposes the functional FORM only. No universal
numerical values for A, B, or theta are given; all three must be fitted to
each species/experiment dataset.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
The exponent (-1/theta) in the rearrangement and the exponent theta on
p^theta are the structural mathematical consequence of Richards's
generalised growth equation; they are not tunable empirical constants.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- A     : competitive suppression coefficient (> 0). Units absorbed by
          theta; larger A = stronger density-dependent yield loss.
- B     : competition-free inverse-mass baseline (> 0). Sets the
          isolated-plant mass: w_isolated = B^(-1/theta).
- theta : nonlinearity exponent (> 0; typically in (0, 1]). Controls
          curvature of w vs. p; theta = 1 is the Shinozaki–Kira case.

init = None on all three: fit() constructs a data-derived start point.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["density"]
PAPER_REF = "summary_formula_bleasdale_1960.md"
EQUATION_LOC = (
    "Bleasdale & Nelder 1960, Nature 188:342, PDF p. 1, Eq. 3: "
    "1/w^theta = A*p^theta + B; rearranged to w = (A*p**theta + B)**(-1/theta)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "A":     {"init": None},
    "B":     {"init": None},
    "theta": {"init": None},
}


def _predict_core(p, A, B, theta):
    """Core formula: w = (A*p**theta + B)**(-1/theta)."""
    p = np.asarray(p, dtype=float)
    arg = A * np.power(np.clip(p, 0.0, None), theta) + B
    arg = np.clip(arg, 1e-300, None)          # keep strictly positive
    return arg ** (-1.0 / theta)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear LS fit of Bleasdale–Nelder (1960) Eq. 3.

    Data-derived initialisation:
      theta0 = 0.5  — mid-range of the "usually less than unity" guidance
                       from the paper; a robust neutral start.
      B0     = 1 / median(w)^theta0  — sets the competition-free intercept
                       so the isolated-plant mass ≈ median observed mass.
      A0     = max(1e-6, (B0 - 1/max(w)^theta0) / max(p)^theta0)
                       — inferred from the constraint that at the highest
                       observed density, w ≈ max(w) is the smallest mass.

    Single-start Levenberg–Marquardt / trust-region is sufficient because
    the Bleasdale–Nelder form is smooth and well-identified from a
    one-sided density series.
    """
    p = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # --- data-derived start ---
    theta0 = 0.5
    med_y = float(np.median(y[y > 0])) if np.any(y > 0) else 1.0
    B0 = med_y ** (-theta0)
    if B0 <= 0 or not np.isfinite(B0):
        B0 = 1.0

    max_p = float(np.max(p)) if p.size > 0 else 1.0
    min_y_pos = float(np.min(y[y > 0])) if np.any(y > 0) else med_y * 0.1
    lhs_high = min_y_pos ** (-theta0)          # 1/w^theta at highest density
    A0 = max(1e-8, (lhs_high - B0) / (max_p ** theta0 + 1e-30))
    if not np.isfinite(A0) or A0 <= 0:
        A0 = 1e-4

    p0 = [A0, B0, theta0]

    # bounds: A > 0, B > 0, theta in (0.01, 5)
    lo = [1e-12, 1e-12, 0.01]
    hi = [1e12,  1e12,  5.0]
    p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]

    def residual(params):
        A, B, theta = params
        return _predict_core(p, A, B, theta) - y

    try:
        sol = least_squares(residual, p0, bounds=(lo, hi),
                            method="trf", max_nfev=6000)
        A, B, theta = sol.x
        if not np.all(np.isfinite([A, B, theta])):
            raise RuntimeError("non-finite solution")
        return {"A": float(A), "B": float(B), "theta": float(theta)}
    except Exception:                          # noqa: BLE001
        return {"A": float(p0[0]), "B": float(p0[1]), "theta": float(p0[2])}


def predict(X: np.ndarray, A: float, B: float, theta: float) -> np.ndarray:
    """Bleasdale–Nelder (1960) mean plant mass.

    w = (A * p**theta + B) ** (-1 / theta)

    X : (n, 1) — column [density] (plants per unit area).
    Returns w (mean plant mass, same units as training target).
    """
    p = np.asarray(X[:, 0], dtype=float)
    return _predict_core(p, A, B, theta)
