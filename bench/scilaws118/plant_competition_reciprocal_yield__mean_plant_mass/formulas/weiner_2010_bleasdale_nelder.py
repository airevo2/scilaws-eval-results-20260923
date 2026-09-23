"""Bleasdale–Nelder two-exponent form as reviewed in Weiner & Freckleton (2010) — mean_plant_mass.

Weiner, J. and Freckleton, R. P. (2010). *Constant Final Yield.*
Annual Review of Ecology, Evolution, and Systematics 41:173–192.
DOI:10.1146/annurev-ecolsys-102209-144642.

Weiner & Freckleton (2010) present the fully general Bleasdale–Nelder
form as their Eq. 5 (PDF p. 5):

    1 / w^theta = A + B * N^phi

Rearranging for mean plant mass w (the benchmark target):

    w = (A + B * N**phi) ** (-1.0 / theta)

where:
  - N is plant density (plants per unit area, the benchmark input).
  - phi (φ) and theta (θ) are independent dimensionless exponents; the
    original Bleasdale & Nelder (1960) paper used a shared exponent theta
    on both sides (phi = theta). Allowing phi ≠ theta permits the total
    yield Y = N*w to have a maximum at finite density (parabolic Y-N
    curve), as noted by Weiner & Freckleton (2010, PDF p. 5).
  - A > 0 is the intraspecific competition-free inverse-mass baseline.
  - B > 0 is the competitive suppression coefficient for the density term.

Special cases:
  - phi = theta = 1 → Shinozaki–Kira (1956) linear reciprocal yield law.
  - phi = theta     → Bleasdale & Nelder (1960) original single-exponent Eq. 3.
  - phi = 1, b ≡ (1/theta) arbitrary → Watkinson (1980) Eq. 7 in BN notation.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. Weiner & Freckleton (2010) review the functional FORM; no universal
numerical values for A, B, phi, or theta are specified.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
None beyond the algebraic rearrangement exponent (-1/theta).

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- A     : competition-free intercept (> 0), units of w^(-theta).
- B     : density-dependent competition coefficient (> 0).
- phi   : density exponent in the reciprocal-mass relationship (> 0).
- theta : mean-plant-mass exponent (> 0); (-1/theta) is the output exponent.

init = None on all four: fit() constructs a data-derived start point.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["density"]
PAPER_REF = "summary_formula_weiner_2010.md"
EQUATION_LOC = (
    "Weiner & Freckleton 2010, Annu. Rev. Ecol. Evol. Syst. 41:173-192, "
    "PDF p. 5, Eq. 5: 1/w^theta = A + B*N^phi; "
    "rearranged to w = (A + B*N**phi)**(-1/theta)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "A":     {"init": None},
    "B":     {"init": None},
    "phi":   {"init": None},
    "theta": {"init": None},
}


def _predict_core(N, A, B, phi, theta):
    """Core formula: w = (A + B*N**phi)**(-1/theta)."""
    N = np.asarray(N, dtype=float)
    arg = A + B * np.power(np.clip(N, 0.0, None), phi)
    arg = np.clip(arg, 1e-300, None)
    return arg ** (-1.0 / theta)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear LS fit of the two-exponent Bleasdale–Nelder form.

    Data-derived initialisation:
      phi0 = theta0 = 0.5  — neutral mid-range start for both exponents.
      A0 = median(w)^(-theta0)  — inferred from competition-free baseline.
      B0 = max(1e-8, (min_w^(-theta0) - A0) / max_N^phi0)
                              — inferred from density-response slope.

    The four-parameter form is under-determined for very sparse density
    series; bounded optimisation keeps the solution physically meaningful.
    """
    N = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    phi0 = 0.5
    theta0 = 0.5
    med_y = float(np.median(y[y > 0])) if np.any(y > 0) else 1.0
    A0 = med_y ** (-theta0)
    if not np.isfinite(A0) or A0 <= 0:
        A0 = 1.0

    max_N = float(np.max(N)) if N.size > 0 else 1.0
    min_y_pos = float(np.min(y[y > 0])) if np.any(y > 0) else med_y * 0.1
    lhs_high = min_y_pos ** (-theta0)
    B0 = max(1e-10, (lhs_high - A0) / (max_N ** phi0 + 1e-30))
    if not np.isfinite(B0) or B0 <= 0:
        B0 = 1e-4

    p0 = [A0, B0, phi0, theta0]

    # bounds: A > 0, B > 0, phi in (0.01, 5), theta in (0.01, 5)
    lo = [1e-12, 1e-12, 0.01, 0.01]
    hi = [1e12,  1e12,  5.0,  5.0]
    p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]

    def residual(params):
        A, B, phi, theta = params
        return _predict_core(N, A, B, phi, theta) - y

    try:
        sol = least_squares(residual, p0, bounds=(lo, hi),
                            method="trf", max_nfev=8000)
        A, B, phi, theta = sol.x
        if not np.all(np.isfinite([A, B, phi, theta])):
            raise RuntimeError("non-finite solution")
        return {
            "A":     float(A),
            "B":     float(B),
            "phi":   float(phi),
            "theta": float(theta),
        }
    except Exception:                          # noqa: BLE001
        return {
            "A":     float(p0[0]),
            "B":     float(p0[1]),
            "phi":   float(p0[2]),
            "theta": float(p0[3]),
        }


def predict(X: np.ndarray, A: float, B: float,
            phi: float, theta: float) -> np.ndarray:
    """Two-exponent Bleasdale–Nelder mean plant mass.

    w = (A + B * N**phi) ** (-1 / theta)

    X : (n, 1) — column [density] (plants per unit area).
    Returns w (mean plant mass, same units as training target).
    """
    N = np.asarray(X[:, 0], dtype=float)
    return _predict_core(N, A, B, phi, theta)
