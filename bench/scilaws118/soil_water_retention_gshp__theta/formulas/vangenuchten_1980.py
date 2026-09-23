"""Van Genuchten (1980) soil-water retention curve — theta.

Van Genuchten, M. Th. (1980). A closed-form equation for predicting the
hydraulic conductivity of unsaturated soils. *Soil Sci. Soc. Am. J.*
44(5): 892–898. DOI:10.2136/sssaj1980.03615995004400050002x.

The van Genuchten SWRC with the Mualem (1976a) constraint m = 1 − 1/n
(Eq. 22, PDF p. 4):

    theta(psi) = theta_r + (theta_s - theta_r) / [1 + (alpha * |psi|)^n]^m

where m = 1 - 1/n.

This is Eqs. 21–22 of van Genuchten (1980).  The benchmark dataset
(Surya/GSHP 2022) stores matric potential in metres as positive values
(matric suction `lab_head_m`); this is passed directly as |psi| without
sign inversion.

Unit convention: alpha is in m⁻¹ (GSHP convention; paper uses cm⁻¹;
conversion alpha [m⁻¹] = 100 × alpha [cm⁻¹]).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The SWRC functional form (Eqs. 21–22) is the scientific claim;
all four material parameters (theta_r, theta_s, alpha, n) are soil-specific
and are LOCAL_FITTABLE (per-cluster).

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
The exponent m = 1 − 1/n is the Mualem constraint (Eq. 22, PDF p. 4);
it is a derived algebraic expression, not a tunable constant.
The exponent 1/2 (tortuosity) in the conductivity expression is not used
here (we predict theta only, not K_r).

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- theta_r : Residual volumetric water content (m³/m³, ≥ 0).
             Per-cluster values in GSHP: typically 0.00–0.25 m³/m³.
- theta_s : Saturated volumetric water content (m³/m³, > theta_r).
             Per-cluster values in GSHP: typically 0.25–0.65 m³/m³.
- alpha   : Inverse air-entry pressure (m⁻¹, > 0).
             Per-cluster values in GSHP: typically 0.01–10 m⁻¹.
- n       : Pore-size distribution index (dimensionless, > 1).
             Per-cluster values in GSHP: constrained to [1.0, 7.0]
             by Surya 2022 (PDF p. 5).

init = None on all four: fit() builds data-derived starting values.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["lab_head_m"]
PAPER_REF = "summary_formula_vangenuchten_1980.md"
EQUATION_LOC = (
    "Van Genuchten (1980) Eqs. 21–22, PDF p. 4 — "
    "theta = theta_r + (theta_s - theta_r) / [1 + (alpha*|psi|)^n]^(1-1/n); "
    "Mualem constraint m = 1 - 1/n from Eq. 22, PDF p. 4."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "theta_r": {"init": None},
    "theta_s": {"init": None},
    "alpha":   {"init": None},
    "n":       {"init": None},
}


def _theta(psi, theta_r, theta_s, alpha, n):
    """Van Genuchten SWRC: theta(psi) with m = 1 - 1/n."""
    m = 1.0 - 1.0 / n
    denom = (1.0 + (alpha * np.abs(psi)) ** n) ** m
    theta = theta_r + (theta_s - theta_r) / denom
    return np.clip(theta, theta_r, theta_s)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the vG SWRC.

    Data-derived starting values:
      theta_s0 = max(observed theta) — proxy for saturation endpoint.
      theta_r0 = max(0, min(observed theta) - 0.01) — proxy for residual.
      n0       = 1.5 — midrange for loamy soils; GSHP mean n ranges from
                 ~1.3 (clay) to ~22 (sand), but 1.5 is a conservative safe
                 start when texture is unknown.
      alpha0   = 1.0 m⁻¹ — geometric mean of GSHP alpha distribution
                 (Table 5, Surya 2022: means range from 1.50–3.17 m⁻¹).

    Bounds follow Surya 2022 (PDF p. 5):
      n in [1.001, 7.0], alpha in (0, 100] m⁻¹.
      theta_r in [0, 0.6], theta_s in (theta_r0, 1.0].
    """
    psi = np.asarray(X_fit[:, 0], dtype=float)
    y   = np.asarray(y_fit, dtype=float)

    theta_s0 = float(np.clip(np.nanmax(y), 0.05, 0.99))
    theta_r0 = float(np.clip(np.nanmin(y) - 0.01, 0.0, theta_s0 - 0.05))
    alpha0 = 1.0   # m⁻¹; geometric mean across GSHP texture classes
    n0     = 1.5   # dimensionless; conservative start for unknown texture

    p0 = [theta_r0, theta_s0, alpha0, n0]

    #              theta_r  theta_s  alpha    n
    param_lo = [0.0,     0.01,    1e-4,    1.001]
    param_hi = [0.6,     1.0,     100.0,   7.0  ]

    # Clip starting point into bounds
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, param_lo, param_hi)]
    # Ensure theta_s0 > theta_r0 in p0
    if p0[1] <= p0[0]:
        p0[1] = min(p0[0] + 0.05, param_hi[1])

    def residual(p):
        return _theta(psi, p[0], p[1], p[2], p[3]) - y

    try:
        sol = least_squares(
            residual, p0,
            bounds=(param_lo, param_hi),
            method="trf",
            max_nfev=8000,
        )
        theta_r, theta_s, alpha, n = sol.x
        if not np.all(np.isfinite([theta_r, theta_s, alpha, n])):
            raise RuntimeError("non-finite fit")
        return {
            "theta_r": float(theta_r),
            "theta_s": float(theta_s),
            "alpha":   float(alpha),
            "n":       float(n),
        }
    except Exception:                                       # noqa: BLE001
        return {
            "theta_r": float(p0[0]),
            "theta_s": float(p0[1]),
            "alpha":   float(p0[2]),
            "n":       float(p0[3]),
        }


def predict(
    X: np.ndarray,
    theta_r: float,
    theta_s: float,
    alpha: float,
    n: float,
) -> np.ndarray:
    """Van Genuchten (1980) SWRC: theta = theta_r + (theta_s - theta_r) /
    [1 + (alpha * |psi|)^n]^(1 - 1/n).

    X: (N, 1) — column [lab_head_m] (matric suction in metres, positive).
    Returns: (N,) volumetric water content (m³/m³).
    """
    psi = np.asarray(X[:, 0], dtype=float)
    return _theta(psi, theta_r, theta_s, alpha, n)
