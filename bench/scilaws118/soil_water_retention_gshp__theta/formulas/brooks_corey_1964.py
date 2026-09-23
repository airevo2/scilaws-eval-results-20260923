"""Brooks and Corey (1964) soil-water retention curve — theta.

Brooks, R. H. and Corey, A. T. (1964). *Hydraulic properties of porous
media*. Hydrology Papers No. 3, Colorado State University, Fort Collins,
March 1964.  Open access: https://mountainscholar.org/bitstream/handle/
10217/61288/HydrologyPapers_n3.pdf

The Brooks-Corey (BC) SWRC relates effective saturation S_e to capillary
pressure P_c via a power law (Eq. 12, PDF p. 11):

    S_e = (P_b / P_c)^lambda,   for P_c >= P_b
    S_e = 1,                     for P_c <  P_b

where S_e = (theta - theta_r) / (theta_s - theta_r).  Re-expressed in
terms of volumetric water content:

    theta = theta_r + (theta_s - theta_r) * (psi_b / psi)^lambda,
            for psi >= psi_b

    theta = theta_s,   for psi < psi_b

Here psi (= |lab_head_m|, metres) plays the role of P_c, and psi_b
(= bubbling pressure in metres) plays the role of P_b.

Notes on the benchmark context:
- The GSHP dataset (Surya 2022) stores matric suction as positive values
  in `lab_head_m` (metres); this is passed directly as psi.
- The van Genuchten (1980) model is the primary SWRC formula for this
  benchmark; Brooks-Corey is included as the classical structural baseline
  and historical predecessor (vG paper cites BC as the limiting case for
  large psi: lambda = n − 1 under Mualem theory, Eq. 32, vG PDF p. 5).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The BC power-law FORM (Eq. 12) is the scientific claim; all
material parameters (theta_r, theta_s, psi_b, lambda) are soil-specific
and are LOCAL_FITTABLE (per-cluster).

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
The structural exponents 2 and 3 appearing in the derived permeability
exponents epsilon = (2 + 3*lambda)/lambda and eta = 2 + 3*lambda (Eqs.
13–14, PDF p. 11) arise from the Burdine pore-geometry integral and are
fixed algebraic factors. These exponents are relevant only for K_rw
prediction; they are NOT used in the theta(psi) formula implemented here.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- theta_r : Residual volumetric water content (m³/m³, >= 0).
             BC paper: S_r (residual saturation) per-sample; typical range
             0.0–0.5 (Appendix III tables, PDF pp. 25–27).
- theta_s : Saturated volumetric water content (m³/m³, > theta_r).
             Corresponds to total pore volume; typical range 0.25–0.65 m³/m³.
- psi_b   : Bubbling pressure (m, > 0).  Below psi_b, pores remain
             fully saturated; above psi_b, drainage begins (Eq. 12,
             PDF p. 11: "P_c < P_b: S_e = 1, theory inapplicable").
- lambda  : Pore-size distribution index (dimensionless, > 0).
             Observed BC values: 1.62–7.30 for unconsolidated media,
             2.89–4.17 for consolidated sandstones (Fig. 2, PDF pp. 12–13).

init = None on all four: fit() builds data-derived starting values.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["lab_head_m"]
PAPER_REF = "summary_formula_brooks_corey_1964.md"
EQUATION_LOC = (
    "Brooks & Corey (1964) Eq. 12, PDF p. 11 — "
    "S_e = (P_b/P_c)^lambda for P_c >= P_b; "
    "theta = theta_r + (theta_s - theta_r) * S_e "
    "(effective saturation definition, PDF p. 7)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "theta_r": {"init": None},
    "theta_s": {"init": None},
    "psi_b":   {"init": None},
    "lam":     {"init": None},
}


def _theta_bc(psi, theta_r, theta_s, psi_b, lam):
    """Brooks-Corey SWRC: theta(psi)."""
    psi = np.abs(psi)
    Se = np.where(psi < psi_b, 1.0, (psi_b / np.where(psi < psi_b, psi_b, psi)) ** lam)
    theta = theta_r + (theta_s - theta_r) * Se
    return np.clip(theta, theta_r, theta_s)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the Brooks-Corey SWRC.

    Data-derived starting values:
      theta_s0 = max(observed theta) — proxy for saturated endpoint.
      theta_r0 = max(0, min(observed theta) - 0.01) — proxy for residual.
      psi_b0   = median(psi among points with theta > 0.9 * theta_s0), or
                 0.1 m if no such points — estimates the bubbling pressure
                 from the wet end of the observed retention curve.
      lam0     = 2.0 — midrange of BC observed values (1.62–7.30).

    Bounds:
      theta_r in [0.0, 0.6], theta_s in (0.01, 1.0],
      psi_b in (1e-4, 50.0] m, lam in (0.01, 20.0].
    """
    psi = np.abs(np.asarray(X_fit[:, 0], dtype=float))
    y   = np.asarray(y_fit, dtype=float)

    theta_s0 = float(np.clip(np.nanmax(y), 0.05, 0.99))
    theta_r0 = float(np.clip(np.nanmin(y) - 0.01, 0.0, theta_s0 - 0.05))

    # Estimate bubbling pressure from wet-end observations
    wet_mask = y > 0.9 * theta_s0
    if wet_mask.any():
        psi_b0 = float(np.median(psi[wet_mask]))
    else:
        psi_b0 = 0.1  # 0.1 m default

    lam0 = 2.0  # midrange of BC observed values (Fig. 2, PDF pp. 12–13)

    p0 = [theta_r0, theta_s0, psi_b0, lam0]

    #              theta_r  theta_s  psi_b   lam
    param_lo = [0.0,     0.01,    1e-4,   0.01]
    param_hi = [0.6,     1.0,     50.0,   20.0]

    # Clip starting point into bounds
    p0 = [min(max(v, lo), hi) for v, lo, hi in zip(p0, param_lo, param_hi)]
    if p0[1] <= p0[0]:
        p0[1] = min(p0[0] + 0.05, param_hi[1])

    def residual(p):
        return _theta_bc(psi, p[0], p[1], p[2], p[3]) - y

    try:
        sol = least_squares(
            residual, p0,
            bounds=(param_lo, param_hi),
            method="trf",
            max_nfev=8000,
        )
        theta_r, theta_s, psi_b, lam = sol.x
        if not np.all(np.isfinite([theta_r, theta_s, psi_b, lam])):
            raise RuntimeError("non-finite fit")
        return {
            "theta_r": float(theta_r),
            "theta_s": float(theta_s),
            "psi_b":   float(psi_b),
            "lam":     float(lam),
        }
    except Exception:                                       # noqa: BLE001
        return {
            "theta_r": float(p0[0]),
            "theta_s": float(p0[1]),
            "psi_b":   float(p0[2]),
            "lam":     float(p0[3]),
        }


def predict(
    X: np.ndarray,
    theta_r: float,
    theta_s: float,
    psi_b: float,
    lam: float,
) -> np.ndarray:
    """Brooks-Corey (1964) SWRC:
    theta = theta_r + (theta_s - theta_r) * (psi_b / psi)^lambda  for psi >= psi_b,
    theta = theta_s                                                  for psi < psi_b.

    X: (N, 1) — column [lab_head_m] (matric suction in metres, positive).
    Returns: (N,) volumetric water content (m³/m³).
    """
    psi = np.asarray(X[:, 0], dtype=float)
    return _theta_bc(psi, theta_r, theta_s, psi_b, lam)
