"""Manning's equation — mean flow velocity in open channels (SI form).

Jobson, H. E., and Froehlich, D. C. (1988). *Basic Hydraulic Principles of
Open-Channel Flow.* U.S. Geological Survey Open-File Report 88-707, Reston,
Virginia. URL: https://pubs.usgs.gov/of/1988/0707/report.pdf

Lesson 7 (PDF pp. 59-62) derives the resistance equations from the force
balance on a prismatic channel control volume. Manning's equation (eq. 7-8,
PDF p. 61) in SI units is

    V = (1/n) * R^(2/3) * S_f^(1/2)

where V is the cross-section mean velocity (m/s), R is the hydraulic radius
(m), and S_f is the friction slope (dimensionless, approximated by bed slope
S_o under uniform flow). The structural exponents 2/3 and 1/2 are empirically
established from Chezy-Manning hydraulic theory and are unchanged across all
applications (PDF p. 61, eq. 7-8). The SI prefactor is identically 1.0 (as
opposed to 1.49 in US customary; PDF p. 53 conversion discussion).

The IFMHA dataset (Erfani et al. 2024) supplies measurements in US customary
units; PROVENANCE.md documents the SI conversion applied by prep_data.py:
  - chan_area (ft^2)   -> A_m2 (m^2)  via x 0.09290304
  - chan_width (ft)    -> W_m  (m)     via x 0.3048
  - R_m = A_m2 / W_m  (hydraulic depth, approximation of R for wide channels)
  - SLOPE (dimensionless ft/ft) is unchanged

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The exponents 2/3 and 1/2 are universal structural factors of the
Manning-Chezy derivation, not empirically tuned coefficients published for a
specific material or measurement campaign. The dimensional prefactor k_n is a
unit-conversion GIVEN (see OTHER_CONSTANTS), not a fitted coefficient.

OTHER_CONSTANTS — universal / unit-conversion factor
----------------------------------------------------
- k_n : the Manning dimensional conversion factor. = 1.0 in SI (V in m/s,
        R in m), = 1.486 (≈ 1.49) in US customary. It is the fixed
        dimensional patch that gives Manning's n the same numerical value in
        both unit systems — "the conversion is included in the formula"
        (Jobson 1988, PDF p. 53). A unit conversion, NOT a paper-fitted
        coefficient → OTHER (MANUAL §1 kind-a; mirrors the sibling
        Darcy-Weisbach baseline that boxes its standard-gravity given g).
        Read from the dict in the body (gold style). The structural exponents
        2/3 and 1/2 stay as LITERALS in the formula body.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- n : Manning's roughness coefficient (s/m^(1/3), dimensionless in common
      usage). Varies by channel material, vegetation, alignment, and
      irregularity; the USGS Cowan procedure (Jobson 1988 eq. 12-3, PDF p. 80)
      establishes material-class lower bounds: n_0 >= 0.020 for earth,
      >= 0.024 for fine gravel, >= 0.025 for rock cuts. Typical ranges
      (Table 12-1, PDF pp. 86-88): 0.025-0.033 (clean straight stream),
      0.040-0.150 (weedy/flood plain), 0.010-0.050 (lined channels).
      Fit per USGS gauge station (COMID) for the Type II benchmark.

init = None: fit() builds its own data-derived start (median of 1/n from
observed V and computed R^(2/3)*S_f^(1/2)).
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["R_m", "SLOPE"]
PAPER_REF = "summary_formula_jobson_1988.md"
EQUATION_LOC = (
    "Jobson & Froehlich (1988) eq. 7-8, PDF p. 61 — V = (1/n)*R^(2/3)*S_f^(1/2); "
    "SI prefactor = 1.0 (conversion discussed PDF p. 53)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    "k_n": 1.0,   # Manning dimensional conversion factor: 1.0 (SI) / 1.486 (US);
                  # "the conversion is included in the formula" (Jobson 1988 p. 53).
                  # A unit conversion, not a fitted coefficient (MANUAL §1 kind-a).
}
LOCAL_FITTABLE = {
    "n": {"init": None},
}

_K_N = OTHER_CONSTANTS["k_n"]   # alias of the boxed unit-conversion factor


def _velocity(R_m, SLOPE, n):
    """Manning's equation (SI): V = (k_n/n) * R^(2/3) * S_f^(1/2)."""
    R_safe = np.clip(R_m, 0.0, None)
    S_safe = np.clip(SLOPE, 0.0, None)
    return (_K_N / n) * R_safe ** (2.0 / 3.0) * S_safe ** 0.5


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded 1-parameter least-squares fit for Manning's n.

    Deterministic, data-derived start:
      n0 = 1 / median(V_obs / (R^(2/3) * S_f^(1/2)))
    where the median is taken over rows with positive R, S_f, and V_obs,
    giving a robust initial estimate of n consistent with the observed data.

    Single-parameter Manning fit is well-conditioned; one-start is sufficient.
    """
    R_m   = np.asarray(X_fit[:, 0], dtype=float)
    SLOPE = np.asarray(X_fit[:, 1], dtype=float)
    y     = np.asarray(y_fit, dtype=float)

    # Compute hydraulic term for each row; mask non-positive entries.
    R_safe = np.clip(R_m, 0.0, None)
    S_safe = np.clip(SLOPE, 0.0, None)
    h = R_safe ** (2.0 / 3.0) * S_safe ** 0.5   # hydraulic conveyance term

    # Data-derived start: solve V = h/n => n = h/V pointwise, take median.
    valid = (y > 0) & (h > 0)
    if valid.any():
        n0 = float(np.median(h[valid] / y[valid]))
    else:
        n0 = 0.035   # generic mid-range value if no valid rows

    # Bounds informed by USGS Cowan material classes (Jobson 1988 Table 12-1):
    #   lower: 0.010 (smooth lined channels)
    #   upper: 0.300 (very rough natural channels / heavy vegetation)
    # Widened slightly to 0.008-0.500 so the optimiser can reach edge cases
    # without hard-clamping at the boundary.
    n_lo, n_hi = 0.008, 0.500
    n0 = float(np.clip(n0, n_lo, n_hi))

    def residual(p):
        return _velocity(R_m, SLOPE, p[0]) - y

    try:
        sol = least_squares(
            residual, [n0],
            bounds=([n_lo], [n_hi]),
            method="trf",
            max_nfev=4000,
        )
        n_fit = float(sol.x[0])
        if not np.isfinite(n_fit):
            raise RuntimeError("non-finite fit")
        return {"n": n_fit}
    except Exception:                               # noqa: BLE001
        return {"n": n0}


def predict(X: np.ndarray, n: float) -> np.ndarray:
    """Manning's equation (SI): V = (1/n) * R^(2/3) * S_f^(1/2).

    X: (n_rows, 2) — columns [R_m, SLOPE].
    Returns V_mps (m/s), clipped to >= 0.
    """
    R_m   = np.asarray(X[:, 0], dtype=float)
    SLOPE = np.asarray(X[:, 1], dtype=float)
    V = _velocity(R_m, SLOPE, n)
    return np.clip(V, 0.0, None)
