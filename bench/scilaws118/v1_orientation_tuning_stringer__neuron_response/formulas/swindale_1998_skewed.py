"""Swindale (1998) skewed von Mises orientation-tuning function — Equation 8.

Swindale, N. V. (1998). Orientation tuning curves: empirical description and
estimation of parameters. *Biological Cybernetics*, 78, 45-56.
DOI: 10.1007/s004220050411.

**Formula — Equation 8 (PDF p. 4, §3.5):**

    phi_r = theta - varphi
    S(theta) = A * exp(k * (cos(2*(phi_r + nu*(cos(2*phi_r) - 1))) - 1))

    where phi_r = theta - varphi (relative orientation, in radians internally),
    nu controls flank asymmetry (skewness), -0.5 <= nu <= 0.5.

**Reduction property (Swindale 1998 p. 4, §3.5):** When nu = 0,
S(theta) reduces identically to the plain von Mises M(theta) = A*exp(k*(cos(2*(theta-varphi))-1)).
Both the flat-topped (Eq. 7) and the skewed (Eq. 8) variants are identical to
the von Mises (Eq. 6) when nu = 0.

**Skewness mechanism:** The term nu*(cos(2*phi_r) - 1) introduces an
orientation-dependent phase shift in the cos argument that displaces the
flanks asymmetrically while keeping the peak (phi_r=0 => cos(0)-1=0) fixed.

**Structural constants (not tunable):**
- The `2` inside cos(2*(...)) and cos(2*phi_r): enforces pi-periodicity.
  Structural algebraic invariant.
- The `-1` in the exponent (outer): normalises peak S(varphi) = A exactly
  when phi_r = 0 (=> cos(0)-1=0 => phase shift = 0 => cos(0)-1 = 0).
  Structural invariant.
- The `-1` in the inner nu*(cos(2*phi_r) - 1): ensures the phase shift is
  zero at the peak phi_r = 0 and symmetric for skewness.  Structural.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None.  A, k, varphi, nu are per-neuron fits; no universal numerical values
are published in Swindale (1998) for the skewed variant.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)  The `2` values (pi-periodicity) and `-1` values (peak normalisation
and phase-shift centering) are structural algebraic invariants coded as literals.

LOCAL_FITTABLE — per-cluster (per-neuron), fitted by fit()
----------------------------------------------------------
- A      : peak amplitude (arb. units, > 0).  init = None: data-derived.
- k      : tuning concentration (dimensionless, > 0.1).  init = None.
- varphi : preferred orientation (degrees, interface units).  init = None.
- nu     : skewness parameter (dimensionless, -0.5 <= nu <= 0.5).
           init = 0.0 (von Mises starting point; best-fit variant in Swindale).
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["orientation_deg"]
PAPER_REF = "summary_formula_swindale_1998.md"
EQUATION_LOC = (
    "Swindale (1998) Eq. 8, PDF p. 4, §3.5 — "
    "S(theta) = A*exp(k*(cos(2*(phi_r + nu*(cos(2*phi_r)-1)))-1)), phi_r = theta - varphi; "
    "nu in [-0.5, 0.5]; reduces to Eq. 6 when nu = 0; best overall fit in Swindale (1998)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "A":      {"init": None},
    "k":      {"init": None},
    "varphi": {"init": None},
    "nu":     {"init": 0.0},
}


def _predict_rad(theta_rad, A, k, varphi_rad, nu):
    """Internal: theta and varphi in radians."""
    phi_r = theta_rad - varphi_rad
    return A * np.exp(k * (np.cos(2.0 * (phi_r + nu * (np.cos(2.0 * phi_r) - 1.0))) - 1.0))


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the skewed von Mises (Eq. 8).

    Swindale (1998) reports the skewed von Mises provides the best overall fit
    to cat V1 data (PDF p. 5, §5.2, Table 1).

    Deterministic data-derived initialisation:
      A0      = max(y) - min(y)
      k0      = 2.0
      varphi0 = orientation of maximum mean response (degrees, mod 180)
      nu0     = 0.0  (von Mises starting point)
    """
    theta_deg = np.asarray(X_fit[:, 0], dtype=float)
    theta_rad = theta_deg * (np.pi / 180.0)
    y = np.asarray(y_fit, dtype=float)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    A0 = max(y_max - y_min, 1e-6)
    k0 = 2.0
    nu0 = 0.0

    unique_deg = np.unique(theta_deg % 180.0)
    if unique_deg.size > 0:
        mean_resp = np.array([
            np.mean(y[np.isclose(theta_deg % 180.0, od, atol=1e-3)])
            for od in unique_deg
        ])
        varphi0_deg = float(unique_deg[np.argmax(mean_resp)])
    else:
        varphi0_deg = 0.0

    varphi0_rad = varphi0_deg * (np.pi / 180.0)

    amp_bound = max(abs(y_max), abs(y_min), 1.0) * 20.0
    lo = [1e-9,  0.1,  0.0, -0.5]
    hi = [amp_bound, 100.0, np.pi,  0.5]
    p0 = [float(np.clip(A0, lo[0], hi[0])),
          float(np.clip(k0, lo[1], hi[1])),
          float(np.clip(varphi0_rad, lo[2], hi[2])),
          float(np.clip(nu0, lo[3], hi[3]))]

    def residual(p):
        return _predict_rad(theta_rad, p[0], p[1], p[2], p[3]) - y

    try:
        sol = least_squares(residual, p0, bounds=(lo, hi),
                            method="trf", max_nfev=6000)
        A, k, varphi_rad, nu = sol.x
        if not np.all(np.isfinite([A, k, varphi_rad, nu])):
            raise RuntimeError("non-finite fit")
        varphi_deg = float(varphi_rad * (180.0 / np.pi))
        return {"A": float(A), "k": float(k), "varphi": varphi_deg, "nu": float(nu)}
    except Exception:                                   # noqa: BLE001
        varphi_deg = float(p0[2] * (180.0 / np.pi))
        return {"A": float(p0[0]), "k": float(p0[1]),
                "varphi": varphi_deg, "nu": float(p0[3])}


def predict(X: np.ndarray, A: float, k: float, varphi: float, nu: float) -> np.ndarray:
    """Swindale (1998) Eq. 8 skewed von Mises orientation tuning.

    S(theta) = A * exp(k * (cos(2*(phi_r + nu*(cos(2*phi_r) - 1))) - 1))
    where phi_r = theta_rad - varphi_rad.

    X: (n_samples, 1) — column [orientation_deg].
    varphi: preferred orientation in degrees.
    nu: skewness parameter in [-0.5, 0.5].
    """
    theta_deg = np.asarray(X[:, 0], dtype=float)
    theta_rad = theta_deg * (np.pi / 180.0)
    varphi_rad = varphi * (np.pi / 180.0)
    return _predict_rad(theta_rad, A, k, varphi_rad, nu)
