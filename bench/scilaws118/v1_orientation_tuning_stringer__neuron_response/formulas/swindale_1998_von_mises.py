"""Swindale (1998) von Mises orientation-tuning function — Equation 6.

Swindale, N. V. (1998). Orientation tuning curves: empirical description and
estimation of parameters. *Biological Cybernetics*, 78, 45-56.
DOI: 10.1007/s004220050411.

**Formula — Equation 6 (PDF p. 3, §3.3):**

    M(theta) = A * exp(k * (cos(2*(theta - phi)) - 1))

where A is the response at preferred orientation phi and k is a concentration
(width) parameter. This is the direct ancestor of Wu & Van Hooser (2025)
Eq. 14 (identical mathematical form).

**Historical note (Swindale 1998 p. 3, §3.3):** The von Mises distribution is
the circular analogue of the Gaussian.  The factor `2` enforces pi-periodicity
(orientation tuning is cyclic over [0, 180) degrees).  The `-1` normalises the
peak so M(phi) = A * exp(0) = A exactly.

**Swindale fitting protocol:** In Swindale (1998), spontaneous activity is
subtracted from raw responses before fitting (PDF p. 4, §4.1); the published
formula therefore does not include a baseline term C.  fit() here operates
directly on benchmark raw data (deconvolved calcium fluorescence); A absorbs
any residual baseline because A in the benchmark context is constrained only
to A > 0 (an overall scale on the bell shape).

**Half-width at half-height (Eq. 6a, PDF p. 3):**

    theta_0.5 = 0.5 * arccos((ln 0.5 + k) / k),  valid for k > -0.5*ln0.5 ~= 0.347.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None.  The FORM M(theta) = A*exp(k*(cos(2*(theta-phi))-1)) is the scientific
claim.  A, k, phi are per-neuron fits; no universal numerical values across
neurons are published in Swindale (1998).

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)  The factor `2` (pi-periodicity) and `-1` (peak normalisation) are
structural algebraic invariants coded as literals.

LOCAL_FITTABLE — per-cluster (per-neuron), fitted by fit()
----------------------------------------------------------
- A   : peak amplitude (arb. units, > 0).  init = None: data-derived.
- k   : tuning concentration (dimensionless, > 0.1).  init = None.
- phi : preferred orientation (degrees, interface units).  init = None.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["orientation_deg"]
PAPER_REF = "summary_formula_swindale_1998.md"
EQUATION_LOC = (
    "Swindale (1998) Eq. 6, PDF p. 3, §3.3 — M(theta) = A*exp(k*(cos(2*(theta-phi))-1)); "
    "half-width Eq. 6a same page."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "A":   {"init": None},
    "k":   {"init": None},
    "phi": {"init": None},
}


def _predict_rad(theta_rad, A, k, phi_rad):
    """Internal: theta and phi in radians."""
    return A * np.exp(k * (np.cos(2.0 * (theta_rad - phi_rad)) - 1.0))


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the Swindale (1998) Eq. 6.

    Deterministic data-derived initialisation:
      A0   = max(y_fit) - min(y_fit)  (above-floor amplitude estimate)
      k0   = 2.0                      (moderate concentration)
      phi0 = orientation of the maximum mean response (degrees, mod 180)
    """
    theta_deg = np.asarray(X_fit[:, 0], dtype=float)
    theta_rad = theta_deg * (np.pi / 180.0)
    y = np.asarray(y_fit, dtype=float)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    A0 = max(y_max - y_min, 1e-6)
    k0 = 2.0

    unique_deg = np.unique(theta_deg % 180.0)
    if unique_deg.size > 0:
        mean_resp = np.array([
            np.mean(y[np.isclose(theta_deg % 180.0, od, atol=1e-3)])
            for od in unique_deg
        ])
        phi0_deg = float(unique_deg[np.argmax(mean_resp)])
    else:
        phi0_deg = 0.0

    phi0_rad = phi0_deg * (np.pi / 180.0)

    amp_bound = max(abs(y_max), abs(y_min), 1.0) * 20.0
    lo = [1e-9,  0.1,  0.0]
    hi = [amp_bound, 100.0, np.pi]
    p0 = [float(np.clip(A0, lo[0], hi[0])),
          float(np.clip(k0, lo[1], hi[1])),
          float(np.clip(phi0_rad, lo[2], hi[2]))]

    def residual(p):
        return _predict_rad(theta_rad, p[0], p[1], p[2]) - y

    try:
        sol = least_squares(residual, p0, bounds=(lo, hi),
                            method="trf", max_nfev=6000)
        A, k, phi_rad = sol.x
        if not np.all(np.isfinite([A, k, phi_rad])):
            raise RuntimeError("non-finite fit")
        return {"A": float(A), "k": float(k),
                "phi": float(phi_rad * (180.0 / np.pi))}
    except Exception:                                   # noqa: BLE001
        return {"A": float(p0[0]), "k": float(p0[1]),
                "phi": float(p0[2] * (180.0 / np.pi))}


def predict(X: np.ndarray, A: float, k: float, phi: float) -> np.ndarray:
    """Swindale (1998) Eq. 6 von Mises orientation tuning.

    M(theta) = A * exp(k * (cos(2*(theta_rad - phi_rad)) - 1))

    X: (n, 1) — column [orientation_deg].
    phi: preferred orientation in degrees.
    """
    theta_deg = np.asarray(X[:, 0], dtype=float)
    theta_rad = theta_deg * (np.pi / 180.0)
    phi_rad = phi * (np.pi / 180.0)
    return _predict_rad(theta_rad, A, k, phi_rad)
