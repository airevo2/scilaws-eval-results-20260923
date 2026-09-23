"""Swindale (1998) wrapped Gaussian orientation-tuning function — Equation 5.

Swindale, N. V. (1998). Orientation tuning curves: empirical description and
estimation of parameters. *Biological Cybernetics*, 78, 45-56.
DOI: 10.1007/s004220050411.

**Formula — Equation 5 (PDF p. 3, §3.2):**

    G(theta) = A * sum_{n=-inf}^{+inf} exp(-(theta - phi + 180*n)^2 / (2*sigma^2))

where phi is the preferred orientation (degrees) and sigma is the tuning-curve
width (degrees).  For biologically plausible sigma, only n = -2 to +2 need be
summed (Swindale 1998 p. 3, §3.2); this truncation is a practical
implementation note, not a free parameter.

**pi-periodicity:** The wrapping term `180*n` (degrees) ensures that G is
periodic over [0, 180).  The spacing of 180 degrees reflects the 180-degree
symmetry of orientation stimuli.

**Structural constant:** The factor `2` in the denominator `2*sigma^2` is the
standard Gaussian exponent; it is an algebraic constant, not a free parameter.
The spacing `180` degrees (= pi radians) is likewise a structural constant
set by the periodicity of orientation, not a fitted value.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None.  A, phi, sigma are per-neuron fits; no universal numerical values are
published in Swindale (1998) for any of them.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)  The `2` in 2*sigma^2, and the `180` degree periodicity (pi radians)
of the wrapping, are structural algebraic invariants coded as literals.
Truncation to n in {-2, -1, 0, +1, +2} is an implementation detail (Swindale
1998 p. 3, §3.2 — "for biologically likely sigma values, only n = -2 to +2
need be included").

LOCAL_FITTABLE — per-cluster (per-neuron), fitted by fit()
----------------------------------------------------------
- A     : peak amplitude (arb. units, > 0).  init = None: data-derived.
- sigma : tuning width (degrees, > 0).  init = None: starts at 30 degrees.
- phi   : preferred orientation (degrees, [0, 180)).  init = None: data-derived.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["orientation_deg"]
PAPER_REF = "summary_formula_swindale_1998.md"
EQUATION_LOC = (
    "Swindale (1998) Eq. 5, PDF p. 3, §3.2 — "
    "G(theta) = A * sum_{n=-2}^{2} exp(-(theta - phi + 180*n)^2 / (2*sigma^2))."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "A":     {"init": None},
    "sigma": {"init": None},
    "phi":   {"init": None},
}

# Truncation range: n in {-2, -1, 0, +1, +2} (Swindale 1998 p. 3, §3.2)
_N_TERMS = np.arange(-2, 3)


def _predict_impl(theta_deg, A, sigma, phi_deg):
    """Internal: all angles in degrees."""
    # Wrapped Gaussian sum over n in {-2, -1, 0, +1, +2}
    # G(theta) = A * sum_n exp(-(theta - phi + 180*n)^2 / (2*sigma^2))
    theta = np.asarray(theta_deg, dtype=float)
    acc = np.zeros_like(theta)
    for n in _N_TERMS:
        diff = theta - phi_deg + 180.0 * n
        acc += np.exp(-(diff * diff) / (2.0 * sigma * sigma))
    return A * acc


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the wrapped Gaussian (Eq. 5).

    Deterministic data-derived initialisation:
      A0     = (max(y) - min(y)) / G_norm  where G_norm = sum_n exp(0) for n=0
               (i.e., the n=0 term at phi evaluates to 1, so A ~ amplitude)
               Since G_norm at peak = 1 + 2*exp(-180^2/(2*sigma0^2)) ~= 1 for
               sigma0=30, A0 = max(y) - min(y).
      sigma0 = 30.0 degrees  (mid-range for V1 neurons)
      phi0   = orientation of maximum mean response (degrees, mod 180)
    """
    theta_deg = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    A0 = max(y_max - y_min, 1e-6)
    sigma0 = 30.0

    unique_deg = np.unique(theta_deg % 180.0)
    if unique_deg.size > 0:
        mean_resp = np.array([
            np.mean(y[np.isclose(theta_deg % 180.0, od, atol=1e-3)])
            for od in unique_deg
        ])
        phi0_deg = float(unique_deg[np.argmax(mean_resp)])
    else:
        phi0_deg = 0.0

    amp_bound = max(abs(y_max), abs(y_min), 1.0) * 20.0
    lo = [1e-9, 1.0,   0.0]
    hi = [amp_bound, 90.0, 180.0]
    p0 = [float(np.clip(A0, lo[0], hi[0])),
          float(np.clip(sigma0, lo[1], hi[1])),
          float(np.clip(phi0_deg, lo[2], hi[2]))]

    def residual(p):
        return _predict_impl(theta_deg, p[0], p[1], p[2]) - y

    try:
        sol = least_squares(residual, p0, bounds=(lo, hi),
                            method="trf", max_nfev=6000)
        A, sigma, phi_deg = sol.x
        if not np.all(np.isfinite([A, sigma, phi_deg])):
            raise RuntimeError("non-finite fit")
        return {"A": float(A), "sigma": float(sigma), "phi": float(phi_deg)}
    except Exception:                                   # noqa: BLE001
        return {"A": float(p0[0]), "sigma": float(p0[1]), "phi": float(p0[2])}


def predict(X: np.ndarray, A: float, sigma: float, phi: float) -> np.ndarray:
    """Swindale (1998) Eq. 5 wrapped Gaussian orientation tuning.

    G(theta) = A * sum_{n=-2}^{2} exp(-(theta - phi + 180*n)^2 / (2*sigma^2))

    X: (n_samples, 1) — column [orientation_deg].
    phi: preferred orientation in degrees.
    sigma: tuning width in degrees.
    """
    theta_deg = np.asarray(X[:, 0], dtype=float)
    return _predict_impl(theta_deg, A, sigma, phi)
