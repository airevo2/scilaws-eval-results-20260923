"""Von Mises orientation-tuning formula with baseline — Wu & Van Hooser (2025).

Wu, Z. & Van Hooser, S. D. (2025). Bayesian estimation of orientation and
direction tuning captures parameter uncertainty. *Frontiers in Neural
Circuits*, 19. DOI: 10.3389/fncir.2025.1542332. Open Access (CC BY).

**Benchmark formula — Equation 14 (PDF p. 4, §2.3 "Von Mises test"):**

    R_vm(theta) = A * exp(k * (cos(2*(theta - phi)) - 1))

where theta is stimulus orientation (radians internally; benchmark CSV column
`orientation_deg` supplies degrees and is converted by the fit/predict
functions).

The benchmark baseline uses the **extended form** (PDF p. 4, Table 1 context):

    R(theta) = C + A * exp(k * (cos(2*(theta - phi)) - 1))

where C is a per-neuron baseline (spontaneous) response that can be negative
for deconvolved calcium imaging (Table 1 bound: C in [-MX, +MX] where MX is
the empirical maximum mean response per cell).

**Structural constants (not tunable):**
- The factor `2` inside cos(2*(theta - phi)) enforces pi-periodicity of the
  orientation tuning curve (cos(2*(theta + pi - phi)) = cos(2*(theta-phi))).
  This is a structural requirement of orientation tuning, not a free parameter
  (Wu 2025 Eq. 14, structural property; Swindale 1998 Eq. 6, same form).
- The `-1` in the exponent normalises the peak: when theta = phi,
  cos(2*0) - 1 = 0, so exp(0) = 1, giving R_vm(phi) = A exactly.
  Structural invariant (Wu 2025 Eq. 14, structural property).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None.  The von Mises FORM (Eq. 14) is the scientific claim.  A, k, phi, C
are all per-neuron fits; no universal numerical values across neurons are
published.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The `2` and `-1` are algebraic structural factors, kept as literals.

LOCAL_FITTABLE — per-cluster (per-neuron), fitted by fit()
----------------------------------------------------------
- C   : baseline response (deconv. fluor. arb. units); can be negative.
        init = None: derived from data minimum.
- A   : peak amplitude above baseline (arb. units, > 0).
        init = None: derived from data range.
- k   : tuning concentration (dimensionless, > 0); larger => sharper.
        init = None: starts at 2.0, a mid-range biologically plausible value.
- phi : preferred orientation (degrees in fit/predict interface, matching
        benchmark CSV units).  Internally converted to radians.
        init = None: determined from argmax of mean response across orientations.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["orientation_deg"]
PAPER_REF = "summary_formula_wu_2025.md"
EQUATION_LOC = (
    "Wu & Van Hooser (2025) Eq. 14, PDF p. 4, §2.3 'Von Mises test'; "
    "extended form with baseline C from Table 1 context, PDF p. 4."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "C":   {"init": None},
    "A":   {"init": None},
    "k":   {"init": None},
    "phi": {"init": None},
}


def _predict_rad(theta_rad, C, A, k, phi_rad):
    """Internal: theta and phi in radians."""
    return C + A * np.exp(k * (np.cos(2.0 * (theta_rad - phi_rad)) - 1.0))


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the extended von Mises form.

    Deterministic, data-derived initialisation:
      C0   = min(y_fit)                          — baseline estimate
      A0   = max(y_fit) - min(y_fit)             — peak-above-baseline estimate
      k0   = 2.0                                 — moderate sharpness
      phi0 = orientation corresponding to max mean response (degrees)

    The orientation axis is pi-periodic, so phi is searched over [0, 180).
    Bounds keep k > 0.1 (ensure bell shape) and A > 0.
    """
    theta_deg = np.asarray(X_fit[:, 0], dtype=float)
    theta_rad = theta_deg * (np.pi / 180.0)
    y = np.asarray(y_fit, dtype=float)

    # Data-derived init
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    C0 = y_min
    A0 = max(y_max - y_min, 1e-6)
    k0 = 2.0

    # Estimate phi0 from orientation with highest mean response (mod 180 deg)
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

    p0 = [C0, A0, k0, phi0_rad]

    # Bounds: C in [-amp_bound, +amp_bound]; A in (0, 2*data_range]; k in (0.1, 10];
    # phi in [0, pi) for orientation.
    # NOTE: Wu (2025) Table 1 states C in [-MX, +MX] where MX = empirical max mean
    # response per cell. A is the peak amplitude *above* C; physical upper bound is
    # ~data_range (= y_max - y_min). We allow 2× data_range as a safety margin.
    # k_max=10 corresponds to half-width ~12° (Swindale 1998 Eq. 6a): narrower than
    # one 10-degree bin, so k > 10 is unreliable with this bin resolution. Keeping
    # A ≤ 2*data_range AND k ≤ 10 prevents near-delta-function fits that produce
    # catastrophic RMSE on held-out interleaved bins.
    amp_bound = max(abs(y_max), abs(y_min), 1.0) * 5.0
    data_range = max(y_max - y_min, 1e-6)
    lo = [-amp_bound, 1e-9, 0.1, 0.0]
    hi = [amp_bound,  2.0 * data_range, 10.0, np.pi]

    # Clamp p0 within bounds
    p0 = [float(np.clip(v, lo[i], hi[i])) for i, v in enumerate(p0)]

    def residual(p):
        return _predict_rad(theta_rad, p[0], p[1], p[2], p[3]) - y

    try:
        sol = least_squares(residual, p0, bounds=(lo, hi),
                            method="trf", max_nfev=6000)
        C, A, k, phi_rad = sol.x
        if not np.all(np.isfinite([C, A, k, phi_rad])):
            raise RuntimeError("non-finite fit")
        phi_deg = float(phi_rad * (180.0 / np.pi))
        return {"C": float(C), "A": float(A), "k": float(k), "phi": phi_deg}
    except Exception:                                   # noqa: BLE001
        phi_deg = float(p0[3] * (180.0 / np.pi))
        return {"C": float(p0[0]), "A": float(p0[1]),
                "k": float(p0[2]), "phi": phi_deg}


def predict(X: np.ndarray, C: float, A: float, k: float, phi: float) -> np.ndarray:
    """Extended von Mises orientation-tuning prediction.

    R(theta) = C + A * exp(k * (cos(2*(theta_rad - phi_rad)) - 1))

    X: (n, 1) — column [orientation_deg].
    phi: preferred orientation in degrees (benchmark CSV units).
    """
    theta_deg = np.asarray(X[:, 0], dtype=float)
    theta_rad = theta_deg * (np.pi / 180.0)
    phi_rad = phi * (np.pi / 180.0)
    return _predict_rad(theta_rad, C, A, k, phi_rad)
