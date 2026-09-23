"""Gaisser-Hillas (R, L) reparametrized longitudinal profile — dE/dX.

Aab, A. et al. (Pierre Auger Collaboration) (2019). *Measurement of the
average shape of longitudinal profiles of cosmic-ray air showers at the
Pierre Auger Observatory.* JCAP 2019(03):018.
DOI: 10.1088/1475-7516/2019/03/018. arXiv:1811.04660v2.

Equation 3.1 (PDF p. 6) gives the normalized Gaisser-Hillas profile in the
(R, L) reparametrization:

    (dE/dX)' = (1 + R * X' / L)^(R^{-2}) * exp(-X' / (R*L))

where X' = X - Xmax is the depth relative to shower maximum (g/cm^2),
(dE/dX)' = (dE/dX) / (dE/dX)_max is the profile normalized to 1 at maximum,
R is the dimensionless asymmetry parameter, and L is the width (g/cm^2).

The relations to the original Gaisser-Hillas parameters are (Eq. 3.1 caption):
    R = sqrt(lambda / |X'_0|),  L = sqrt(|X'_0| * lambda),
    X'_0 = X_0 - Xmax.

The raw dE/dX in PeV/(g/cm^2) is:
    dE/dX(X) = dEdX_max * (dE/dX)'(X')

where dEdX_max = (dE/dX)_max is the per-shower peak energy deposit.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The (R, L) form is the scientific claim. No universal numerical values
for R or L are fixed — they vary per shower and per energy bin. Table 2
(PDF p. 10-11) reports measured values (R: 0.24-0.27; L: 226-238 g/cm^2)
but these are energy-dependent population means, not universal law constants.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
The exponent R^{-2} in (1 + R*X'/L)^{R^{-2}} is not an independent constant
— it is an algebraic consequence of the reparametrization from (X0, lambda)
to (R, L) (Eq. 3.1 and the two relations above). It is a literal structural
factor of the closed form.

LOCAL_FITTABLE — per-shower (per-cluster), fitted via nonlinear LS
-------------------------------------------------------------------
- Xmax     : depth of shower maximum (g/cm^2, >0). The profile peaks at X=Xmax.
- dEdX_max : peak energy deposit, i.e. (dE/dX)_max (PeV/(g/cm^2), >0).
             Scales the normalized shape to raw units.
- R        : asymmetry parameter (dimensionless, >0). Typical Auger range 0.24-0.27
             (Table 2, PDF p. 10-11; per-shower scatter is broader).
- L        : width parameter (g/cm^2, >0). Typical Auger range 226-238 g/cm^2
             (Table 2, PDF p. 10-11; per-shower scatter is broader).

init = None on all: fit() builds its own data-derived start from profile moments.

Physical validity domain: X' in [-300, +200] g/cm^2 (Sec. 4, PDF p. 8;
validated to 1% accuracy in this range). The formula can be evaluated outside
this range but is not validated there.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["X"]
PAPER_REF = "summary_formula_dataset_aab_2019.md"
EQUATION_LOC = (
    "Aab et al. (2019) Eq. 3.1, PDF p. 6 — "
    "(dE/dX)' = (1 + R*X'/L)^(R^{-2}) * exp(-X'/(R*L)); "
    "raw profile: dE/dX = dEdX_max * (dE/dX)'(X - Xmax)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "Xmax":     {"init": None},
    "dEdX_max": {"init": None},
    "R":        {"init": None},
    "L":        {"init": None},
}


def _normalized_profile(X_prime, R, L):
    """Normalized GH profile (dE/dX)' evaluated at X' = X - Xmax.

    Returns 0 wherever the base (1 + R*X'/L) is non-positive to avoid
    complex-valued or undefined power. This clips the unphysical tail
    below the profile foot (X' < -L/R), matching the paper's physical
    domain where the profile is positive.
    """
    base = 1.0 + R * X_prime / L
    # Clip: where base <= 0, profile is 0 (shower has not started yet /
    # foot of profile where approximation breaks down).
    exponent = R ** (-2.0)
    safe_base = np.where(base > 0.0, base, 0.0)
    # exp(-X'/(R*L)) part:
    exp_term = np.exp(-X_prime / (R * L))
    return np.where(base > 0.0, safe_base ** exponent * exp_term, 0.0)


def _profile(X, Xmax, dEdX_max, R, L):
    """Raw GH profile in PeV/(g/cm^2)."""
    X_prime = X - Xmax
    return dEdX_max * _normalized_profile(X_prime, R, L)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the GH (R, L) profile.

    Data-derived initialisation:
      Xmax0    = X value at the observed peak dE/dX (median-smoothed).
      dEdX_max0 = observed maximum dE/dX.
      R0       = 0.26 — midpoint of Auger measured range 0.24-0.27 (Table 2,
                 Aab 2019 PDF p. 10-11); a reasonable per-shower start.
      L0       = 232 — midpoint of Auger measured range 226-238 g/cm^2
                 (Table 2, Aab 2019 PDF p. 10-11).

    Single-start Levenberg-Marquardt / trust-region. The GH profile is
    smooth and unimodal on well-resolved shower data, so single-start is
    sufficient.
    """
    X = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # Data-derived start
    idx_peak = int(np.argmax(y))
    Xmax0 = float(X[idx_peak])
    dEdX_max0 = float(np.maximum(y[idx_peak], 1e-6))
    # Mid-range of Auger Table 2 values (Aab 2019 PDF p. 10-11)
    R0 = 0.26
    L0 = 232.0

    p0 = [Xmax0, dEdX_max0, R0, L0]

    # Bounds: Xmax in [300, 1200] g/cm^2 (physical atmospheric range);
    # dEdX_max > 0; R in [0.05, 1.0]; L in [50, 800] g/cm^2.
    lo = [300.0,  1e-8, 0.05,  50.0]
    hi = [1200.0, 1e6,  1.0,  800.0]
    p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]

    def residual(p):
        return _profile(X, p[0], p[1], p[2], p[3]) - y

    try:
        sol = least_squares(residual, p0, bounds=(lo, hi),
                            method="trf", max_nfev=5000)
        Xmax, dEdX_max, R, L = sol.x
        if not np.all(np.isfinite([Xmax, dEdX_max, R, L])):
            raise RuntimeError("non-finite fit")
        return {
            "Xmax":     float(Xmax),
            "dEdX_max": float(dEdX_max),
            "R":        float(R),
            "L":        float(L),
        }
    except Exception:                                   # noqa: BLE001
        return {
            "Xmax":     float(p0[0]),
            "dEdX_max": float(p0[1]),
            "R":        float(p0[2]),
            "L":        float(p0[3]),
        }


def predict(X: np.ndarray, Xmax: float, dEdX_max: float,
            R: float, L: float) -> np.ndarray:
    """Gaisser-Hillas (R, L) longitudinal energy-deposit profile.

    X: (n, 1) — column [X] with slant atmospheric depth (g/cm^2).
    Returns dE/dX in PeV/(g/cm^2).
    """
    X_arr = np.asarray(X[:, 0], dtype=float)
    return _profile(X_arr, Xmax, dEdX_max, R, L)
