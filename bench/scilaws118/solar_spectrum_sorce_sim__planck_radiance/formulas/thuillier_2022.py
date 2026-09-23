"""Planck spectral radiance — L(lambda, T).

Thuillier, G. et al. (2022). Characteristics of solar-irradiance spectra from
measurements, modeling, and theoretical approach. Light: Science & Applications
11:79. DOI:10.1038/s41377-022-00750-7. Equation (1), PDF p. 12.

Formula:

    L(lambda, T) = 2 h c^2 lambda^{-5} / (exp(hc / (lambda k T)) - 1)

where lambda is wavelength in metres (SI form). In the SORCE SIM benchmark
dataset wavelength is stored in nanometres; this implementation converts
internally via lambda_m = lambda_nm * 1e-9. Output is therefore in
W m^-2 nm^-1 sr^-1 (per nanometre of bandwidth), consistent with the
benchmark target planck_radiance = SSI / Omega (Omega = 6.80e-5 sr).

LAW_CONSTANTS
-------------
Empty. This Type II law has no cross-group *fitted* invariant constant: the
Planck FORM itself is the entire universal claim. h, c, k_B are universal
CODATA givens the form merely *consumes* (not fitted on these data), so they
are OTHER, not LAW (field-classification re-audit 2026-06-02; cf. the gold
exemplar's e2 and her_hor_pt's F/R — universal constants consumed by a form
sit in OTHER). The only quantity the formula fits is per-cluster T_eff →
LOCAL_FITTABLE. Hence LAW_CONSTANTS = {}.

OTHER_CONSTANTS
---------------
The three universal physical constants the Planck form consumes (given, never
fitted on these data; the paper plugs them in):
h   : Planck constant (J s).  Thuillier 2022 PDF p. 12, txt L2329-2331:
      "h are the light velocity, the Boltzman constant, and the Planck
      constant, respectively." Exact 2019 SI redefinition value 6.62607015e-34.
c   : Speed of light in vacuum (m s^-1). Same passage ("c ... the light
      velocity"). Exact 1983 SI metre-definition value 2.99792458e8.
k_B : Boltzmann constant (J K^-1). Same passage ("k ... the Boltzman
      constant"; "Boltzman" sic in paper). Exact 2019 SI value 1.380649e-23.

The factor 2 in the numerator, the exponent -5 on lambda, and the subtrahend 1
in the denominator are structural operator-definition constants (Bose-Einstein
photon mode counting) encoded as inline numerals. The unit conversion factor
1e-9 (d lambda_m -> d lambda_nm) is absorbed into the implementation of
_planck_nm and is not a physical parameter.

LOCAL_FITTABLE
--------------
T_eff : Solar effective brightness temperature (K) for each cluster (one
        daily solar spectrum). Typical range: 5500-6700 K (Thuillier 2022
        Fig. 8, PDF p. 12). init = None: fit() uses a Wien-displacement-law
        estimate from the data as the starting point (T0 = 2.898e6 / argmax(L)).
        Fitted via bounded Brent scalar minimisation in log(L) space.

Type: Type II. Each cluster is a single day of SORCE SIM observations; the
daily solar brightness temperature varies ~10-15 K over the solar cycle, so
T_eff is recovered per cluster. The Planck formula constants h, c, k_B are
universal CODATA givens consumed by the form (OTHER_CONSTANTS, not fitted).

Column mapping:
  X[:, 0] = wavelength_nm (lambda in nm; converted to m inside _planck_nm).
  T_eff   = per-cluster solar brightness temperature (K).
  Output  = planck_radiance in W m^-2 nm^-1 sr^-1.

Caveats:
  - The conversion from SSI to L uses Omega = pi*(R_sun_arcsec * pi/648000)^2
    ~ 6.80e-5 sr (not ~ 2.15e-5 as misstated in PROVENANCE.md, which omitted
    the factor of pi). This is consistent with Thuillier 2022 p. 12:
    "L is derived from SSI/Omega, where Omega is the solid angle of the Sun".
  - The fit minimises sum of squared residuals in log(L) to weight all
    wavelengths equally on a multiplicative scale, appropriate given that L
    spans ~4 orders of magnitude across the SIM range (310-2390 nm).
  - The wavelength-axis OOD split (fit: UV-VIS, predict: NIR) means the
    fit() call sees the Planck peak; predict() is scored on the long-wavelength
    Rayleigh-Jeans tail where L ~ T / lambda^4. The recovered T_eff should
    extrapolate seamlessly because the Planck function is a single continuous
    formula.
"""

import numpy as np
from scipy.optimize import minimize_scalar

USED_INPUTS = ["wavelength_nm"]
PAPER_REF = "summary_formula_dataset_thuillier_2022.md"
EQUATION_LOC = (
    "Thuillier et al. (2022) Eq. (1), PDF p. 12 — "
    "L(lambda,T) = 2*h*c^2*lambda^{-5} / (exp(hc/(lambda*k*T)) - 1); "
    "lambda in nm via lambda_m = lambda_nm * 1e-9; output in W m^-2 nm^-1 sr^-1."
)

# LAW_CONSTANTS: empty — the Planck FORM is the whole universal claim; the only
# fitted quantity (per-cluster T_eff) is LOCAL. h, c, k_B are universal givens
# the form consumes (not fitted on these data) → OTHER, mirroring the gold
# exemplar's e2 and her_hor_pt's F/R.
LAW_CONSTANTS = {}

# OTHER_CONSTANTS: universal physical constants consumed by the Planck form.
# h:   Thuillier 2022 PDF p. 12, txt L2329-2331; exact 2019 SI value.
# c:   same passage ("the light velocity"); exact 1983 SI value.
# k_B: same passage ("the Boltzman constant"); exact 2019 SI value.
OTHER_CONSTANTS = {
    "h":   6.62607015e-34,   # Planck constant, J s (exact, 2019 SI)
    "c":   2.99792458e8,     # speed of light, m s^-1 (exact, 1983 SI)
    "k_B": 1.380649e-23,     # Boltzmann constant, J K^-1 (exact, 2019 SI)
}

LOCAL_FITTABLE = {
    "T_eff": {"init": None},  # per-cluster solar brightness temperature, K
}

# Derived composite constants (read-only, from OTHER_CONSTANTS)
_h = OTHER_CONSTANTS["h"]
_c = OTHER_CONSTANTS["c"]
_kB = OTHER_CONSTANTS["k_B"]

_2hc2 = 2.0 * _h * _c * _c      # 2 h c^2  [W m^2 sr^-1]
_hc_kB = (_h * _c) / _kB         # h c / k_B  [m K]  ≈ 1.4388e-2 m K


def _planck_nm(lambda_nm, T_eff):
    """Planck function: lambda in nm, output in W m^-2 nm^-1 sr^-1.

    L = 2hc^2 (lambda_nm*1e-9)^{-5} / (exp(hc_kB/(lambda_nm*1e-9*T)) - 1)
      * 1e-9   [d(lambda_m) -> d(lambda_nm) bandwidth factor]
    = 2hc^2 * 1e36 * lambda_nm^{-5} / (exp(hc_kB*1e9/(lambda_nm*T)) - 1)
    """
    exponent = _hc_kB * 1e9 / (lambda_nm * T_eff)  # dimensionless
    exponent = np.minimum(exponent, 709.0)           # clip overflow
    L_per_m = _2hc2 * (lambda_nm * 1e-9) ** (-5) / (np.exp(exponent) - 1.0)
    return L_per_m * 1e-9                            # W m^-2 nm^-1 sr^-1


def fit(X_fit: np.ndarray, y_fit: np.ndarray, **law_constants) -> dict:
    """Fit the per-cluster effective brightness temperature T_eff.

    Minimises sum of squared log-residuals (log-space chi^2) over T_eff
    using bounded Brent scalar minimisation. The log-space objective weights
    all wavelengths equally on a multiplicative scale, appropriate because L
    spans ~4 orders of magnitude across the SIM range.

    Starting point: Wien displacement law applied to the cluster's peak L.
      T0 = b / lambda_at_max(L)  where b = 2.898e6 nm K (Wien displacement constant).

    Bounds: [4000, 8000] K. This safely brackets all known solar-photosphere
    conditions (Thuillier 2022 Fig. 8 shows 5200-6600 K across all datasets).
    """
    lambda_nm = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    mask = (y > 0) & np.isfinite(y) & (lambda_nm > 0)
    if mask.sum() < 2:
        return {"T_eff": 5772.0}

    lam_fit = lambda_nm[mask]
    y_m = y[mask]
    log_y = np.log(y_m)

    # Wien-displacement starting point
    idx_peak = int(np.argmax(y_m))
    T0 = 2.898e6 / lam_fit[idx_peak]
    T0 = float(np.clip(T0, 4500.0, 7500.0))

    def loss(T):
        if T <= 0:
            return 1e30
        L_pred = _planck_nm(lam_fit, T)
        L_pred = np.maximum(L_pred, 1e-300)
        return float(np.sum((np.log(L_pred) - log_y) ** 2))

    try:
        result = minimize_scalar(
            loss, bounds=(4000.0, 8000.0), method="bounded",
            options={"xatol": 0.1, "maxiter": 500},
        )
        T_fit = float(result.x)
        if not np.isfinite(T_fit):
            raise RuntimeError("non-finite T_eff")
        return {"T_eff": T_fit}
    except Exception:
        return {"T_eff": T0}


def predict(X: np.ndarray, T_eff: float) -> np.ndarray:
    """Planck spectral radiance for the given cluster effective temperature.

    X: (n, 1) array, column [wavelength_nm].
    T_eff: per-cluster solar brightness temperature (K) — the LOCAL_FITTABLE
      param the harness supplies from fit(). The harness calls
      predict(X, **LAW_CONSTANTS, **local); LAW_CONSTANTS is empty here, so the
      only kwarg is T_eff. The universal constants h, c, k_B are read from
      OTHER_CONSTANTS at import (via the module-level composites _2hc2 / _hc_kB).

    Returns planck_radiance in W m^-2 nm^-1 sr^-1.
    """
    lambda_nm = np.asarray(X[:, 0], dtype=float)
    # _2hc2 = 2*h*c^2 and _hc_kB = h*c/k_B are precomputed from OTHER_CONSTANTS.
    return _planck_nm(lambda_nm, float(T_eff))
