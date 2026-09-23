"""M_BH–sigma power-law from Tremaine et al. (2002).

Tremaine et al. (2002) Eq. (19), PDF p. 10:

    log10(M_BH / M_sun) = alpha + beta * log10(sigma / sigma_0)

with best-fit parameters (chi^2 estimator, 31 galaxies):
    alpha  = 8.13 ± 0.06
    beta   = 4.02 ± 0.32
    sigma_0 = 200 km/s  (structural pivot)

This is the standard reference fit for the M_BH–sigma relation prior to
Kormendy & Ho (2013); widely used and cited.

Symbol map:
  paper sigma  -->  CSV sigma0  (Nuker slit convention, half-length r_e;
                    see caveat below)
  sigma_0 = 200 km/s is a fixed structural constant of the formula.

Published coefficients (PDF p. 10, Eq. 19):
  alpha = 8.13, beta = 4.02, sigma_0 = 200

Caveat: Tremaine et al. use the "Nuker" slit convention (half-length r_e) for
sigma, while the benchmark's sigma0 is from HyperLeda (aperture-homogenized).
The systematic difference is typically small (<5%) but may bias the fit
slightly.
"""

import numpy as np

USED_INPUTS = ["sigma0"]
PAPER_REF = "summary_formula_dataset_tremaine_2002.md"
EQUATION_LOC = "Eq. 19, p. 10 (PDF p. 10)"
LAW_CONSTANTS = {
    "alpha": 8.13,   # intercept at sigma=200 km/s (Tremaine et al. 2002 Eq. 19, PDF p. 10)
    "beta": 4.02,    # power-law slope (Tremaine et al. 2002 Eq. 19, PDF p. 10)
}
# === OTHER_CONSTANTS — fixed structural pivot/normalization ===
OTHER_CONSTANTS = {
    "sigma_pivot": 200.0,  # km/s — fixed velocity-dispersion normalization of Eq. 19
                           # (sigma_0 = 200 km/s; Tremaine et al. 2002, PDF p. 10, txt lines 22/27/265)
}
LOCAL_FITTABLE = {}


def predict(X, **params):
    """Predict log10(M_BH/M_sun) from velocity dispersion.

    Parameters
    ----------
    X     : (n, 1) array — column 0 is sigma0 in km/s

    Returns
    -------
    (n,) array of log10(M_BH / M_sun)
    """
    sigma0 = np.asarray(X[:, 0], dtype=float)
    sigma_pivot = OTHER_CONSTANTS["sigma_pivot"]
    alpha = LAW_CONSTANTS["alpha"]
    beta = LAW_CONSTANTS["beta"]
    return alpha + beta * np.log10(sigma0 / sigma_pivot)
