"""M_BH–sigma relation from McConnell & Ma (2013).

McConnell & Ma (2013) Eq. (2), PDF p. 4, with best-fit parameters from
Table 2, PDF p. 8 (full 72-galaxy sample, MPFITEXY):

    log10(M_BH / M_sun) = alpha + beta * log10(sigma / 200 km/s)
    alpha = 8.32 ± 0.05
    beta  = 5.64 ± 0.32
    intrinsic scatter epsilon_0 = 0.43 dex

This is a substantially updated fit over prior compilations, including
new massive-end BHs (NGC 4889, NGC 3842 at M_BH ~ 10^10 M_sun) that
steepen the slope.

Symbol map:
  paper sigma  -->  CSV sigma0  (km/s; McConnell & Ma use sigma integrated
                    from r_inf to r_eff; the benchmark uses HyperLeda sigma0)
  pivot 200 km/s is a fixed structural constant.

Published coefficients (PDF p. 8, Table 2, row "All" / MPFITEXY):
  alpha = 8.32, beta = 5.64, pivot = 200

No per-galaxy free parameters.

Caveat: This fit includes the most massive BHs known (BCG-class galaxies),
which dominate the high-sigma end and inflate the slope relative to earlier
compilations (Gebhardt 2000: beta=3.75; Tremaine 2002: beta=4.02).
"""

import numpy as np

USED_INPUTS = ["sigma0"]
PAPER_REF = "summary_formula_dataset_mcconnell_2013.md"
EQUATION_LOC = "Eq. 2, p. 4 and Table 2, p. 8 (full-sample MPFITEXY)"
LAW_CONSTANTS = {
    "alpha": 8.32,   # intercept at sigma=200 km/s (McConnell & Ma 2013 Table 2, PDF p. 8)
    "beta": 5.64,    # power-law slope (McConnell & Ma 2013 Table 2, PDF p. 8)
}
# === OTHER_CONSTANTS — fixed structural pivot/normalization ===
OTHER_CONSTANTS = {
    "sigma_pivot": 200.0,  # km/s — fixed velocity-dispersion normalization of Eq. 2;
                           # "log10(M.) = 8.32 + 5.64 log10(sigma/200 km s-1)"
                           # (McConnell & Ma 2013, PDF p. 4 line 311 / p. 8 line 1729)
}
LOCAL_FITTABLE = {}


def predict(X, **params):
    """Predict log10(M_BH/M_sun) from velocity dispersion (M-sigma formula).

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
