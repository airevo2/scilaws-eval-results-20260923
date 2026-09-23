"""M_BH–M_bulge relation from McConnell & Ma (2013).

McConnell & Ma (2013) Table 2, PDF p. 8, M_BH–M_bulge fit
(35 early-type galaxies with dynamical M_bulge, MPFITEXY):

    log10(M_BH / M_sun) = alpha + beta * log10(M_bulge / 1e11 M_sun)
    alpha = 8.46 ± 0.08
    beta  = 1.05 ± 0.11
    intrinsic scatter epsilon_0 = 0.34 dex

Uses the log spheroid mass (log_M_sph) available in the benchmark CSV
as a proxy for the dynamical bulge mass M_bulge.

Symbol map:
  paper M_bulge  -->  CSV log_M_sph (in log10(M_sun) units; to use in formula:
                      M_bulge / 1e11 = 10^(log_M_sph - 11))

Published coefficients (PDF p. 8, Table 2, row "M_bulge" / MPFITEXY, 35 ETGs):
  alpha = 8.46, beta = 1.05, pivot = 1e11 M_sun

No per-galaxy free parameters.

Caveat: This formula uses log_M_sph as input.  Rows with NaN in log_M_sph
(19 galaxies in the benchmark CSV) will produce NaN predictions.  The paper's
M_bulge is from dynamical mass-to-light ratios; the benchmark uses stellar
spheroid masses, which may differ by up to ~0.2 dex.
"""

import numpy as np

USED_INPUTS = ["log_M_sph"]
PAPER_REF = "summary_formula_dataset_mcconnell_2013.md"
EQUATION_LOC = "Table 2, p. 8 (M_BH–M_bulge fit, 35 ETGs, MPFITEXY)"
LAW_CONSTANTS = {
    "alpha": 8.46,   # intercept at M_bulge=1e11 M_sun (McConnell & Ma 2013 Table 2, PDF p. 8)
    "beta": 1.05,    # power-law slope on log10(M_bulge/1e11) (McConnell & Ma 2013 Table 2, PDF p. 8)
}
# === OTHER_CONSTANTS — fixed structural pivot/normalization ===
OTHER_CONSTANTS = {
    "log_mass_pivot": 11.0,  # log10(M_sun) — fixed mass normalization 1e11 M_sun of the fit;
                             # "log10(M.) = 8.46 + 1.05 log10(Mbulge/10^11 M_sun)"
                             # (McConnell & Ma 2013, PDF p. 8 lines 21/580/1607)
}
LOCAL_FITTABLE = {}


def predict(X, **params):
    """Predict log10(M_BH/M_sun) from log spheroid stellar mass.

    Parameters
    ----------
    X     : (n, 1) array — column 0 is log_M_sph = log10(M_sph / M_sun)

    Returns
    -------
    (n,) array of log10(M_BH / M_sun)
    """
    log_M_sph = np.asarray(X[:, 0], dtype=float)
    log_mass_pivot = OTHER_CONSTANTS["log_mass_pivot"]
    alpha = LAW_CONSTANTS["alpha"]
    beta = LAW_CONSTANTS["beta"]
    # log10(M_bulge / 1e11) = log_M_sph - 11
    return alpha + beta * (log_M_sph - log_mass_pivot)
