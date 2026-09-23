"""Easy-to-use M_BH scaling relation from Jin & Davis (2023).

Jin & Davis (2023) Table 1 (PDF p. 4, "Easy-to-use track"), first entry:

    log10(M_BH / M_sun) = log10(sigma0) + log10(M_sph) - 0.56 * Pseudobulge - 4.57

RMSE = 0.33 on the calibration sample.

This is a raw PySR output (not MCMC re-fit) from Table 1's "easy" track;
coefficients are directly from the table in the paper (PDF p. 4).

Symbol map:
  paper sigma_0    -->  CSV sigma0      (km/s, central velocity dispersion)
  paper M*_sph     -->  CSV log_M_sph   (log10(M_sph/M_sun); note: the formula
                         uses log10(sigma0) + log10(M_sph) = log10(sigma0) +
                         log_M_sph because log_M_sph is already in log form)
  paper Pseudobulge -->  CSV Pseudobulge (0 or 1 integer flag)

Note on input form: log_M_sph is already log10(M_sph/M_sun); sigma0 is NOT
in log form in the CSV, so log10(sigma0) is taken inside predict.

Note: Rows with NaN in log_M_sph (~19 galaxies) produce NaN predictions.

Caveat: This is a raw PySR symbolic expression without uncertainty estimates.
The four coefficients (1.0, 1.0, -0.56, -4.57) are given to 2 decimal places
in the paper's Table 1 (PDF p. 4, "easy" track row 1).
"""

import numpy as np

USED_INPUTS = ["sigma0", "log_M_sph", "Pseudobulge"]
PAPER_REF = "summary_formula+dataset_jin_2023.md"
EQUATION_LOC = "Table 1, 'Easy-to-use track' row 1, p. 4 (PDF p. 4)"
LAW_CONSTANTS = {
    "c_sigma": 1.0,      # coefficient on log10(sigma0) (Jin & Davis 2023 Table 1 "easy" row, PDF p. 4)
    "c_msph": 1.0,       # coefficient on log_M_sph (Jin & Davis 2023 Table 1 "easy" row, PDF p. 4)
    "c_pseudo": -0.56,   # coefficient on Pseudobulge indicator (Jin & Davis 2023 Table 1 "easy" row, PDF p. 4)
    "c_offset": -4.57,   # additive offset (Jin & Davis 2023 Table 1 "easy" row, PDF p. 4)
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X, **params):
    """Predict log10(M_BH/M_sun) from sigma0, log_M_sph, Pseudobulge.

    Parameters
    ----------
    X         : (n, 3) array
                  col 0: sigma0      (km/s, NOT in log form)
                  col 1: log_M_sph   (log10(M_sph/M_sun), already in log form)
                  col 2: Pseudobulge (0 or 1)

    Returns
    -------
    (n,) array of log10(M_BH / M_sun); NaN where log_M_sph is NaN
    """
    sigma0     = np.asarray(X[:, 0], dtype=float)
    log_M_sph  = np.asarray(X[:, 1], dtype=float)
    pseudobulge = np.asarray(X[:, 2], dtype=float)
    c_sigma = LAW_CONSTANTS["c_sigma"]
    c_msph = LAW_CONSTANTS["c_msph"]
    c_pseudo = LAW_CONSTANTS["c_pseudo"]
    c_offset = LAW_CONSTANTS["c_offset"]
    return (c_sigma  * np.log10(sigma0)
            + c_msph   * log_M_sph
            + c_pseudo * pseudobulge
            + c_offset)
