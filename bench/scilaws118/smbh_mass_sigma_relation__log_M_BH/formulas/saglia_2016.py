"""M_BH–sigma relation from Saglia et al. (2016).

Saglia et al. (2016) Eq. (15) and Table 11, PDF p. 33:

    log10(M_BH / M_sun) = a * log10(sigma) + ZP

Primary fit is the "All" sample (N=96) using Bayesian MCMC:
    a  = 5.246 ± 0.274
    ZP = -3.77  ± 0.631

where sigma is the effective velocity dispersion in km/s.

Symbol map:
  paper sigma  -->  CSV sigma0  (km/s; Saglia et al. measure sigma within
                    approximately R_e from IFU or long-slit spectroscopy)

Published coefficients (PDF p. 33, Table 11, "All" row):
  a = 5.246, ZP = -3.77

No per-galaxy free parameters.

Note: The formula uses log10(sigma) directly (not log10(sigma/200)), so
the intercept ZP is defined at sigma=1 km/s — it absorbs the unit
convention (ZP = -3.77 at sigma in km/s, M_BH in M_sun).

Caveat: The "All" sample includes pseudobulges, which scatter systematically
below the classical-bulge relation (Table 11 pseudo-bulge row: a=2.129,
ZP=+2.526). The benchmark sample similarly contains pseudobulge hosts
(Pseudobulge=1), so scatter for those rows is expected to be larger.
"""

import numpy as np

USED_INPUTS = ["sigma0"]
PAPER_REF = "summary_formula+dataset_saglia_2016.md"
EQUATION_LOC = "Eq. 15 and Table 11 ('All' row), p. 33 (PDF p. 33)"
LAW_CONSTANTS = {
    "a": 5.246,    # slope on log10(sigma) (Saglia et al. 2016 Table 11 "All" row, PDF p. 33)
    "ZP": -3.77,   # zero-point at sigma=1 km/s (Saglia et al. 2016 Table 11 "All" row, PDF p. 33)
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X, **params):
    """Predict log10(M_BH/M_sun) from velocity dispersion (Saglia 2016).

    Parameters
    ----------
    X  : (n, 1) array — column 0 is sigma0 in km/s

    Returns
    -------
    (n,) array of log10(M_BH / M_sun)
    """
    sigma0 = np.asarray(X[:, 0], dtype=float)
    a = LAW_CONSTANTS["a"]
    ZP = LAW_CONSTANTS["ZP"]
    return a * np.log10(sigma0) + ZP
