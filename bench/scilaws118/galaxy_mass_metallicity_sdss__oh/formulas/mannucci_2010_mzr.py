"""Mannucci et al. (2010) mass-metallicity relation — 4th-order polynomial.

Mannucci, F. et al. 2010, MNRAS, 408, 2115.
Equation (1), PDF pp. 4-5.

Formula
-------
    12 + log(O/H) = 8.96 + 0.31*m - 0.23*m^2 - 0.017*m^3 + 0.046*m^4

where m = log10(M_star / M_sun) - 10.

This is the 4th-order polynomial fit to the mass-metallicity relation of
SDSS galaxies, with metallicities measured using the Kewley-Ellison R23
calibration. Equation (1) appears at the page boundary between pp. 4-5 in
the published PDF. The fit is to median metallicities in stellar mass bins
(bin width 0.15 dex) from log(M_*) ~ 8 to 11.5.

LAW_CONSTANTS — paper-published, frozen (PDF pp. 4-5, Eq. 1)
-------------------------------------------------------------
    c0 = 8.96    : constant (zero-order coefficient at m=0, i.e. M_*=10^10)
    c1 = 0.31    : linear coefficient
    c2 = -0.23   : quadratic coefficient
    c3 = -0.017  : cubic coefficient
    c4 = 0.046   : quartic coefficient

OTHER_CONSTANTS — none
---------------------------------------------------------
    (empty)

Type designation: Type I — each galaxy is an independent row; no per-cluster
fit; LOCAL_FITTABLE = {}.

Column mapping: log_Mstar -> m + 10 (paper uses m = log M_* - 10).

Caveats: Mannucci et al. use a different metallicity calibration than
Tremonti et al. (2004) — the MPA-JHU OH values in the released dataset
follow the Tremonti 2004 / Bayesian emission-line calibration. Expected
to be systematically offset from Mannucci (2010) by ~0.06 dex at the
median (different calibration methods), but the shape is comparable.
"""

import numpy as np

USED_INPUTS = ["log_Mstar"]
PAPER_REF   = "summary_formula_dataset_mannucci_2010.md"
EQUATION_LOC = "Mannucci et al. 2010, MNRAS 408 2115, Eq. (1), PDF pp. 4-5"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "c0":  8.96,    # constant at m=0 — PDF pp. 4-5 Eq. (1)
    "c1":  0.31,    # linear coefficient — PDF pp. 4-5 Eq. (1)
    "c2": -0.23,    # quadratic coefficient — PDF pp. 4-5 Eq. (1)
    "c3": -0.017,   # cubic coefficient — PDF pp. 4-5 Eq. (1)
    "c4":  0.046,   # quartic coefficient — PDF pp. 4-5 Eq. (1)
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE  = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, c0: float, c1: float, c2: float,
            c3: float, c4: float) -> np.ndarray:
    """Predict 12+log(O/H) from log stellar mass (4th-order polynomial).

    X: (n, 1) — column log_Mstar (= log10 M_star / M_sun).
    Returns: (n,) array of 12+log(O/H) in dex.
    """
    m = np.asarray(X[:, 0], dtype=float) - 10.0
    return c0 + c1 * m + c2 * m**2 + c3 * m**3 + c4 * m**4
