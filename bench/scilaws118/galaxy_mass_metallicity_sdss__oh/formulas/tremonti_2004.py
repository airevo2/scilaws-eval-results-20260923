"""Tremonti et al. (2004) mass-metallicity relation — quadratic polynomial.

Tremonti, C. A. et al. 2004, ApJ, 613, 898.
Equation (3), PDF p. 7.

Formula
-------
    12 + log(O/H) = -1.492 + 1.847 * x - 0.08026 * x^2

where x = log10(M_star / M_sun), the base-10 logarithm of stellar mass.

This is the canonical SDSS mass-metallicity relation derived from the
full sample of ~53,400 star-forming galaxies from SDSS DR4. Calibrated
over 8.5 < log(M_*) < 11.5. Derived using Bayesian emission-line
metallicity estimates from the MPA-JHU pipeline.

LAW_CONSTANTS — paper-published, frozen (PDF p. 7, Eq. 3)
---------------------------------------------------------
    a0 = -1.492   : constant offset
    a1 =  1.847   : linear coefficient
    a2 = -0.08026 : quadratic coefficient

OTHER_CONSTANTS — none (formula is dimensionless polynomial)
---------------------------------------------------------
    (empty)

Type designation: Type I — each galaxy is an independent row; no per-cluster
fit; LOCAL_FITTABLE = {}.

Column mapping: log_Mstar -> x (= log M_star in solar masses).

Note on log_SFR: intentionally excluded — Tremonti et al. (2004) has no
SFR-dependent metallicity formula; the FMR concept was introduced six years
later by Mannucci et al. (2010, MNRAS, 408, 2115). The dataset's log_SFR
column exists to support the Mannucci / Curti / Andrews FMR sister-files.

Caveats: valid range is 8.5 < log M_* < 11.5 per Tremonti et al. (2004)
Fig. 3. Extrapolation to log M_* < 8.5 or > 11.5 is outside the
calibrated regime. The released test set (log M_* > 10.5) partially
probes the upper end of the calibrated range and the sparse tail above 11.5.
"""

import numpy as np

USED_INPUTS = ["log_Mstar"]
PAPER_REF   = "summary_formula+dataset_tremonti_2004.md"
EQUATION_LOC = "Tremonti et al. 2004, ApJ 613 898, Eq. (3), PDF p. 7"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a0": -1.492,     # constant offset — PDF p. 7 Eq. (3)
    "a1":  1.847,     # linear coefficient — PDF p. 7 Eq. (3)
    "a2": -0.08026,   # quadratic coefficient — PDF p. 7 Eq. (3)
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE  = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, a0: float, a1: float, a2: float) -> np.ndarray:
    """Predict 12+log(O/H) from log stellar mass.

    X: (n, 1) — column log_Mstar (= log10 M_star / M_sun).
    Returns: (n,) array of 12+log(O/H) in dex.
    """
    x = np.asarray(X[:, 0], dtype=float)
    return a0 + a1 * x + a2 * x * x
