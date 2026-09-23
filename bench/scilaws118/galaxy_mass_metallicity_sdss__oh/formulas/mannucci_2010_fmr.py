"""Mannucci et al. (2010) Fundamental Metallicity Relation — 2D polynomial.

Mannucci, F. et al. 2010, MNRAS, 408, 2115.
Equation (2), PDF p. 5.

Formula
-------
    12 + log(O/H) = 8.90 + 0.37*m - 0.14*s - 0.19*m^2 + 0.12*m*s - 0.054*s^2

where m = log10(M_star / M_sun) - 10, and s = log10(SFR / M_sun yr^-1).

This is the Fundamental Metallicity Relation (FMR) — a 2D quadratic surface
fit to median metallicities of SDSS galaxies in bins of stellar mass AND
star-formation rate (0.05 dex bins). The FMR reduces residual scatter by
~50% compared to the 1D mass-metallicity relation, with a residual scatter
of 0.05 dex. The relation holds for z < 2.5 galaxies within a common FMR
surface.

The formula uses BOTH stellar mass and SFR — it is dimensionally richer than
Tremonti 2004 or Andrews & Martini 2013 (which use only mass). The extra
dimension adds log SFR as an input.

LAW_CONSTANTS — paper-published, frozen (PDF p. 5, Eq. 2)
---------------------------------------------------------
    f0 = 8.90    : constant offset at m=s=0 — PDF p. 5 Eq. (2)
    f1 = 0.37    : linear-m coefficient — PDF p. 5 Eq. (2)
    f2 = -0.14   : linear-s coefficient — PDF p. 5 Eq. (2)
    f3 = -0.19   : quadratic-m coefficient — PDF p. 5 Eq. (2)
    f4 = 0.12    : cross-term m*s coefficient — PDF p. 5 Eq. (2)
    f5 = -0.054  : quadratic-s coefficient — PDF p. 5 Eq. (2)

OTHER_CONSTANTS — none
---------------------------------------------------------
    (empty)

Type designation: Type I — each galaxy is an independent row; no per-cluster
fit; LOCAL_FITTABLE = {}.

Column mapping: log_Mstar -> m + 10, log_SFR -> s.

Caveats: Mannucci et al. (2010) use a different metallicity calibration
(Kewley-Ellison R23) than the MPA-JHU Bayesian pipeline used in the
released dataset (OH_P50). A systematic offset of ~0.06-0.08 dex is
expected. The FMR's main contribution is its lower scatter (~0.05 dex)
vs. a mass-only fit. High-SFR galaxies (log SFR > 2) in the test set
are outside the Mannucci 2010 calibration range.
"""

import numpy as np

USED_INPUTS = ["log_Mstar", "log_SFR"]
PAPER_REF   = "summary_formula_dataset_mannucci_2010.md"
EQUATION_LOC = "Mannucci et al. 2010, MNRAS 408 2115, Eq. (2), PDF p. 5"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "f0":  8.90,    # constant — PDF p. 5 Eq. (2)
    "f1":  0.37,    # linear m — PDF p. 5 Eq. (2)
    "f2": -0.14,    # linear s — PDF p. 5 Eq. (2)
    "f3": -0.19,    # quadratic m — PDF p. 5 Eq. (2)
    "f4":  0.12,    # cross m*s — PDF p. 5 Eq. (2)
    "f5": -0.054,   # quadratic s — PDF p. 5 Eq. (2)
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE  = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, f0: float, f1: float, f2: float,
            f3: float, f4: float, f5: float) -> np.ndarray:
    """Predict 12+log(O/H) from log stellar mass and log SFR (FMR).

    X: (n, 2) — columns [log_Mstar, log_SFR].
    Returns: (n,) array of 12+log(O/H) in dex.
    """
    m = np.asarray(X[:, 0], dtype=float) - 10.0
    s = np.asarray(X[:, 1], dtype=float)
    return f0 + f1 * m + f2 * s + f3 * m**2 + f4 * m * s + f5 * s**2
