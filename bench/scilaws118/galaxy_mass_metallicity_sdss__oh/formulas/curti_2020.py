"""Curti et al. (2020) 4-parameter mass-metallicity relation.

Curti, M. et al. 2020, MNRAS, 491, 944.
Equation (2), PDF p. 7; parameters from Table 4, PDF p. 7.

Formula
-------
    12 + log(O/H) = Z0 - (gamma / beta) * log10(1 + (M_* / M0)^(-beta))

where:
    Z0    = asymptotic metallicity at high mass (dex)
    M0    = characteristic turnover mass (M_sun, linear)
    gamma = low-mass power-law slope
    beta  = sharpness of the knee (how quickly the curve transitions)
    M_*   = stellar mass = 10^log_Mstar (M_sun)

Equivalently:
    12 + log(O/H) = Z0 - (gamma/beta) * log10(1 + 10^(-beta*(log_Mstar - log_M0)))

This 4-parameter form generalizes earlier MZR parametrizations. At low
stellar mass (M_* << M0), the relation scales as Z ~ Z0 - (gamma/beta)^(-1)
approaching a power law with slope gamma. At high mass (M_* >> M0), the
relation saturates at Z0. The parameter beta controls the width of the
transition knee.

Calibrated on SDSS DR7 galaxies using a new set of Te-based strong-line
metallicity calibrations (derived in Curti et al. 2017, 2020). Valid for
8 < log(M_*) < 11.5.

LAW_CONSTANTS — paper-published, frozen (Table 4 Eq. 2 row, PDF p. 7)
----------------------------------------------------------------------
    Z0     = 8.793  : asymptotic metallicity (dex) — Table 4 PDF p. 7
    log_M0 = 10.02  : log10 characteristic turnover mass — Table 4 PDF p. 7
    gamma  = 0.28   : low-mass power-law slope — Table 4 PDF p. 7
    beta   = 1.2    : sharpness of the knee — Table 4 PDF p. 7

OTHER_CONSTANTS — none
---------------------------------------------------------
    (empty)

Type designation: Type I — each galaxy is an independent row; no per-cluster
fit; LOCAL_FITTABLE = {}.

Column mapping: log_Mstar -> log_Mstar (= log10 M_* / M_sun).

Note on log_SFR: Curti et al. (2020) also publish a Fundamental Metallicity
Relation (FMR; Eq. 5 / Table 6, log_SFR-dependent) that is implemented
separately in curti_2020_fmr.py — see §9.18 in audit log for the (MZR, FMR)
sister-file split convention shared with Mannucci 2010.

Caveats: Curti et al. (2020) use a new set of empirical strong-line
calibrations anchored to Te-based direct metallicities (Curti et al. 2017).
These differ from the MPA-JHU Bayesian pipeline (Tremonti 2004 method)
used for OH_P50 in the dataset. Systematic offsets are expected (~0.05 dex
depending on metallicity regime). The shape parameters (gamma, beta, log_M0)
are physically motivated and should align reasonably with the dataset.
Table 4 also reports uncertainties: Z0=8.793±0.005, log_M0=10.02±0.09,
gamma=0.28±0.02, beta=1.2±0.2.
"""

import numpy as np

USED_INPUTS = ["log_Mstar"]
PAPER_REF   = "summary_formula_dataset_curti_2020.md"
EQUATION_LOC = ("Curti et al. 2020, MNRAS 491 944, "
                "Eq. (2) PDF p. 7; Table 4 (Eq. 2 row) PDF p. 7")

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "Z0":     8.793,  # asymptotic metallicity (dex) — Table 4 PDF p. 7
    "log_M0": 10.02,  # log10 characteristic turnover mass — Table 4 PDF p. 7
    "gamma":  0.28,   # low-mass power-law slope — Table 4 PDF p. 7
    "beta":   1.2,    # sharpness of the knee — Table 4 PDF p. 7
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE  = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, Z0: float, log_M0: float,
            gamma: float, beta: float) -> np.ndarray:
    """Predict 12+log(O/H) from log stellar mass (Curti 2020 4-param MZR).

    X: (n, 1) — column log_Mstar (= log10 M_star / M_sun).
    Returns: (n,) array of 12+log(O/H) in dex.
    """
    log_Mstar = np.asarray(X[:, 0], dtype=float)
    # (M_* / M0)^(-beta) = 10^(-beta * (log_Mstar - log_M0))
    exponent = -beta * (log_Mstar - log_M0)
    return Z0 - (gamma / beta) * np.log10(1.0 + 10.0 ** exponent)
