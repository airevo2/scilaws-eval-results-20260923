"""Curti et al. (2020) Fundamental Metallicity Relation — SFR-dependent MZR (Eq. 5).

Curti, M. et al. 2020, MNRAS (published online 2019), arXiv:1705.05728.
Equation (5), PDF p. 17; Table 6 (Total SFR row), PDF p. 17.

Formula
-------
    12 + log(O/H) = Z0 - (gamma / beta) * log10(1 + (M_star / M0(SFR))^(-beta))

where M0 depends on SFR:
    log10(M0(SFR)) = m0 + m1 * log10(SFR)

This is the Fundamental Metallicity Relation parametrisation from Curti 2020.
The characteristic turnover mass M0 is allowed to depend on SFR, so that at
fixed stellar mass, more actively star-forming galaxies are predicted to be
less metal-rich. This reduces the residual scatter around the MZR by
incorporating SFR as a secondary parameter, analogous to Mannucci 2010 Eq. 4.

LAW_CONSTANTS — paper-published, frozen (Table 6 Total SFR row, PDF p. 17)
---------------------------------------------------------------------------
    Z0 = 8.779   : asymptotic metallicity at high mass (dex) — Table 6 PDF p. 17
    m0 = 10.11   : log10(M0) intercept (no SFR correction) — Table 6 PDF p. 17
    m1 = 0.56    : log10(M0) slope w.r.t. log_SFR — Table 6 PDF p. 17
    gamma = 0.31 : low-mass power-law slope — Table 6 PDF p. 17
    beta  = 2.1  : sharpness of the MZR knee — Table 6 PDF p. 17

OTHER_CONSTANTS — none
---------------------------------------------------------
    (empty)

Type designation: Type I — each galaxy is an independent row; no per-cluster
fit; LOCAL_FITTABLE = {}.

Column mapping: log_Mstar -> log10(M_star/M_sun), log_SFR -> log10(SFR/[M_sun/yr]).

Caveats: Curti et al. (2020) use Te-based strong-line calibrations anchored to
direct metallicities (Curti 2017). The released dataset uses MPA-JHU Bayesian
values (OH_P50) which are systematically ~0.28 dex higher than Te-based values
(known C9 calibration-method mismatch). Systematic offsets in the absolute
metallicity scale are therefore expected. The shape of the SFR-dependent
turnover is physically motivated and should reproduce the FMR trend qualitatively.
"""

import numpy as np

USED_INPUTS = ["log_Mstar", "log_SFR"]
PAPER_REF   = "summary_formula_dataset_curti_2020.md"
EQUATION_LOC = ("Curti et al. 2020, MNRAS 491 944, "
                "Eq. (5) PDF p. 17; Table 6 (Total SFR row) PDF p. 17")

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "Z0":    8.779,  # asymptotic metallicity (dex) — Table 6 PDF p. 17
    "m0":   10.11,   # log10(M0) intercept — Table 6 PDF p. 17
    "m1":    0.56,   # log10(M0) SFR slope — Table 6 PDF p. 17
    "gamma": 0.31,   # low-mass power-law slope — Table 6 PDF p. 17
    "beta":  2.1,    # sharpness of the knee — Table 6 PDF p. 17
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE  = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, Z0: float, m0: float, m1: float,
            gamma: float, beta: float) -> np.ndarray:
    """Predict 12+log(O/H) from log stellar mass and log SFR (Curti 2020 FMR).

    X: (n, 2) — columns [log_Mstar, log_SFR].
    Returns: (n,) array of 12+log(O/H) in dex.

    Derivation:
        log_M0   = m0 + m1 * log_SFR           (SFR-dependent turnover mass)
        exponent = -beta * (log_Mstar - log_M0) (power-law argument)
        OH       = Z0 - (gamma/beta) * log10(1 + 10^exponent)
    """
    log_Mstar = np.asarray(X[:, 0], dtype=float)
    log_SFR   = np.asarray(X[:, 1], dtype=float)
    log_M0  = m0 + m1 * log_SFR
    exponent = -beta * (log_Mstar - log_M0)
    return Z0 - (gamma / beta) * np.log10(1.0 + 10.0 ** exponent)
