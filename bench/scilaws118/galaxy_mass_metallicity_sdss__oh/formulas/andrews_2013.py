"""Andrews & Martini (2013) asymptotic logarithmic mass-metallicity relation.

Andrews, B. H. & Martini, P. 2013, ApJ, 765, 140.
Equation (5), PDF p. 12; parameters from Table 4, PDF p. 13 (MZR row).

Formula
-------
    12 + log(O/H) = Z_asm - log10(1 + (M_TO / M_*)^gamma)

where:
    Z_asm  = asymptotic metallicity at high mass (dex)
    M_TO   = turnover stellar mass (M_sun, linear)
    gamma  = power-law slope at low mass
    M_*    = stellar mass in solar masses = 10^log_Mstar

Equivalently in log form:
    12 + log(O/H) = Z_asm - log10(1 + 10^(log_M_TO - log_Mstar)^gamma)

This asymptotic logarithmic form captures the saturation of metallicity
at high stellar mass. Parameters are from the MZR fit row of Table 4
(stacking only on stellar mass, i.e., the global MZR, not the SFR bins).
Calibrated on direct-method metallicities from SDSS stacks, valid range
log(M_*) = 7.4-10.5 (Table 4, PDF p. 13).

LAW_CONSTANTS — paper-published, frozen (Table 4, PDF p. 13, MZR row)
----------------------------------------------------------------------
    Z_asm     = 8.798   : asymptotic metallicity (dex) — Table 4 col (3), MZR row
    log_M_TO  = 8.901   : log10 turnover mass (log M_sun) — Table 4 col (2), MZR row
    gamma     = 0.640   : power-law slope — Table 4 col (4), MZR row

OTHER_CONSTANTS — none
---------------------------------------------------------
    (empty)

Type designation: Type I — each galaxy is an independent row; no per-cluster
fit; LOCAL_FITTABLE = {}.

Column mapping: log_Mstar -> log_Mstar (log10 M_* / M_sun).

Caveats: calibrated on direct-method (Te-based) metallicities, which yield
systematically lower values (~0.2-0.4 dex) than strong-line calibrations
used in Tremonti et al. (2004) or the MPA-JHU OH_P50. The dataset target
OH_P50 is from the MPA-JHU strong-line Bayesian pipeline (Tremonti 2004
method), so Andrews & Martini's Z_asm and the overall level will be offset
from the mean of the released data. The shape of the saturation is
nonetheless physically motivated.

Note: Eq. (5) PDF p. 12 uses the full formula form; Table 4 PDF p. 13
provides the numerical parameters for the MZR (M_* stacks) fit.

Note on log_SFR: Andrews & Martini (2013) §5.3 + Fig. 12 also present an
FMR projection (mu_alpha = log_Mstar - alpha * log_SFR with alpha = 0.66,
slope 0.43), but the linear intercept is graphical-only (must be read off
Fig. 12) — not paper-frozen — and Table 5 only tabulates alpha per
calibration. Therefore no andrews_2013_fmr.py sister file exists; only the
MZR row of Table 4 is paper-frozen closed-form.
"""

import numpy as np

USED_INPUTS = ["log_Mstar"]
PAPER_REF   = "summary_formula_dataset_andrews_2013.md"
EQUATION_LOC = ("Andrews & Martini 2013, ApJ 765 140, "
                "Eq. (5) PDF p. 12; Table 4 MZR row PDF p. 13")

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "Z_asm":    8.798,  # asymptotic metallicity (dex) — Table 4 PDF p. 13 MZR row
    "log_M_TO": 8.901,  # log10 turnover mass (log M_sun) — Table 4 PDF p. 13 MZR row
    "gamma":    0.640,  # power-law slope — Table 4 PDF p. 13 MZR row
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE  = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, Z_asm: float, log_M_TO: float,
            gamma: float) -> np.ndarray:
    """Predict 12+log(O/H) from log stellar mass using asymptotic MZR.

    X: (n, 1) — column log_Mstar (= log10 M_star / M_sun).
    Returns: (n,) array of 12+log(O/H) in dex.
    """
    log_Mstar = np.asarray(X[:, 0], dtype=float)
    # M_TO / M_* = 10^(log_M_TO - log_Mstar)
    log_ratio = log_M_TO - log_Mstar
    return Z_asm - np.log10(1.0 + 10.0 ** (gamma * log_ratio))
