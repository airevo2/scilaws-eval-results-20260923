"""Andrews et al. (2013) Bayesian power-law regression — SDF00 stellar mass model.

Citation: Andrews, Rosenfeld, Kraus & Wilner (2013), ApJ 771, 129.
Section 3.2.2, PDF p. 13.

Formula
-------
The paper establishes a log-log linear regression between mm-wave disk
luminosity / flux density and host stellar mass:

    log10(F_mm / Jy) = A + B * log10(M_star / M_sun)

where A and B are best-fit regression coefficients (Bayesian Kelly 2007
method) that depend on which pre-main-sequence (pre-MS) stellar evolutionary
model grid is used to infer M_star.

This module uses the SDF00 (Siess, Dufour & Forestini 2000) pre-MS model grid.

LAW_CONSTANTS (frozen, from PDF p. 13)
---------------------------------------
    A_sdf00 = -1.6   : log-space intercept, SDF00 model (95% CI: -1.6 ± 0.2)
    B_sdf00 =  1.7   : log-space slope,     SDF00 model (95% CI: 1.7 ± 0.4)

Verbatim PDF text (p. 13): "We find an intercept A=−1.2 ± 0.3, −1.8 ± 0.2,
and −1.6 ± 0.2 and slope B=2.0 ± 0.5, 1.5 ± 0.4, and 1.7 ± 0.4 for the
DM97, BCAH98, and SDF00 pre-MS model grids, respectively."

OTHER_CONSTANTS
---------------
None. The formula is dimensionally clean: log10(F_mm/Jy) and log10(M_star/M_sun)
are both dimensionless log ratios; A and B are pure numbers.

Type designation: Type I — each row is an independent star-disk system.
No per-cluster parameters. LOCAL_FITTABLE is empty.

Column mapping
--------------
    log10_M_star_SDF00 (col 7 in raw CSV, renamed here) = log10(M_star/M_sun) via SDF00
    F_mm               (col 0, SR target) = log10(F_mm / Jy)

Note: This baseline uses the SDF00 stellar mass column `log10_M_star_SDF00`
from the released CSV. SDF00 predictions are intermediate between DM97 and
BCAH98; they slightly under-predict dynamical masses by ~0.05 dex on average
(Andrews 2013 §3.2.1, PDF p. 12).

Caveats
-------
- Population-level statistical relationship with 0.7 ± 0.1 dex intrinsic
  scatter (Andrews 2013 §3.2.2, PDF p. 13-14). This is not a deterministic
  single-star predictor.
- ~45% of the Taurus catalog rows are 3-sigma upper limits (is_F_uplim=1).
  The original paper's Bayesian regression accounts for censoring; SR methods
  see upper-limit flux values as ordinary y-values.
- SDF00 model convergence issues for M_star < 0.1 M_sun (Andrews 2013
  §3.2.1, PDF p. 11); very-low-mass sources may carry larger systematic
  uncertainty.
"""

import numpy as np

USED_INPUTS = ["log10_M_star_SDF00"]
PAPER_REF   = "summary_formula+dataset_andrews_2013.md"
EQUATION_LOC = "Andrews et al. 2013, ApJ 771, 129, Section 3.2.2, PDF p. 13"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "A_sdf00": -1.6,   # log-space intercept, SDF00 grid (PDF p. 13)
    "B_sdf00":  1.7,   # log-space slope,     SDF00 grid (PDF p. 13)
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}   # dimensionally clean formula; no unit factors

LOCAL_FITTABLE = {}    # Type I — no per-cluster parameters


def predict(X: np.ndarray, A_sdf00: float, B_sdf00: float) -> np.ndarray:
    """Predict log10(F_mm / Jy) from log10(M_star / M_sun) using SDF00 fit.

    X: (n, 1) array; X[:, 0] = log10_M_star_SDF00.
    Returns (n,) array of predicted log10(F_mm / Jy).
    """
    X = np.asarray(X, dtype=float)
    log_M = X[:, 0]
    return A_sdf00 + B_sdf00 * log_M
