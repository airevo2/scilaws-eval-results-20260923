"""Cruz Reyes & Anderson (2023) cluster-parallax calibration of the Galactic
Cepheid Leavitt law in the optical Gaia Wesenheit magnitude W_G.

Cruz Reyes M. & Anderson R.I., A&A 672, A85 (2023).
DOI: 10.1051/0004-6361/202244775.  arXiv: 2208.09403.

Formula (PDF p. 14, Eq. 10):
    M_W = alpha * (log P - 1) + delta + gamma * [Fe/H]

with pivot log P = 1 (P = 10 d). For the Gaia W_G Wesenheit band the
solar-metallicity ABL fit (PDF p. 16, Eqs. 23–24) gives:

    alpha = -3.242 ± 0.047 mag / log P   (Eq. 23)
    delta = -6.004 ± 0.019 mag            (Eq. 23, zero-point at [Fe/H] = 0)
    gamma = -0.384 ± 0.051 mag / dex      (Eq. 24, fixed from Breuval et al. 2022)

LAW_CONSTANTS — paper-published frozen values
----------------------------------------------
alpha, delta from Eq. 23 (PDF p. 16); gamma from Eq. 24 (PDF p. 16)
— all three are the primary scientific claim of this calibration.
gamma is taken from Breuval et al. 2022 Table 5 (WG row, PDF p. 12) and
adopted as fixed in the Cruz Reyes 2023 fit; it is reported as a fitted
result in Eq. 24, so it is the published value for this ensemble.

OTHER_CONSTANTS — structural constants, not scored
---------------------------------------------------
pivot: 1.0  — the log P pivot (P = 10 d). Structural choice of
parameterisation (PDF p. 14, Eq. 10); not a physics discovery.

Type designation: Type I — one row per Cepheid star, no per-cluster
refitting; the PLZ constants are global. LOCAL_FITTABLE = {}.

Column mapping (paper notation → released CSV):
    log P  →  log_P   (base-10 log of period in days)
    [Fe/H] →  feh     (photospheric iron abundance in dex; all rows valid
                       post wave-12 NaN drop, 2026-05-26)
"""

import numpy as np

USED_INPUTS = ["log_P", "feh"]
PAPER_REF = "summary_formula+dataset_cruzreyes_2023.md"
EQUATION_LOC = "Eq. 10 (formula form), Eqs. 23-24 (W_G coefficients), PDF p. 16"

LAW_CONSTANTS = {
    "alpha": -3.242,   # mag/log P — LL slope, Eq. 23, PDF p. 16
    "delta": -6.004,   # mag       — zero-point at [Fe/H]=0, solar metallicity, Eq. 23, PDF p. 16
    "gamma": -0.384,   # mag/dex   — metallicity coefficient, Eq. 24, PDF p. 16 (fixed from Breuval+2022)
}
OTHER_CONSTANTS = {
    "pivot": 1.0,      # log P pivot (P = 10 d) — structural, Eq. 10, PDF p. 14
}
LOCAL_FITTABLE = {}    # Type I — no per-cluster parameters


def predict(X: np.ndarray, alpha: float = -3.242, delta: float = -6.004,
            gamma: float = -0.384) -> np.ndarray:
    """Predict absolute Gaia Wesenheit magnitude M_W for each Cepheid.

    X: (n, 2) — column 0 = log_P, column 1 = feh.
    Returns M_W in mag.
    """
    pivot = OTHER_CONSTANTS["pivot"]
    log_P = np.asarray(X[:, 0], dtype=float)
    feh   = np.asarray(X[:, 1], dtype=float)   # NaN preserved for missing rows
    return alpha * (log_P - pivot) + delta + gamma * feh
