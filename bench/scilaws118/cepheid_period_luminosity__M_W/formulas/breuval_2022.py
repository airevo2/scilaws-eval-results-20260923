"""Breuval et al. (2022) period-luminosity-metallicity relation for the Gaia
Wesenheit magnitude W_G.

Breuval L. et al., ApJ 939(2), 89 (2022).
DOI: 10.3847/1538-4357/ac97e2.  arXiv: 2205.06280.

Formula (PDF p. 2, Eq. 1):
    M = alpha * (log P - log P0) + delta + gamma * [Fe/H]

with pivot log P0 = 0.7 (P0 ≈ 5 days). For the Gaia W_G Wesenheit band,
the coefficients from Table 4 and Table 5 (PDF pp. 10, 12) are:

    alpha = -3.338  mag / log P  (LMC slope fixed in PLZ fit; Table 4, WG row, PDF p. 10)
    gamma = -0.384  mag / dex    (metallicity coefficient; Table 5, WG row, PDF p. 12)
    delta = -4.958  mag          (zero-point at solar metallicity; Table 5, WG row, PDF p. 12)

LAW_CONSTANTS — paper-published frozen values
----------------------------------------------
alpha from Table 4 (LMC αfixed column, WG row, PDF p. 10): -3.338 ± 0.012.
gamma from Table 5 (γ column, WG row, PDF p. 12): -0.384 ± 0.051.
delta from Table 5 (δ column, WG row, PDF p. 12): -4.958 ± 0.025.
These are the paper's primary fitted constants — adopted as frozen LAW values.

OTHER_CONSTANTS — structural constants, not scored
---------------------------------------------------
pivot: 0.7  — the log P pivot (P₀ ≈ 5 d). Structural constant of the
parameterisation (Eq. 1, PDF p. 2); not a physics discovery.

Type designation: Type I — global PLZ with no per-cluster parameters.
LOCAL_FITTABLE = {}.

Column mapping (paper notation → released CSV):
    log P  →  log_P   (base-10 log of period in days)
    [Fe/H] →  feh     (photospheric iron abundance in dex; all rows valid
                       post wave-12 NaN drop, 2026-05-26)

Note on alpha source: The LMC slope α = -3.338 is fixed from the LMC
PL fit (Table 4, LMC αfixed column, WG row). The MW free slope (αfree)
for WG in Table 4 is −3.112 ± 0.060, but the PLZ fit fixes α to the LMC
value following standard practice in this calibration paper. The frozen
LAW value -3.338 is the correct published value for this formula module.

Note on the released target: The benchmark uses M_W^G (Gaia Wesenheit,
computed as WG + 5·log10(plx_µas) − 25). This paper's W_G definition uses
the BP/RP reddening coefficient 1.90 (Breuval 2022 §2 / Table 1), whereas
the benchmark data is the Gaia W_G as listed in Cruz Reyes 2023 Table 10
(coefficient 1.921; Cruz Reyes 2023 Eq. 6). The (1.921 − 1.90) × (BP − RP)
≈ 0.02 mag systematic from the coefficient difference is small compared
with the ≈ 0.05 mag intrinsic PLZ scatter and is absorbed into the fit
residuals (breuval rmse 0.373 vs cruzreyes rmse 0.376 — nearly identical).
Cruz Reyes 2023 later adopted γ_WG = -0.384 from this paper (Breuval 2022
Table 5) as their fixed metallicity slope; so γ is the same in both modules.
"""

import numpy as np

USED_INPUTS = ["log_P", "feh"]
PAPER_REF = "summary_formula+dataset_breuval_2022.md"
EQUATION_LOC = "Eq. 1 (formula form, PDF p. 2); Table 4 (alpha, WG, PDF p. 10); Table 5 (gamma, delta, WG, PDF p. 12)"

LAW_CONSTANTS = {
    "alpha": -3.338,   # mag/log P — LMC slope (fixed), Table 4 WG row, PDF p. 10
    "gamma": -0.384,   # mag/dex   — metallicity coefficient, Table 5 WG row, PDF p. 12
    "delta": -4.958,   # mag       — zero-point at solar metallicity, Table 5 WG row, PDF p. 12
}
OTHER_CONSTANTS = {
    "pivot": 0.7,      # log P pivot (P₀ ≈ 5 d) — structural constant, Eq. 1, PDF p. 2
}
LOCAL_FITTABLE = {}    # Type I — no per-cluster parameters


def predict(X: np.ndarray, alpha: float = -3.338, gamma: float = -0.384,
            delta: float = -4.958) -> np.ndarray:
    """Predict absolute Gaia Wesenheit magnitude M_W for each Cepheid.

    X: (n, 2) — column 0 = log_P, column 1 = feh.
    Returns M_W in mag.
    """
    pivot = OTHER_CONSTANTS["pivot"]
    log_P = np.asarray(X[:, 0], dtype=float)
    feh   = np.asarray(X[:, 1], dtype=float)  
    return alpha * (log_P - pivot) + delta + gamma * feh
