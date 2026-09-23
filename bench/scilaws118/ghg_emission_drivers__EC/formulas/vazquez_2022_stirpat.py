"""STIRPAT-extended log-linear baseline for energy consumption — Vázquez et al. 2022.

Citation: Vázquez D., Guimerà R., Sales-Pardo M., Guillén-Gosálbez G. (2022).
Automatic modeling of socioeconomic drivers of energy consumption and pollution
using Bayesian symbolic regression. Sustainable Production and Consumption 30,
596–607. DOI: 10.1016/j.spc.2021.12.025.

Formula (Eq. 9, PDF p. 5; coefficients from Table 2 EC row, PDF pp. 5–6):

    log(EC) = log(a) + α·log(GDP) + β·log(TP) + ω_AP·log(AP)
              + ω_DP·log(DP) + ω_UR·log(UR) + ω_AT·log(AT_K)

where AT_K is mean annual surface temperature in Kelvin (°C + 273.15 per the
paper Nomenclature table, PDF p. 2, which lists AT in K).

LAW_CONSTANTS — all from Table 2, EC row, PDF pp. 5–6 (p < 0.01 for each):
    log_a    = 18.405  (intercept in log space)
    alpha    =  0.460  (log GDP coefficient)
    beta     =  0.911  (log TP coefficient)
    omega_AP =  2.309  (log AP coefficient)
    omega_DP = -0.033  (log DP coefficient)
    omega_UR =  0.541  (log UR coefficient)
    omega_AT = -6.211  (log AT_K coefficient)

OTHER_CONSTANTS — unit conversion:
    AT_K_OFFSET = 273.15   [K − °C offset; structural, not fitted; Kelvin
                            formula per paper Nomenclature p. 2]

Type I: single global fit over all country-year rows; no per-cluster parameters.
LOCAL_FITTABLE = {} (empty dict — Type I).

Column mapping (CSV → paper notation):
    GDP → A (affluence, constant 2015 USD/person)
    TP  → P (total population, persons)
    AP  → active-population share (%), 15–64 age group
    DP  → population density (persons/km²)
    UR  → urbanisation rate (%)
    AT  → T (mean annual surface temperature, °C in CSV → K in formula)

Caveat: AT is stored in °C in the released CSV; the formula converts to K
internally via AT_K_OFFSET to preserve the published coefficient ω_AT = −6.211.
"""

import numpy as np

USED_INPUTS = ["GDP", "TP", "AP", "DP", "UR", "AT"]
PAPER_REF = "summary_formula+dataset_vazquez_2022.md"
EQUATION_LOC = "Eq. 9, PDF p. 5; Table 2 EC row, PDF pp. 5–6"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "log_a":    18.405,
    "alpha":     0.460,
    "beta":      0.911,
    "omega_AP":  2.309,
    "omega_DP": -0.033,
    "omega_UR":  0.541,
    "omega_AT": -6.211,
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "AT_K_OFFSET": 273.15,   # Celsius -> Kelvin offset; Vazquez 2022 Nomenclature p. 2
}

LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters


def predict(
    X: np.ndarray,
    log_a: float,
    alpha: float,
    beta: float,
    omega_AP: float,
    omega_DP: float,
    omega_UR: float,
    omega_AT: float,
) -> np.ndarray:
    """Predict EC (TJ) from X = [GDP, TP, AP, DP, UR, AT] columns.

    LAW_CONSTANTS arrive as named params via predict(X, **LAW_CONSTANTS);
    the OTHER constant AT_K_OFFSET is read from the OTHER_CONSTANTS dict.
    """
    AT_K_OFFSET = OTHER_CONSTANTS["AT_K_OFFSET"]
    GDP  = np.asarray(X[:, 0], dtype=float)
    TP   = np.asarray(X[:, 1], dtype=float)
    AP   = np.asarray(X[:, 2], dtype=float)
    DP   = np.asarray(X[:, 3], dtype=float)
    UR   = np.asarray(X[:, 4], dtype=float)
    AT_K = np.asarray(X[:, 5], dtype=float) + AT_K_OFFSET

    log_EC = (
        log_a
        + alpha    * np.log(GDP)
        + beta     * np.log(TP)
        + omega_AP * np.log(AP)
        + omega_DP * np.log(DP)
        + omega_UR * np.log(UR)
        + omega_AT * np.log(AT_K)
    )
    return np.exp(log_EC)
