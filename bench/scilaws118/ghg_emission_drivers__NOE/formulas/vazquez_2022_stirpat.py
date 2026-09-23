"""Vázquez et al. 2022 modified-STIRPAT formula for national N2O emissions.

Citation: Vázquez D., Guimerà R., Sales-Pardo M. & Guillén-Gosálbez G. (2022).
"Automatic modeling of socioeconomic drivers of energy consumption and pollution
using Bayesian symbolic regression." Sustainable Production and Consumption 30,
596-607. DOI: 10.1016/j.spc.2021.12.025. CC BY 4.0.

Formula (Eq. 9, PDF p. 5 / journal p. 600):

    log(NOE) = a + w_GDP*log(GDP) + w_TP*log(TP) + w_AP*log(AP)
               + w_DP*log(DP) + w_UR*log(UR) + w_AT*log(AT_K)

where AT_K = AT_celsius + 273.15  (paper Nomenclature p. 2: AT in Kelvin).

LAW_CONSTANTS — from Table 2, NOE row, PDF p. 5 / journal p. 600:
    a      = 20.299   (intercept)
    w_GDP  =  0.378   (log(GDP) coefficient)
    w_TP   =  0.940   (log(TP) coefficient)
    w_AP   = -1.656   (log(AP) coefficient)
    w_DP   = -0.145   (log(DP) coefficient)
    w_UR   = -0.052   (log(UR) coefficient; p > 0.1, not significant but included)
    w_AT   = -2.542   (log(AT) coefficient, AT in Kelvin)

OTHER_CONSTANTS — unit conversion only:
    K_OFFSET = 273.15   (Celsius-to-Kelvin offset; SI definition, not from paper)

Type designation: Type I. All seven coefficients are global (one OLS fit on the
pooled 168-country × 25-year EORA26 panel; no per-country refit). LOCAL_FITTABLE
is empty. (PDF p. 5 / journal p. 600, Table 2 caption: "Coefficients of the
STIRPAT equation.")

Column mapping (paper -> released CSV):
    GDP per capita        -> GDP   [USD 2015 per capita]
    Total population      -> TP    [persons]
    Active population %   -> AP    [%; ages 15-64]
    Population density    -> DP    [persons/km2]
    Urbanisation rate     -> UR    [%; % urban]
    Avg air temperature   -> AT    [degrees C in CSV; converted to K internally]

Caveats:
    - The paper states AT in Kelvin (Nomenclature, PDF p. 2 / journal p. 597).
      The released CSV stores AT in degrees C (OWID ERA5 source). This module
      converts AT to Kelvin before taking log. With CSV AT range -4.97 to
      29.78 degrees C, the Kelvin values are 268.18-302.93 K -- all strictly
      positive; no floor needed.
    - UR is not statistically significant for NOE (p > 0.1, Table 4, PDF p. 9).
      The coefficient w_UR = -0.052 is included in the formula as it appears in
      Table 2; the paper still lists this term in Eq. 9.
    - The paper's coefficients were fitted on EORA26-sourced N2O (PRIMAP-hist).
      The benchmark uses the same NOE column from PRIMAP-hist, with WB/ERA5
      substitutes for the socioeconomic inputs.
    - Published performance: STIRPAT R2 = 0.809, MSE = 0.649 (log-space NOE),
      CVE = 0.698. Table 1, PDF p. 5 / journal p. 600.
"""

import numpy as np

USED_INPUTS = ["GDP", "TP", "AP", "DP", "UR", "AT"]
PAPER_REF   = "summary_formula+dataset_vazquez_2022.md"
EQUATION_LOC = "Vazquez 2022 Eq. 9 + Table 2 (NOE row), PDF p. 5 / journal p. 600"

# LAW_CONSTANTS: OLS coefficients from Table 2, NOE row, PDF p. 5 / journal p. 600.
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a":     20.299,
    "w_GDP":  0.378,
    "w_TP":   0.940,
    "w_AP":  -1.656,
    "w_DP":  -0.145,
    "w_UR":  -0.052,
    "w_AT":  -2.542,
}

# OTHER_CONSTANTS: unit conversion; not a paper-fit value.
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "K_OFFSET": 273.15,   # Celsius to Kelvin; SI definition
}

# Type I: no per-cluster parameters.
LOCAL_FITTABLE = {}

_K_OFFSET = OTHER_CONSTANTS["K_OFFSET"]


def predict(X: np.ndarray, **params) -> np.ndarray:
    """Evaluate Vazquez 2022 STIRPAT-extended for NOE in kg N2O.

    X.shape = (n, 6) in USED_INPUTS order: (GDP, TP, AP, DP, UR, AT[degrees C]).
    params = LAW_CONSTANTS (frozen paper values).
    Returns NOE in kg N2O.
    """
    a     = params.get("a",     LAW_CONSTANTS["a"])
    w_GDP = params.get("w_GDP", LAW_CONSTANTS["w_GDP"])
    w_TP  = params.get("w_TP",  LAW_CONSTANTS["w_TP"])
    w_AP  = params.get("w_AP",  LAW_CONSTANTS["w_AP"])
    w_DP  = params.get("w_DP",  LAW_CONSTANTS["w_DP"])
    w_UR  = params.get("w_UR",  LAW_CONSTANTS["w_UR"])
    w_AT  = params.get("w_AT",  LAW_CONSTANTS["w_AT"])

    GDP = np.asarray(X[:, 0], dtype=float)
    TP  = np.asarray(X[:, 1], dtype=float)
    AP  = np.asarray(X[:, 2], dtype=float)
    DP  = np.asarray(X[:, 3], dtype=float)
    UR  = np.asarray(X[:, 4], dtype=float)
    AT  = np.asarray(X[:, 5], dtype=float)

    # Convert Celsius to Kelvin (paper uses K; Nomenclature PDF p. 2)
    AT_K = AT + _K_OFFSET   # range 268-303 K; always positive

    log_NOE = (
        a
        + w_GDP * np.log(GDP)
        + w_TP  * np.log(TP)
        + w_AP  * np.log(AP)
        + w_DP  * np.log(DP)
        + w_UR  * np.log(UR)
        + w_AT  * np.log(AT_K)
    )
    return np.exp(log_NOE)
