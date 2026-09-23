"""Vázquez et al. 2022 modified-STIRPAT formula for national CH4 emissions.

Citation: Vázquez D., Guimerà R., Sales-Pardo M. & Guillén-Gosálbez G. (2022).
"Automatic modeling of socioeconomic drivers of energy consumption and pollution
using Bayesian symbolic regression." Sustainable Production and Consumption 30,
596-607. DOI: 10.1016/j.spc.2021.12.025. CC BY 4.0.

Formula (Eq. 9, PDF p. 5):

    log(ME) = a + w_GDP*log(GDP) + w_TP*log(TP) + w_AP*log(AP)
               + w_DP*log(DP) + w_UR*log(UR) + w_AT*log(AT_K)

where AT_K = AT_celsius + 273.15  (paper Nomenclature p. 2: AT in Kelvin).

LAW_CONSTANTS — from Table 2, ME row, PDF p. 5:
    a      = -32.862   (intercept; *** p < 0.01)
    w_GDP  =   0.302   (log(GDP) coefficient; *** p < 0.01)
    w_TP   =   0.931   (log(TP) coefficient; *** p < 0.01)
    w_AP   =   0.540   (log(AP) coefficient; *** p < 0.01)
    w_DP   =  -0.217   (log(DP) coefficient; *** p < 0.01)
    w_UR   =   0.149   (log(UR) coefficient; *** p < 0.01)
    w_AT   =   5.900   (log(AT) coefficient, AT in Kelvin; *** p < 0.01)

OTHER_CONSTANTS — unit conversion only:
    K_OFFSET = 273.15   (Celsius-to-Kelvin offset; SI definition, not from paper)

Type designation: Type I. All seven coefficients are global (one OLS fit on the
pooled 168-country × 25-year EORA26 panel; no per-country refit). LOCAL_FITTABLE
is empty. (PDF p. 5, Table 2 caption: "Coefficients of the STIRPAT equation.")

Column mapping (paper -> released CSV):
    GDP per capita        -> GDP   [USD 2015 per capita]
    Total population      -> TP    [persons]
    Active population %   -> AP    [%; ages 15-64]
    Population density    -> DP    [persons/km^2]
    Urbanisation rate     -> UR    [%; % urban]
    Avg air temperature   -> AT    [degrees C in CSV; converted to K internally]

Caveats:
    - The paper states AT in Kelvin (Nomenclature, PDF p. 2). The released CSV
      stores AT in degrees C (OWID ERA5 source). This module converts AT to Kelvin
      before taking log. With CSV AT range -4.97 to 29.78 degrees C, the Kelvin
      values are 268.18-302.93 K -- all strictly positive; no floor needed.
    - The paper's coefficients were fitted on EORA26-sourced CH4 (PRIMAP-hist).
      The benchmark uses PRIMAP-hist CH4 (HISTCR, M.0.EL). The structural form is
      preserved; the absolute-scale intercept shift is absorbed by the intercept.
    - Published performance (Table 1, ME row, PDF p. 5): R^2 = 0.809, MSE = 0.625,
      BIC = 9968, CVE = 0.673.
"""

import numpy as np

USED_INPUTS = ["GDP", "TP", "AP", "DP", "UR", "AT"]
PAPER_REF   = "summary_formula+dataset_vazquez_2022.md"
EQUATION_LOC = "Vazquez 2022 Eq. 9 + Table 2 (ME row), PDF p. 5"

# LAW_CONSTANTS: OLS coefficients from Table 2, ME row, PDF p. 5.
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a":     -32.862,
    "w_GDP":   0.302,
    "w_TP":    0.931,
    "w_AP":    0.540,
    "w_DP":   -0.217,
    "w_UR":    0.149,
    "w_AT":    5.900,
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
    """Evaluate Vazquez 2022 STIRPAT-extended for ME in kg CH4.

    X.shape = (n, 6) in USED_INPUTS order: (GDP, TP, AP, DP, UR, AT[degrees C]).
    params = LAW_CONSTANTS (frozen paper values).
    Returns ME in kg CH4.
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

    # Convert Celsius -> Kelvin (paper uses K; Nomenclature PDF p. 2)
    AT_K = AT + _K_OFFSET   # range 268-303 K; always positive

    log_ME = (
        a
        + w_GDP * np.log(GDP)
        + w_TP  * np.log(TP)
        + w_AP  * np.log(AP)
        + w_DP  * np.log(DP)
        + w_UR  * np.log(UR)
        + w_AT  * np.log(AT_K)
    )
    return np.exp(log_ME)
