"""Vázquez et al. 2022 Bayesian Machine Scientist (BMS) formula for national CH4.

Citation: Vázquez D., Guimerà R., Sales-Pardo M. & Guillén-Gosálbez G. (2022).
"Automatic modeling of socioeconomic drivers of energy consumption and pollution
using Bayesian symbolic regression." Sustainable Production and Consumption 30,
596-607. DOI: 10.1016/j.spc.2021.12.025. CC BY 4.0.

Formula (PDF p. 9, equation at bottom of page; parameters from Table 3, PDF p. 9):

    log(ME) = a7 + a3 / (a4*a5/GDP - TP^a4)
                 + DP^(a4^2 * a6 / DP) / (a1*GDP*(-a4*AT_K)^(-a3) + a2)

where AT_K = AT_celsius + 273.15  (paper Nomenclature p. 2: AT in Kelvin).

Feature selection: AP and UR are excluded from the BMS model for ME
(Table 4, PDF p. 9 shows ME:AP as orange discrepancy, ME:UR as orange discrepancy;
text PDF p. 10: "the active population is not considered as a necessary driver for
ME according to the BMS model. Similarly, [...] urban rate is not a driver for ME").

LAW_CONSTANTS — from Table 3, ME column (log(ME)), PDF p. 9:
    a1 = -9.654e-26   (coefficient on GDP in denominator of term2)
    a2 = -0.278        (additive constant in denominator of term2)
    a3 = -21.508       (exponent and numerator of term1)
    a4 = -2.858e-2     (structural coefficient in both terms)
    a5 =  91.108       (multiplier in denominator of term1)
    a6 = -1.954e3      (exponent coefficient in numerator of term2)
    a7 = -11.293       (intercept / additive constant)

OTHER_CONSTANTS — unit conversion and numerical guard:
    K_OFFSET  = 273.15    (Celsius to Kelvin; SI definition)
    LOG_FLOOR = 1.0e-300  (numerical guard: prevents underflow in DP^(...))

Type designation: Type I. All seven a-coefficients are global (BMS fit on pooled
168-country x 25-year EORA26 panel; no per-country refit). LOCAL_FITTABLE is
empty. (PDF p. 9 Table 3 caption: "Values of the parameters of the BMS models.")

Column mapping (paper -> released CSV):
    GDP per capita        -> GDP   [USD 2015 per capita]
    Total population      -> TP    [persons]
    Population density    -> DP    [persons/km^2]
    Avg air temperature   -> AT    [degrees C in CSV; converted to K internally]
    (AP, UR excluded from this BMS expression for ME)

Caveats:
    - The paper states AT in Kelvin (Nomenclature, PDF p. 2). This module converts
      CSV degrees C to Kelvin before evaluation. With CSV AT range -4.97 to 29.78
      degrees C the Kelvin values are 268.18-302.93 K, ensuring (-a4*AT_K) > 0
      since a4 = -2.858e-2 < 0, so -a4 > 0, and AT_K > 0.
    - The exponent (-a3) = 21.508 applied to (-a4*AT_K) ~ 0.02857 * 270 ~ 7.72
      produces (~7.72)^21.508 ~ 1.6e19, making the denominator of term2 very
      small. This produces large log(ME) values; the a1 factor (-9.654e-26) nearly
      cancels the large numerator in term1 to give a physically reasonable result.
    - DP^(a4^2 * a6 / DP): the exponent a4^2*a6/DP = (8.168e-4)*(-1954)/DP
      = -1.596/DP which is small and negative for typical densities (DP ~ 50-500),
      so DP^(exponent) ~ e^(exponent * ln(DP)) is close to 1 but < 1.
    - Published performance (Table 1, ME row, PDF p. 5): R^2 = 0.825, MSE = 0.572,
      BIC = 9605, CVE = 0.609.
"""

import numpy as np

USED_INPUTS = ["GDP", "TP", "DP", "AT"]
PAPER_REF   = "summary_formula+dataset_vazquez_2022.md"
EQUATION_LOC = "Vazquez 2022 PDF p. 9 (equation at bottom); Table 3 (ME column), PDF p. 9; Table 4 confirms AP, UR excluded"

# LAW_CONSTANTS: BMS-fitted parameters from Table 3, ME column, PDF p. 9.
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a1": -9.654e-26,
    "a2": -0.278,
    "a3": -21.508,
    "a4": -2.858e-2,
    "a5":  91.108,
    "a6": -1.954e3,
    "a7": -11.293,
}

# OTHER_CONSTANTS: unit conversion and numerical guard; not paper-fit values.
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "K_OFFSET":  273.15,    # Celsius to Kelvin; SI definition
    "LOG_FLOOR": 1.0e-300,  # numerical floor to prevent underflow in DP^(...)
}

# Type I: no per-cluster parameters.
LOCAL_FITTABLE = {}

_K_OFFSET  = OTHER_CONSTANTS["K_OFFSET"]
_LOG_FLOOR = OTHER_CONSTANTS["LOG_FLOOR"]


def predict(X: np.ndarray, **params) -> np.ndarray:
    """Evaluate Vazquez 2022 BMS expression for ME in kg CH4.

    X.shape = (n, 4) in USED_INPUTS order: (GDP, TP, DP, AT[degrees C]).
    params = LAW_CONSTANTS (frozen paper values).
    Returns ME in kg CH4.
    """
    a1 = params.get("a1", LAW_CONSTANTS["a1"])
    a2 = params.get("a2", LAW_CONSTANTS["a2"])
    a3 = params.get("a3", LAW_CONSTANTS["a3"])
    a4 = params.get("a4", LAW_CONSTANTS["a4"])
    a5 = params.get("a5", LAW_CONSTANTS["a5"])
    a6 = params.get("a6", LAW_CONSTANTS["a6"])
    a7 = params.get("a7", LAW_CONSTANTS["a7"])

    GDP = np.asarray(X[:, 0], dtype=float)
    TP  = np.asarray(X[:, 1], dtype=float)
    DP  = np.asarray(X[:, 2], dtype=float)
    AT  = np.asarray(X[:, 3], dtype=float)

    # Convert Celsius -> Kelvin (paper uses K; Nomenclature PDF p. 2)
    AT_K = AT + _K_OFFSET   # range 268-303 K; always positive

    # term1 = a3 / (a4*a5/GDP - TP^a4)
    # denominator: a4*a5/GDP - TP^a4
    # a4 = -0.02858, so a4*a5 = -0.02858 * 91.108 ~ -2.604
    # TP^a4 = TP^(-0.02858) ~ exp(-0.02858 * ln(TP)), small positive < 1
    denom1 = a4 * a5 / GDP - TP ** a4
    term1 = a3 / denom1

    # term2 = DP^(a4^2 * a6 / DP) / (a1*GDP*(-a4*AT_K)^(-a3) + a2)
    # exponent in numerator: a4^2 * a6 / DP
    exp_dp = a4 ** 2 * a6 / np.maximum(DP, _LOG_FLOOR)
    numer2 = np.exp(exp_dp * np.log(np.maximum(DP, _LOG_FLOOR)))

    # (-a4) > 0 since a4 < 0; AT_K > 0 always
    neg_a4_ATK = (-a4) * AT_K   # positive
    denom2 = a1 * GDP * neg_a4_ATK ** (-a3) + a2
    term2 = numer2 / denom2

    log_ME = a7 + term1 + term2
    return np.exp(log_ME)
