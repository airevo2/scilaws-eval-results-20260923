"""Vázquez et al. 2022 Bayesian Machine Scientist (BMS) formula for national N2O.

Citation: Vázquez D., Guimerà R., Sales-Pardo M. & Guillén-Gosálbez G. (2022).
"Automatic modeling of socioeconomic drivers of energy consumption and pollution
using Bayesian symbolic regression." Sustainable Production and Consumption 30,
596-607. DOI: 10.1016/j.spc.2021.12.025. CC BY 4.0.

Formula (Table 3 and page-9 formula block, PDF p. 9 / journal p. 604):

    log(NOE) = a4 + a4^AT * (DP/GDP + UR^2/(a1*GDP))^(-a3/AP)
                            * log(a2^DP * TP)

where AT is in Kelvin (= AT_Celsius + 273.15), per the paper's
Nomenclature (PDF p. 2): "AT — Average temperature (K)". The released
CSV stores AT in Celsius; predict() converts to Kelvin via AT_K_OFFSET.

Audit trail: this docstring + predict() were repaired on 2026-05-25 by
relay-A gen 9 wave 82 after an opus PDF audit (see audit/triages/
vazquez_2022_bms_NOE_PDF_AUDIT_2026-05-25.md) found that AT was being
used in Celsius despite the paper convention being Kelvin. The sister
EC baseline (already fixed in wave 81) and the STIRPAT-NOE baseline
both apply the Kelvin conversion. With a4=0.999, the factor a4^AT
changes from ~0.98 (Celsius @20°C) to ~0.75 (Kelvin @293 K) — a
material magnitude shift. Formula algebra is unchanged.

LAW_CONSTANTS -- from Table 3, NOE column, PDF p. 9 / journal p. 604:
    a1 = 90.005   (denominator scaling in UR^2/(a1*GDP) term)
    a2 =  1.000   (base of a2^DP term; with a2=1.000, a2^DP = 1 for any DP)
    a3 =  1.598   (base for -a3/AP exponent)
    a4 =  0.999   (offset constant and base of a4^AT factor)

OTHER_CONSTANTS -- numerical guards + unit conversion:
    LOG_FLOOR    = 1.0e-20   (floor for inner parenthesis and log argument)
    AT_K_OFFSET  = 273.15    (Celsius → Kelvin for AT; paper convention)

Type designation: Type I. All four BMS constants are global (BMS fit on pooled
168-country x 25-year EORA26 panel; no per-country refit). LOCAL_FITTABLE is
empty. (PDF p. 8 / journal p. 603: "model with the shortest description length
[...] using the entire dataset.")

Column mapping (paper -> released CSV):
    GDP per capita        -> GDP   [USD 2015 per capita]
    Total population      -> TP    [persons]
    Active population %   -> AP    [%; ages 15-64; used as denominator in exponent]
    Population density    -> DP    [persons/km2]
    Urbanisation rate     -> UR    [%; % urban]
    Avg air temperature   -> AT    [degrees C in CSV → Kelvin in predict; used as exponent on a4]

Caveats:
    - The exponent of the inner parenthesis is -a3/AP (division), not -a3^AP
      (power). With AP stored as whole-number percentage (~45-85%), the division
      form gives sensible exponents (~-0.02 to -0.04); the power form would give
      astronomically large exponents for AP > 1.
    - With a2 = 1.000 exactly, a2^DP = 1 for all DP, so log(a2^DP * TP) = log(TP).
      This effectively removes DP from the log term (the paper notes this redundancy
      in the feature selection analysis, PDF p. 10 / journal p. 604).
    - a4 = 0.999 gives a4^AT ~ 0.746 at AT=293 K (= 20 degrees C → Kelvin via
      AT_K_OFFSET). Over the panel range AT_C in [-20, 30] = AT_K in [253, 303],
      a4^AT_K ranges from ~0.776 to ~0.738 — a slowly-decaying factor that
      remains well below 1 across the panel (in contrast to the old Celsius
      reading, which gave near-unity values ~0.98).
    - The inner parenthesis (DP/GDP + UR^2/(a1*GDP)) is always positive given
      positive GDP, DP, UR; LOG_FLOOR guards against near-zero edge cases.
    - Published performance: BMS R2 = 0.817, MSE = 0.620, CVE = 0.639 (NOE).
      Table 1, PDF p. 5 / journal p. 600.
"""

import numpy as np

USED_INPUTS = ["GDP", "TP", "AP", "DP", "UR", "AT"]
PAPER_REF   = "summary_formula+dataset_vazquez_2022.md"
EQUATION_LOC = "Vazquez 2022 Table 3 (NOE column) + page-9 formula block, PDF p. 9"

# LAW_CONSTANTS: BMS-fitted parameters from Table 3, NOE column, PDF p. 9.
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a1": 90.005,
    "a2":  1.000,
    "a3":  1.598,
    "a4":  0.999,
}

# OTHER_CONSTANTS: numerical guard + unit conversion; not paper-fit values.
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "LOG_FLOOR":   1.0e-20,   # floor for inner parenthesis before power and log
    "AT_K_OFFSET": 273.15,    # Celsius → Kelvin for AT (paper convention; matches sister STIRPAT)
}

# Type I: no per-cluster parameters.
LOCAL_FITTABLE = {}

_LOG_FLOOR    = OTHER_CONSTANTS["LOG_FLOOR"]
_AT_K_OFFSET  = OTHER_CONSTANTS["AT_K_OFFSET"]


def predict(X: np.ndarray, **params) -> np.ndarray:
    """Evaluate Vazquez 2022 BMS expression for NOE in kg N2O.

    X.shape = (n, 6) in USED_INPUTS order: (GDP, TP, AP, DP, UR, AT[degrees C]).
    AT is converted to Kelvin (= AT_C + 273.15) per paper Nomenclature.
    params = LAW_CONSTANTS (frozen paper values).
    Returns NOE in kg N2O.
    """
    a1 = params.get("a1", LAW_CONSTANTS["a1"])
    a2 = params.get("a2", LAW_CONSTANTS["a2"])
    a3 = params.get("a3", LAW_CONSTANTS["a3"])
    a4 = params.get("a4", LAW_CONSTANTS["a4"])

    GDP  = np.asarray(X[:, 0], dtype=float)
    TP   = np.asarray(X[:, 1], dtype=float)
    AP   = np.asarray(X[:, 2], dtype=float)
    DP   = np.asarray(X[:, 3], dtype=float)
    UR   = np.asarray(X[:, 4], dtype=float)
    AT_C = np.asarray(X[:, 5], dtype=float)
    AT_K = AT_C + _AT_K_OFFSET   # Celsius → Kelvin per paper Nomenclature

    # a4^AT_K: slowly decaying factor (a4~0.999, AT_K in [253, 303])
    a4_AT = a4 ** AT_K

    # inner parenthesis: DP/GDP + UR^2/(a1*GDP)
    inner = np.maximum(DP / GDP + UR ** 2 / (a1 * GDP), _LOG_FLOOR)

    # exponent: -a3/AP (division form; AP in whole-number %)
    exponent = -a3 / AP

    # log(a2^DP * TP): with a2=1.000 this is log(TP)
    # clip TP to LOG_FLOOR to guard log(0)
    log_arg = np.maximum(a2 ** DP * TP, _LOG_FLOOR)
    log_term = np.log(log_arg)

    log_NOE = a4 + a4_AT * (inner ** exponent) * log_term
    return np.exp(log_NOE)
