"""York, Rosa & Dietz (2003) STIRPAT log-linear formula for CH4 emissions.

Citation: York R., Rosa E.A. & Dietz T. (2003). "STIRPAT, IPAT and ImPACT:
analytic tools for unpacking the driving forces of environmental impacts."
Ecological Economics 46(3), 351-365. DOI: 10.1016/S0921-8009(03)00188-5.

Functional form (Eq. 1, PDF p. 4 / journal p. 354):

    I = a * P^b * A^c * T^d

Log-linearised (Eq. 2, PDF p. 4 / journal p. 354):

    log(I) = log_a + b*log(P) + c*log(A) + d*log(UR)

Applied to ME with three covariates available in the released CSV:
    P -> TP  (total population)
    A -> GDP (per-capita GDP)
    T proxy -> UR (urbanisation rate)

LAW_CONSTANTS -- Table 1 Model 3, PDF p. 11 / journal p. 361 (N=146, 1996):
    log_a = -9.009   (intercept; absorbs unit offset)
    b      =  0.976  (population elasticity; log TP)
    c      =  0.915  (affluence elasticity; log GDP)
    d      =  0.624  (urbanisation coefficient; log UR)

OTHER_CONSTANTS: none (dimensionally clean log-linear model).

Type designation: Type I. All four parameters are global (York 2003 fits a
single OLS cross-section; no per-country refit). LOCAL_FITTABLE is empty.

Column mapping (paper -> released CSV):
    P (population)     -> TP    [persons]
    A (affluence)      -> GDP   [USD per capita]
    T proxy (urban%)   -> UR    [% urban]

Caveats:
    - York 2003 coefficients were fit on a 1996 WRI/UN cross-section of 146
      nations for CO2 in thousands of metric tons. This module applies the same
      structural STIRPAT form to CH4 emissions in kg. The log_a intercept absorbs
      the unit/species mismatch; frozen paper values produce absolute-scale offset
      but correct structural elasticity shape (OOD extrapolation signal).
    - Model 3 in York 2003 also includes (GDP)^2, %Industry, and Tropical dummy
      -- variables not present in the released CSV. This module uses the three
      available columns (TP, GDP, UR), which is the reduced York 2003 form.
    - Published performance (Table 1, Model 3, PDF p. 11): R^2 = 0.894 on 1996
      CO2 cross-section. Expected to be weaker on the multi-year CH4 panel.
"""

import numpy as np

USED_INPUTS = ["GDP", "TP", "UR"]
PAPER_REF   = "summary_formula_york_2003.md"
EQUATION_LOC = "York 2003 Eq. 1-2, PDF p. 4 / journal p. 354; INIT from Table 1 Model 3, PDF p. 11"

# LAW_CONSTANTS: Model 3 OLS coefficients, Table 1, PDF p. 11 / journal p. 361.
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "log_a": -9.009,
    "b":      0.976,
    "c":      0.915,
    "d":      0.624,
}

# OTHER_CONSTANTS: none.
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}

# Type I: no per-cluster parameters.
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, **params) -> np.ndarray:
    """Evaluate York 2003 STIRPAT for ME in kg CH4.

    X.shape = (n, 3) in USED_INPUTS order: (GDP, TP, UR).
    params = LAW_CONSTANTS (frozen paper values).
    Returns ME in kg CH4 (frozen intercept implies unit-scale mismatch;
    log_a absorbs the CO2-kg-CH4 conversion when refitted on train.csv).

    C11 calibration-crudeness caveat (log-centering mismatch):
        York 2003 Table 1 Model 3 reports a CENTERED-model intercept of -9.009.
        The paper explicitly states (PDF p. 9-10, §4): GDP per capita "is centered by
        subtracting its sample mean in logarithmic form (which equals 8.23 for the CO2
        emissions analysis)" and urbanisation "is centered by subtracting its sample mean
        in logarithmic form (which equals 3.81 for the CO2 emissions analysis)" (Table 1
        footnote, PDF p. 11: "Both percent urban and GDP per capita were centered by
        subtracting their respective means in logarithmic form.").

        This benchmark applies raw (uncentered) log(GDP) and log(UR) inputs directly.
        Converting York's centered intercept to the raw-input form would give:
            log_a_uncentered = -9.009 - 0.915 * 8.23 - 0.624 * 3.81 ≈ -18.92
        i.e. the shipped value -9.009 is York's published *centered* Table-1 Model-3
        intercept, applied here against raw inputs (a documented C11 crudeness).

        Why it is shipped as-is (the deliberately-weak frozen rung):
        1. The coefficients are York 2003's CO2 (kt) values; this task is CH4 in kg, so
           the absolute scale is governed by a CO2->CH4 unit mismatch that dominates any
           centering offset. The frozen -9.009 happens to give CH4-kg magnitudes of the
           right order; a re-centered intercept (-18.92) would predict ~5 orders too low.
        2. The shipped formula's measured OOD-test performance is SMAPE = 1.04,
           pooled R2 = 0.287 (reference_metrics.json) -- a meaningful-but-weak baseline,
           its intended role as the bottom rung of the ladder.

        This formula is shipped as-is per S2-P7 (multi-baseline ladder includes a
        paper-frozen weak baseline). The §9.6 NEGATIVE_POOLED_R2 metric outcome is
        documented in metadata.notes. No code change to predict() is made; this note
        is a transparency annotation only.
    """
    log_a = params.get("log_a", LAW_CONSTANTS["log_a"])
    b     = params.get("b",     LAW_CONSTANTS["b"])
    c     = params.get("c",     LAW_CONSTANTS["c"])
    d     = params.get("d",     LAW_CONSTANTS["d"])

    GDP = np.asarray(X[:, 0], dtype=float)
    TP  = np.asarray(X[:, 1], dtype=float)
    UR  = np.asarray(X[:, 2], dtype=float)

    log_ME = log_a + b * np.log(TP) + c * np.log(GDP) + d * np.log(UR)
    return np.exp(log_ME)
