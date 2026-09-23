"""Johnson & Johnson (2023) spirometry reference equation -- FVC (litres).

Type: Type I -- each row is an independent individual. LOCAL_FITTABLE = {}.

Johnson DC, Johnson BG (2023). Spirometry Reference Equations Including
Existing and Novel Parameters. The Open Respiratory Medicine Journal, 17,
e187430642212260. DOI: 10.2174/18743064-v16-e221227-2022-14.

Closed-form predicted-mean equation for the non-ratio parameters (FVC, FEV1,
FEV3, ...), Tables 1a/1b, table note (PDF pp. 4-6):

    FVC = b0 + b1*age + b2*age^2 + b3*height^2

age in years (fractional = months/12 at exam), height in centimetres.

STRATUM SCOPING (2026-05-30 audit re-build)
-------------------------------------------
Johnson 2023 publishes this same four-coefficient form for EIGHT demographic
strata (sex x ethnicity x age-group). This benchmark task is scoped to ONE
stratum so that the law is a clean four-constant closed form rather than a
32-constant per-segment lookup table (which is empirical aggregation, not a
symbolic law -- see VERDICT.md "Audit re-build -- 2026-05-30").

The single stratum is:
    Male, Caucasian/Mexican-American, youth (age 8 to <20)
    -> Table 1a, row "FVC", block "Caucasian/Mexican-American < 20 year of
       age (N = 1019)".
It is the highest-fit stratum of all eight on the NHANES 2007-2012 data
(R2 ~ 0.89: the youth window spans childhood lung growth) and shares ethnicity
(Caucasian/Mexican-American) with the sister task spirometry_nhanes__FEV1_L.

The data slice (data/{train,test}.csv) is filtered to exactly this stratum, so
sex and ethnicity are constant across all rows and are not inputs; the only
inputs are age and height.

LAW_CONSTANTS -- paper-published, frozen (Table 1a, PDF pp. 4-5, FVC row)
------------------------------------------------------------------------
    b0 =  0.012        intercept (L)
    b1 = -0.21296      linear age coefficient (L/yr)
    b2 =  0.0112910    quadratic age coefficient (L/yr^2); positive -> FVC
                       accelerates upward through the childhood growth window
    b3 =  0.00017278   height^2 coefficient (L/cm^2); positive -> taller -> higher

Calibrated on NHANES III (1988-1994). Applied here as a frozen out-of-cohort
baseline to the NHANES 2007-2012 validation data.

OTHER_CONSTANTS -- none (single stratum; no age-break branch).
LOCAL_FITTABLE  -- none (Type I).
"""

import numpy as np

USED_INPUTS = ["age", "height_cm"]
PAPER_REF = "summary_formula_johnson_2023.md"
EQUATION_LOC = (
    "Johnson & Johnson 2023, Table 1a, row 'FVC', block "
    "'Caucasian/Mexican-American < 20 year of age (N=1019)', PDF pp. 4-5; "
    "form: FVC = b0 + b1*age + b2*age^2 + b3*height^2 (table note; Section 2.6, PDF p. 3)."
)

# LAW_CONSTANTS -- Male / CauMexAm / youth FVC row, Table 1a (PDF pp. 4-5).
LAW_CONSTANTS = {
    "b0":  0.012,
    "b1": -0.21296,
    "b2":  0.0112910,
    "b3":  0.00017278,
}

OTHER_CONSTANTS = {}      # single stratum -- no structural branch constants

LOCAL_FITTABLE = {}       # Type I -- no per-cluster fitting


def predict(
    X: np.ndarray,
    b0: float,
    b1: float,
    b2: float,
    b3: float,
) -> np.ndarray:
    """Predict FVC (litres) from age and height.

    X: (n, 2) -- columns [age, height_cm] per USED_INPUTS.
      age       -- fractional years at exam
      height_cm -- standing height in cm

    Returns: (n,) array of predicted FVC in litres.
    """
    X = np.asarray(X, dtype=float)
    age = X[:, 0]
    height = X[:, 1]
    return b0 + b1 * age + b2 * age * age + b3 * height * height
