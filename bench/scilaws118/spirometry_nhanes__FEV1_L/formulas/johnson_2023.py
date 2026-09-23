"""Johnson & Johnson (2023) spirometry reference equation — FEV1 (litres).

Type: Type I — each row is an independent individual. LOCAL_FITTABLE = {}.

Johnson DC, Johnson BG (2023). Spirometry Reference Equations Including
Existing and Novel Parameters. The Open Respiratory Medicine Journal, 17,
e187430642212260. DOI: 10.2174/18743064-v16-e221227-2022-14.

Closed-form predicted-mean equation for the non-ratio parameters (FEV1, FVC,
FEV3, ...), Tables 1a/1b, table note (PDF pp. 4-6):

    FEV1 = b0 + b1*age + b2*age^2 + b3*height^2

age in years (fractional = months/12 at exam), height in centimetres.

STRATUM SCOPING (2026-05-29 audit re-build)
-------------------------------------------
Johnson 2023 publishes this same four-coefficient form for EIGHT demographic
strata (sex x ethnicity x age-group). This benchmark task is scoped to ONE
stratum so that the law is a clean four-constant closed form rather than a
32-constant per-segment lookup table (which is empirical aggregation, not a
symbolic law — see VERDICT.md "Audit re-build — 2026-05-29").

The single stratum is:
    Female, Caucasian/Mexican-American, adult (age >= 18)
    -> Table 1b, row "FEV1", block "Caucasian/Mexican-American >= 18 year of
       age (N = 2113)".

The data slice (data/{train,test}.csv) is filtered to exactly this stratum, so
sex and ethnicity are constant across all rows and are not inputs; the only
inputs are age and height.

LAW_CONSTANTS — paper-published, frozen (Table 1b, PDF p. 5, FEV1 row)
---------------------------------------------------------------------
    b0 =  0.777        intercept (L)
    b1 = -0.00921      linear age coefficient (L/yr); negative -> FEV1 declines
                       with age in adults
    b2 = -0.0001374    quadratic age coefficient (L/yr^2); small curvature
    b3 =  0.00010647   height^2 coefficient (L/cm^2); positive -> taller -> higher

Calibrated on NHANES III (1988-1994). Applied here as a frozen out-of-cohort
baseline to the NHANES 2007-2012 validation data.

OTHER_CONSTANTS — none (single adult stratum; no age-break branch).
LOCAL_FITTABLE  — none (Type I).
"""

import numpy as np

USED_INPUTS = ["age", "height_cm"]
PAPER_REF = "summary_formula_johnson_2023.md"
EQUATION_LOC = (
    "Johnson & Johnson 2023, Table 1b, row 'FEV1', block "
    "'Caucasian/Mexican-American >= 18 year of age (N=2113)', PDF p. 5; "
    "form: FEV1 = b0 + b1*age + b2*age^2 + b3*height^2 (table note; Section 2.6, PDF p. 3)."
)

# LAW_CONSTANTS — Female / CauMexAm / adult FEV1 row, Table 1b (PDF p. 5).
LAW_CONSTANTS = {
    "b0":  0.777,
    "b1": -0.00921,
    "b2": -0.0001374,
    "b3":  0.00010647,
}

OTHER_CONSTANTS = {}      # single adult stratum — no structural branch constants

LOCAL_FITTABLE = {}       # Type I — no per-cluster fitting


def predict(X: np.ndarray, b0: float, b1: float, b2: float, b3: float) -> np.ndarray:
    """Predict FEV1 (litres) from age and height.

    X: (n, 2) — columns [age, height_cm] per USED_INPUTS.
      age       — fractional years at exam
      height_cm — standing height in cm

    LAW coefficients (b0..b3) arrive as named params via predict(X, **LAW_CONSTANTS).
    The ^2 powers on age and height are structural (the form), kept inline.

    Returns: (n,) array of predicted FEV1 in litres.
    """
    X = np.asarray(X, dtype=float)
    age = X[:, 0]
    height = X[:, 1]
    return b0 + b1 * age + b2 * age * age + b3 * height * height
