"""Wang & Tang (2014) one-parameter Budyko model derived from generalised proportionality.

Wang, D. and Tang, Y. (2014). A one-parameter Budyko model for water balance
captures emergent behavior in darwinian hydrologic models. Geophysical Research
Letters, 41(13), 4569–4577. DOI: 10.1002/2014GL060509.
Eq. (10), PDF p. 5 (journal p. 4573).

Formula
-------
    E/P = [1 + Ep/P - sqrt((1 + Ep/P)^2 - 4*epsilon*(2-epsilon)*Ep/P)]
          / [2*epsilon*(2-epsilon)]

where Ep/P is the aridity index (dimensionless) and epsilon (ε) is a single
free parameter in (0, 1] representing the ratio of initial evaporation ratio
(lambda) to the Horton index (H = E/W).

LAW_CONSTANTS — frozen paper values
--------------------------------------
- epsilon = 0.55: published best fit for 246 MOPEX US watersheds.
  Figure 2b caption, PDF p. 5 (journal p. 4574): "best fit curve of equation
  (10) where epsilon = 0.55". The MOPEX dataset is the closest published
  calibration context to the MACH US catchment benchmark.
  (Alternative global value: epsilon = 0.58 for ~470 global watersheds,
  Figure 2a caption, PDF p. 5, journal p. 4574.)

OTHER_CONSTANTS — universal factors
--------------------------------------
None. The integers 1, 2, and 4 in Eq. (10) are fixed structural constants
(inline numerals); no universal physics constants are needed.

Type designation
----------------
Type I — epsilon is a single scalar fitted globally across the dataset, not
per catchment. LOCAL_FITTABLE is empty.

Column mapping
--------------
  Ep/P (phi) -> pet_over_p  (aridity index, dimensionless = PET/P)
  epsilon    -> epsilon      (ratio of initial evaporation to total; see LAW)

Caveats
-------
- epsilon must satisfy 0 < epsilon <= 1 for the formula to be defined.
  At epsilon = 0 the denominator 2*epsilon*(2-epsilon) = 0; the physical
  range avoids this. At epsilon = 1 the formula reduces to the strict upper
  Budyko bound.
- The formula is valid for mean annual water balance at watershed scale
  assuming negligible interannual storage change.
- The PDF for wang_tang_2014 is image-based (pdf_to_text output is sparse);
  the epsilon = 0.55 value is confirmed by the summary_formula_wang_2014.md
  which documents the Figure 2b caption directly from the PDF visual read.
"""

import numpy as np

USED_INPUTS = ["pet_over_p"]
PAPER_REF = "summary_formula_wang_2014.md"
EQUATION_LOC = "Eq. 10, PDF p. 5 (journal p. 4573, Geophys. Res. Lett. 41:4569-4577, 2014)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "epsilon": 0.55,
    # epsilon = 0.55: published best fit for 246 MOPEX US watersheds.
    # Figure 2b caption, PDF p. 5 (journal p. 4574): "best fit curve of
    # equation (10) where epsilon = 0.55". This is the closest published
    # calibration to the MACH US catchment context.
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}  # Integers 1, 2, 4 in Eq. (10) are structural inline numerals
LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, epsilon: float = 0.55, **params) -> np.ndarray:
    """Predict et_over_p using the Wang & Tang (2014) one-parameter Budyko model.

    X: (n_rows, 1) array — column 0 = pet_over_p (aridity index phi = Ep/P).
    epsilon: ratio of initial evaporation to Horton index (LAW_CONSTANT;
             default = 0.55 for MOPEX US watersheds; physical range (0, 1]).
    Returns: (n_rows,) array of predicted et_over_p in (0, 1).
    """
    phi = np.asarray(X[:, 0], dtype=float)
    denom = 2.0 * epsilon * (2.0 - epsilon)
    discriminant = (1.0 + phi) ** 2 - 4.0 * epsilon * (2.0 - epsilon) * phi
    # Clamp discriminant to avoid negative sqrt due to floating point
    discriminant = np.maximum(discriminant, 0.0)
    return (1.0 + phi - np.sqrt(discriminant)) / denom
