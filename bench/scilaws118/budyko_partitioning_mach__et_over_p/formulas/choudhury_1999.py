"""Choudhury (1999) generalised Budyko equation for annual evaporation fraction.

Choudhury, B.J. (1999). Evaluation of an empirical equation for annual
evaporation using field observations and results from a biophysical model.
Journal of Hydrology, 216(1-2), 99–110. DOI: 10.1016/S0022-1694(98)00293-5.
Eq. (3), PDF p. 2 (journal p. 100).

Formula
-------
    E = P / {1 + (P / R_n)^alpha}^(1/alpha)

As the ratio et_over_p = E/P, writing phi = R_n/P (energy-over-water index):

    et_over_p = 1 / (1 + (1/phi)^alpha)^(1/alpha)

where phi = pet_over_p = PET/P (aridity index; PET serves as proxy for R_n,
the water equivalent of annual net radiation, in this benchmark).

LAW_CONSTANTS — frozen paper values
--------------------------------------
- alpha = 1.8: best fit for river basins. Abstract p. 1 (journal p. 99):
  "minimum value of the MAE was 36 mm (5% of the mean evaporation) obtained
  for alpha = 1.8" (confirmed in choudhury_1999.txt line 22).
  The paper also reports alpha = 2.6 for field observations (abstract p. 1,
  line 20: "MAE was 33 mm ... for alpha = 2.6"). For river basins at the
  MACH benchmark spatial scale, alpha = 1.8 is the published global best fit.

OTHER_CONSTANTS — universal factors
--------------------------------------
None. The formula is dimensionally clean; all terms are dimensionless ratios.

Type designation
----------------
Type I — one row per catchment; alpha is a single scalar fitted globally
across the entire dataset, not per catchment. LOCAL_FITTABLE is empty.

Column mapping
--------------
  P   -> ppt_mm_yr   (annual precipitation, mm yr^-1) — not needed directly
  R_n -> pet_mm_yr   (water equivalent of net radiation, mm yr^-1; proxied
                      by Hargreaves PET in the MACH data)
  P/R_n = 1/phi -> 1/pet_over_p (inverse aridity index)
  alpha -> alpha (shape parameter; see LAW_CONSTANTS)

Caveats
-------
- Choudhury (1999) uses R_n (net radiation equivalent, mm yr^-1) as the energy
  input, not Penman PET directly. This benchmark substitutes pet_mm_yr
  (Hargreaves PET) as the closest available proxy; the fitted alpha may differ
  from the published 1.8 due to PET method differences.
- The formula is known to overestimate E for tundra sites (observed 72 mm,
  predicted 137 mm for alpha = 2.6; PDF p. 8, §3). The MACH dataset covers
  continental US catchments, so this caveat is less critical.
- alpha must be > 0 for the formula to be defined.
"""

import numpy as np

USED_INPUTS = ["pet_over_p"]
PAPER_REF = "summary_formula_dataset_choudhury_1999.md"
EQUATION_LOC = "Eq. 3, PDF p. 2 (journal p. 100, Journal of Hydrology 216:99-110, 1999)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "alpha": 1.8,
    # alpha = 1.8: published global best fit for river basins.
    # Abstract p. 1 (journal p. 99): "minimum value of the MAE was 36 mm
    # (5% of the mean evaporation) obtained for alpha = 1.8."
    # Confirmed in choudhury_1999.txt line 22.
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}  # Dimensionally clean; no external physics constants needed
LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, alpha: float = 1.8, **params) -> np.ndarray:
    """Predict et_over_p using the Choudhury (1999) generalised Budyko formula.

    X: (n_rows, 1) array — column 0 = pet_over_p (aridity index phi = PET/P).
    alpha: shape parameter (LAW_CONSTANT; default = 1.8 for river basins).
    Returns: (n_rows,) array of predicted et_over_p in (0, 1).
    """
    phi = np.asarray(X[:, 0], dtype=float)
    # et_over_p = 1 / (1 + (1/phi)^alpha)^(1/alpha)
    return 1.0 / (1.0 + (1.0 / phi) ** alpha) ** (1.0 / alpha)
