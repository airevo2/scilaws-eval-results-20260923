"""Yang et al. (2008) analytical derivation of the mean annual water-energy balance equation.

Yang, H., Yang, D., Lei, Z., and Sun, F. (2008). New analytical derivation of
the mean annual water-energy balance equation. Water Resources Research, 44,
W03410. DOI: 10.1029/2007WR006135.
Eq. (25), PDF p. 4 (Water Resources Research 44, W03410, 2008).

Formula
-------
    E = E_0 * P / (P^n + E_0^n)^(1/n)

As the ratio et_over_p = E/P, writing phi = E_0/P (aridity index):

    et_over_p = phi / (1 + phi^n)^(1/n)
              = 1 / (1 + (1/phi)^n)^(1/n)

where phi = pet_over_p = PET/P (potential evapotranspiration divided by
precipitation). Parameter n > 0 is dimensionless, representing integrated
catchment characteristics (vegetation type, soil water-holding capacity,
root depth, average slope, land use).

LAW_CONSTANTS — frozen paper values
--------------------------------------
- n = 2.0: illustrative value actually plotted in Figure 1 (PDF p. 4).
  Figure 1 shows curves for n = 0.3, 0.5, 1, 2, 5 — n = 2 is explicitly
  depicted. The paper does not publish a single universal global n; the
  calibration on 108 Chinese catchments (§3.3, PDF p. 5) yields
  per-catchment n values; no pooled mean is stated.
  Per the benchmark protocol, n = 2 is chosen as the LAW_CONSTANT because
  it is directly locatable on Figure 1 (PDF p. 4) — the value is plotted
  and labelled in that figure.
  (EQUATION_LOC: n = 2, Figure 1, PDF p. 4.)

OTHER_CONSTANTS — universal factors
--------------------------------------
None. The formula is dimensionally clean; all terms are dimensionless ratios.

Type designation
----------------
Type I — one row per catchment; n is treated as a globally fitted scalar
in this benchmark (one row per catchment precludes per-catchment fitting).
LOCAL_FITTABLE is empty.

Column mapping
--------------
  P   -> ppt_mm_yr   (annual precipitation, mm yr^-1) — not needed directly
  E_0 -> pet_mm_yr   (Penman PET, mm yr^-1) — not needed directly
  phi = E_0/P -> pet_over_p  (aridity index, dimensionless)
  n   -> n           (catchment characteristic parameter; see LAW_CONSTANTS)

Caveats
-------
- Yang (2008) uses Penman PET (Shuttleworth [1993]) for E_0; the MACH
  benchmark uses Hargreaves PET, which may systematically differ, shifting
  the effective n calibration.
- The formula is mathematically equivalent in form to Mezentsev [1955] /
  Choudhury [1999] with parameter n (≡ alpha in Choudhury's notation).
  (Table 1, PDF p. 2 of 9.)
- n must be > 0 for the formula to be defined.
"""

import numpy as np

USED_INPUTS = ["pet_over_p"]
PAPER_REF = "summary_formula_yang_2008.md"
EQUATION_LOC = "Eq. 25, PDF p. 4 (Water Resources Research 44, W03410, 2008); n = 2, Figure 1, PDF p. 4"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "n": 2.0,
    # n = 2.0: directly plotted in Figure 1 (PDF p. 4), which shows
    # curves for n = 0.3, 0.5, 1, 2, 5. The paper calibrates n per
    # catchment on 108 Chinese basins (§3.3, PDF p. 5) but publishes
    # no single global best-fit n for any dataset. n = 2 is chosen as
    # the LAW_CONSTANT because it is explicitly locatable on Figure 1.
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}  # Dimensionally clean; no external physics constants needed
LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, n: float = 2.0, **params) -> np.ndarray:
    """Predict et_over_p using the Yang et al. (2008) analytical Budyko formula.

    X: (n_rows, 1) array — column 0 = pet_over_p (aridity index phi = E0/P).
    n: catchment characteristic parameter (LAW_CONSTANT; default = 2.0;
       n = 2 is explicitly plotted in Figure 1, PDF p. 4; must be > 0).
    Returns: (n_rows,) array of predicted et_over_p in (0, 1).
    """
    phi = np.asarray(X[:, 0], dtype=float)
    # et_over_p = phi / (1 + phi^n)^(1/n)
    return phi / (1.0 + phi ** n) ** (1.0 / n)
