"""Chave et al. (2014) constrained pantropical allometric model (Eq. 5).

Chave J et al. (2014). "Improved allometric models to estimate the aboveground
biomass of tropical trees." Global Change Biology, 20, 3177-3190.
DOI: 10.1111/gcb.12629. PDF page 6 (journal page 3182).

Formula (Eq. 5, PDF p. 6, journal p. 3182):

    AGB_est = 0.0559 * (rho * D**2 * H)

This is the constrained-exponent alternative with alpha fixed at 1.0
(linear in the product rho * D^2 * H). AIC = 3211 (slightly worse than Eq. 4
AIC = 3130; the paper's convention is that greater AIC = worse fit, PDF p. 6).

LAW_CONSTANTS — paper-published calibration, frozen
----------------------------------------------------
    c = 0.0559   (leading coefficient; Eq. (5) PDF p. 6, journal p. 3182)

The exponent 1.0 (alpha = 1) is the structural constraint of this model
variant and is encoded as an inline numeral.

OTHER_CONSTANTS — none needed
------------------------------
No universal physics constants or unit-conversion factors required.

Type designation: Type I — single global pantropical calibration.

Column mapping (raw CSV -> released CSV -> formula):
    Dry.total.AGB(kg)     -> agb_kg       (output; column 0)
    Wood.specific.gravity -> wood_density  (rho; column 1)
    DBH(cm)               -> diameter_cm  (D;   column 2)
    Total.height(m)       -> height_m     (H;   column 3)

Caveats:
    - This model is provided as a second reference baseline to complement
      Eq. (4). It is structurally simpler (exponent forced to 1) and
      generally fits slightly worse than Eq. (4) on held-out data.
"""

import numpy as np

USED_INPUTS = ["wood_density", "diameter_cm", "height_m"]
PAPER_REF   = "summary_formula+dataset_chave_2014.md"
EQUATION_LOC = "Eq. 5, PDF p. 6 (journal p. 3182), Chave et al. 2014"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "c": 0.0559,   # leading coefficient; Eq. (5) PDF p. 6, journal p. 3182
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}   # no unit-conversion or universal-physics factors needed
LOCAL_FITTABLE  = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, c: float) -> np.ndarray:
    """AGB (kg) for each tropical tree under the Chave 2014 constrained Eq. 5.

    X: (n, 3) — columns wood_density (rho, g/cm^3), diameter_cm (D, cm),
                           height_m (H, m).
    """
    rho = np.asarray(X[:, 0], dtype=float)
    D   = np.asarray(X[:, 1], dtype=float)
    H   = np.asarray(X[:, 2], dtype=float)
    return c * (rho * D**2 * H)
