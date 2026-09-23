"""Chave et al. (2014) pantropical allometric model for tropical tree AGB.

Chave J et al. (2014). "Improved allometric models to estimate the aboveground
biomass of tropical trees." Global Change Biology, 20, 3177-3190.
DOI: 10.1111/gcb.12629. PDF page 6 (journal page 3182).

Formula (Eq. 4, PDF p. 6, journal p. 3182):

    AGB_est = 0.0673 * (rho * D**2 * H)**0.976

Where:
    AGB_est = estimated oven-dry aboveground biomass (kg)
    rho     = wood specific gravity (g/cm^3)
    D       = trunk diameter at breast height (cm)
    H       = total tree height (m)

Published context (paraphrased from PDF, journal p. 3182 — AIC values are
positive as printed; the paper's convention is that greater AIC = worse fit):
    AGB_est = 0.0673 * (rho * D^2 * H)^0.976 (sigma = 0.357; AIC = 3130; df = 4002) (4)

LAW_CONSTANTS — paper-published calibration, frozen
----------------------------------------------------
Both values are from Eq. (4), PDF p. 6 (journal p. 3182), Chave et al. 2014:
    c     = 0.0673   (leading coefficient)
    alpha = 0.976    (exponent)

OTHER_CONSTANTS — none needed
------------------------------
The formula is dimensionally complete as published; no universal physics
constants or unit-conversion factors are required.

Type designation: Type I — the two published constants (c, alpha) are a
single global pantropical calibration with no per-site refit. Each tree
row is independent. LOCAL_FITTABLE is empty; no fit() is defined.

Column mapping (raw CSV -> released CSV -> formula):
    Dry.total.AGB(kg)    -> agb_kg        (output; column 0)
    Wood.specific.gravity -> wood_density  (rho; column 1)
    DBH(cm)              -> diameter_cm   (D;   column 2)
    Total.height(m)      -> height_m      (H;   column 3)

Caveats:
    - The sigma = 0.357 is the residual standard error on ln(AGB), indicating
      substantial multiplicative scatter about the power-law model. The
      pantropical fit applies across tropical biomes; test-set OOD on large
      trees (D > 28.5 cm) probes sparser calibration territory.
    - Eq. (5) in the same paper uses a constrained exponent = 1.0:
      AGB_est = 0.0559 * (rho * D^2 * H). It is implemented as chave_2014_eq5.py.
"""

import numpy as np

USED_INPUTS = ["wood_density", "diameter_cm", "height_m"]
PAPER_REF   = "summary_formula+dataset_chave_2014.md"
EQUATION_LOC = "Eq. 4, PDF p. 6 (journal p. 3182), Chave et al. 2014"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "c":     0.0673,   # leading coefficient; Eq. (4) PDF p. 6
    "alpha": 0.976,    # exponent;            Eq. (4) PDF p. 6
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}   # no unit-conversion or universal-physics factors needed
LOCAL_FITTABLE  = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, c: float, alpha: float) -> np.ndarray:
    """AGB (kg) for each tropical tree under the Chave 2014 pantropical Eq. 4.

    X: (n, 3) — columns wood_density (rho, g/cm^3), diameter_cm (D, cm),
                           height_m (H, m).
    """
    rho = np.asarray(X[:, 0], dtype=float)
    D   = np.asarray(X[:, 1], dtype=float)
    H   = np.asarray(X[:, 2], dtype=float)
    return c * np.power(rho * D**2 * H, alpha)
