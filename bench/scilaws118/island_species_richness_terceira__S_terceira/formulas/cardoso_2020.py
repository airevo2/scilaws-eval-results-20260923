"""Cardoso et al. 2020 — symbolic regression formula for arthropod species richness.

Cardoso P, Branco VV, Borges PAV, Carvalho JC, Rigal F, Gabriel R, Mammola S,
Cascalho J, Correia L (2020) "Automated Discovery of Relationships, Models, and
Principles in Ecology." Frontiers in Ecology and Evolution 8:530135.
DOI: 10.3389/fevo.2020.530135. Case Study 2 "Modeling Species Richness", PDF p. 4.

Formula (PDF p. 4, display equation in "Modeling Species Richness" section)
---------------------------------------------------------------------------
The Eureqa symbolic regression run on the 52-site Terceira arthropod dataset
returns, in 23 of 25 independent cross-validation runs, the sparse functional form:

    S = (a / D) - b

where:
  S — arthropod species richness per 1-ha pitfall trap site (count)
  D — the Cardoso et al. (2013) disturbance index (dimensionless, 0-100 scale;
      higher = more disturbed, i.e. further from native forest)

LAW_CONSTANTS (PDF p. 4, paragraph beginning "where a and b were fitting parameters")
---------------------------------------------------------------------------------------
Published mean values across 25 SR runs (5-fold cross-validation):
  a = 140.787   (range 134.700-145.775)
  b = 1.325     (range 1.078-1.483)

Mean training R2 = 0.603 (range 0.576-0.644).
Mean testing  R2 = 0.601 (range 0.449-0.737).

Both a and b are the paper's own published calibration values for this dataset.
They are the science claim — the disturbance-richness relationship expressed
through these two scalars.

OTHER_CONSTANTS
---------------
None — the formula S = a/D - b has no unit conversion factors or universal
physics constants. The disturbance index is dimensionless.

Type designation
----------------
Type I — 52 independent pitfall-trap sites on Terceira Island; no cluster
structure; no per-cluster refit. LOCAL_FITTABLE is empty.

Column mapping (paper notation -> released CSV column names)
------------------------------------------------------------
  D -> d  (disturbance, USED_INPUTS[0])
  S -> S_terceira  (target, column 0)

Caveats
-------
- Formula is undefined at D = 0. All 52 Terceira sites have D > 0 (minimum
  D ≈ 14.7), so the domain is safe.
- At D = a/b ≈ 106.2, the formula predicts S = 0 exactly. For D > 106.2,
  S < 0, which is not biologically meaningful. No site in the dataset exceeds
  D = 77, so this boundary is outside the empirical range.
"""

import numpy as np

USED_INPUTS = ["d"]
PAPER_REF   = "summary_formula_dataset_cardoso_2020.md"
EQUATION_LOC = (
    "Cardoso 2020 'Modeling Species Richness' section, PDF p. 4, "
    "display equation 'S = (a/D) − b'"
)

LAW_CONSTANTS = {
    "a": 140.787,   # PDF p. 4 — mean SR coefficient (disturbance-scaling factor)
    "b": 1.325,     # PDF p. 4 — mean SR intercept offset
}
OTHER_CONSTANTS = {}  # no unit conversion factors; dimensionless inputs
LOCAL_FITTABLE  = {}  # Type I — no per-cluster parameters


def predict(X: np.ndarray, a: float, b: float) -> np.ndarray:
    """Predict arthropod species richness S from disturbance D.

    X: (n, 1) — column d (disturbance index, dimensionless).
    a, b: LAW_CONSTANTS (published SR mean values, or refitted by harness).
    Returns S_terceira (count, float).
    """
    d = np.asarray(X[:, 0], dtype=float)
    return a / d - b
