"""Cardoso et al. 2020 — GLM baseline for arthropod species richness.

Cardoso P, Branco VV, Borges PAV, Carvalho JC, Rigal F, Gabriel R, Mammola S,
Cascalho J, Correia L (2020) "Automated Discovery of Relationships, Models, and
Principles in Ecology." Frontiers in Ecology and Evolution 8:530135.
DOI: 10.3389/fevo.2020.530135. Case Study 2 "Modeling Species Richness", PDF p. 4.

Formula (PDF p. 4, display equation in "Modeling Species Richness" section)
---------------------------------------------------------------------------
Multi-model AICc inference across five cross-validation folds selects a
Generalized Linear Model with Poisson error distribution and log link:

    S = exp(a + b*H - c*P - d*D)

where:
  S — arthropod species richness per 1-ha pitfall trap site (count)
  H — altitude above sea level (m)
  P — annual precipitation (mm/yr)
  D — the Cardoso et al. (2013) disturbance index (dimensionless, 0-100 scale)

Sign convention: a, b, c, d are positive values; the formula carries explicit
minus signs for P and D, encoding their negative effects on species richness.

LAW_CONSTANTS (PDF p. 4, paragraph beginning "a, b, c, and d are fitting parameters")
---------------------------------------------------------------------------------------
Published mean values across 5 k-fold cross-validation fits:
  a = 1.894    (range 1.116-2.577)   — log-scale intercept
  b = 0.00419  (range 0.00360-0.00574)  — altitude coefficient
  c = 0.000972 (range 0.000726-0.001212) — precipitation coefficient
  d = 0.0251   (range 0.0118-0.0331)  — disturbance coefficient

Mean training R2 = 0.529 (range 0.469-0.573).
Mean testing  R2 = 0.528 (range 0.313-0.770).

All four coefficients are the paper's own published cross-fold mean calibration
values. They constitute the science claim of the GLM model.

OTHER_CONSTANTS
---------------
None — the exponential functional form is a distributional assumption (Poisson
log link), not a universal physics constant. No unit conversions needed.

Type designation
----------------
Type I — 52 independent pitfall-trap sites on Terceira Island; no cluster
structure; no per-cluster refit. LOCAL_FITTABLE is empty.

Column mapping (paper notation -> released CSV column names)
------------------------------------------------------------
  H -> h  (altitude in m,  USED_INPUTS[0])
  P -> p  (precipitation mm/yr, USED_INPUTS[1])
  D -> d  (disturbance index dimensionless, USED_INPUTS[2])
  S -> S_terceira  (target, column 0)

Caveats
-------
- Slope (sl) and temperature (t) were candidate variables in the SR/GLM runs
  but not retained by AICc selection in any cross-validation fold (PDF p. 4).
  They are present in the released CSV but not consumed by this formula.
"""

import numpy as np

USED_INPUTS = ["h", "p", "d"]
PAPER_REF   = "summary_formula_dataset_cardoso_2020.md"
EQUATION_LOC = (
    "Cardoso 2020 'Modeling Species Richness' section, PDF p. 4, "
    "display equation 'S = e^(a+bH-cP-dD)'"
)

LAW_CONSTANTS = {
    "a": 1.894,       # PDF p. 4 — log-scale intercept (cross-fold mean)
    "b": 0.00419,     # PDF p. 4 — altitude coefficient (cross-fold mean)
    "c": 0.000972,    # PDF p. 4 — precipitation coefficient (cross-fold mean)
    "d": 0.0251,      # PDF p. 4 — disturbance coefficient (cross-fold mean)
}
OTHER_CONSTANTS = {}  # log link is a distributional form, no extra constants
LOCAL_FITTABLE  = {}  # Type I — no per-cluster parameters


def predict(X: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Predict arthropod species richness S from altitude, precipitation, disturbance.

    X: (n, 3) — columns h (altitude m), p (precipitation mm/yr), d (disturbance).
    a, b, c, d: LAW_CONSTANTS (published GLM cross-fold mean values, or refitted).
    Returns S_terceira (count, float) as exp(a + b*h - c*p - d*D).
    """
    h    = np.asarray(X[:, 0], dtype=float)
    p    = np.asarray(X[:, 1], dtype=float)
    dist = np.asarray(X[:, 2], dtype=float)
    return np.exp(a + b * h - c * p - d * dist)
