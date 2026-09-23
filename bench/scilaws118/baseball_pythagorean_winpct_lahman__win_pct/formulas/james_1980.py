"""Pythagorean won-loss formula in the original Bill James (1980) formulation
with the canonical exponent gamma = 2.

James, B. (1980). The Bill James Baseball Abstract. Self-published.
(Citation-only reference; no PDF mirrored. The canonical mathematical
derivation is Miller 2007.)

Formula
-------
Bill James' original "Pythagorean" formula (named for the sum-of-squares form):

    win_pct = R^2 / (R^2 + RA^2)

where R is season runs scored and RA is season runs allowed.
This is the beta = 0 special case of Miller 2007 Eq. (1.2) with gamma = 2.

The formula is cited and reproduced in Miller (2007), abstract and PDF p. 1
(Introduction, first paragraph): "Initially in baseball the exponent gamma
was taken to be 2 (which led to the name)."

LAW_CONSTANTS
-------------
- gamma = 2: the canonical Bill James exponent (1980 Baseball Abstract),
  cited in Miller (2007) PDF p. 1 Introduction paragraph 1.

OTHER_CONSTANTS
---------------
None. The formula is dimensionless.

Type designation: Type I. The exponent gamma = 2 is the universal constant
for this baseline. LOCAL_FITTABLE = {}.

Column mapping (paper -> CSV):
  RS (runs scored per game / season) -> R  (column 1, season total)
  RA (runs allowed per game / season) -> RA (column 2, season total)
  Scale-invariant (as with all Pythagorean formulas; see miller_2007.py).

Caveats:
- The gamma = 2 constant is systematically too large for modern
  scoring environments; the empirical optimum is near 1.82 (Miller).
  This module ships the historical canonical value, not a refitted one.
  It is expected to produce slightly higher RMSE than miller_2007.py.
"""

import numpy as np

USED_INPUTS  = ["R", "RA"]
PAPER_REF    = "summary_formula_miller_2007.md"
EQUATION_LOC = "Bill James (1980); cited in Miller 2007 PDF p. 1 Introduction"

LAW_CONSTANTS = {
    "gamma": 2.0,    # James (1980) canonical exponent; cited in Miller 2007 PDF p. 1
}
OTHER_CONSTANTS = {}   # dimensionless formula
LOCAL_FITTABLE  = {}   # Type I


def predict(X: np.ndarray, gamma: float) -> np.ndarray:
    """Pythagorean win percentage under the original Bill James gamma = 2.

    X: (n, 2) — columns R (runs scored), RA (runs allowed).
    Returns array of shape (n,) — win_pct in (0, 1).
    """
    R  = np.asarray(X[:, 0], dtype=float)
    RA = np.asarray(X[:, 1], dtype=float)
    Rg  = np.power(R,  gamma)
    RAg = np.power(RA, gamma)
    return Rg / (Rg + RAg)
