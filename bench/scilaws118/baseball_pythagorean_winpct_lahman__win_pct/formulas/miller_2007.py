"""Pythagorean won-loss formula from Miller (2007), using Miller's empirical
best-fit gamma.

Miller, S. J. (2007). A derivation of the Pythagorean won-loss formula in
baseball. Chance, 20(1):40-48. American Statistical Association.
Open-access mirror: https://web.williams.edu/Mathematics/sjmiller/public_html/
399/handouts/PythagWonLoss_Paper.pdf

Formula
-------
For season-total runs R (scored) and RA (runs allowed), with the Weibull
location parameter beta negligible relative to season totals:

    win_pct = R^gamma / (R^gamma + RA^gamma)          (simplified Eq. 1.2)

Full formula with beta (Eq. 1.2, PDF p. 1; Theorem 2.2 / Eq. 2.6, PDF p. 4):

    win_pct = (R - beta)^gamma / ((R - beta)^gamma + (RA - beta)^gamma)

For season totals, |beta| ~ 0.5 * G (runs per game) is negligible relative
to R, RA ~ 500-800, so this module sets beta = 0.

LAW_CONSTANTS
-------------
- gamma = 1.79: mean best-fit exponent from the method of least squares
  applied to the 14 American League teams of the 2004 baseball season
  (Miller 2007, abstract and PDF p. 1 / p. 2 first paragraph).
  Standard deviation 0.09. The maximum-likelihood estimate is 1.74 (std 0.06).
  This module uses the least-squares mean 1.79 as the primary LAW constant.

OTHER_CONSTANTS
---------------
None. The formula is dimensionless; no universal physics constants are needed.

Type designation: Type I. The exponent gamma is a universal league-wide
constant; no per-team or per-season fitting is performed in Miller's primary
analysis. LOCAL_FITTABLE = {}.

Column mapping (paper -> CSV):
  RS (paper runs scored per game) -> R  (column 1, season total)
  RA (paper runs allowed per game) -> RA (column 2, season total)
  The formula is scale-invariant: R^gamma/(R^gamma+RA^gamma) =
  (R/G)^gamma / ((R/G)^gamma + (RA/G)^gamma), so season totals and
  per-game averages give the same result.

Caveats:
- Miller fits beta = -0.5 runs/game for the per-game discrete correction;
  at season-total scale (R ~ 600, RA ~ 600, G ~ 162) the correction is
  |beta_season| = 0.5 * 162 ~ 81, which is ~13% of R -- not entirely
  negligible. This module ships gamma frozen at the paper's value and sets
  beta = 0 (season-total approximation), consistent with the canonical
  Bill James formulation. The sister module james_1980.py uses gamma = 2.
"""

import numpy as np

USED_INPUTS  = ["R", "RA"]
PAPER_REF    = "summary_formula_miller_2007.md"
EQUATION_LOC = "Eq. (1.2) PDF p. 1; Theorem 2.2 / Eq. (2.6) PDF p. 4"

LAW_CONSTANTS = {
    "gamma": 1.79,   # least-squares mean over 14 AL 2004 teams; Miller 2007 abstract + PDF p. 1-2
}
OTHER_CONSTANTS = {}   # dimensionless formula; no external physics constants needed
LOCAL_FITTABLE  = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, gamma: float) -> np.ndarray:
    """Pythagorean win percentage under Miller 2007's best-fit gamma.

    X: (n, 2) — columns R (runs scored), RA (runs allowed).
    Returns array of shape (n,) — win_pct in (0, 1).
    """
    R  = np.asarray(X[:, 0], dtype=float)
    RA = np.asarray(X[:, 1], dtype=float)
    Rg  = np.power(R,  gamma)
    RAg = np.power(RA, gamma)
    return Rg / (Rg + RAg)
