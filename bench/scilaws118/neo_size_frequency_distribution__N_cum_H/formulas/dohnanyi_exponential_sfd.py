"""Dohnanyi collisional-equilibrium SFD, slope fixed at 0.5 — best rung.

A collisionally-relaxed small-body population reaches the Dohnanyi
(1969) equilibrium size distribution: a differential power law
dN/dD ∝ D^(-3.5), equivalently a cumulative magnitude slope of
alpha = 0.5 (each magnitude fainter, ~10^0.5 ≈ 3.16x more objects):

    N(<H) = A * 10^(0.5 * H).

The slope 0.5 is FIXED by collisional-cascade theory (Dohnanyi 1969;
O'Brien & Greenberg 2003); only the amplitude A is fit on the v2 train.

On the held-out test the theory-fixed slope generalises marginally
better than the restricted-range empirical fit of the previous rung
(test log_mae ~ 0.041 vs ~0.044): a first-principles slope acts as a
regulariser where a fit on a limited magnitude range scatters.  The two
exponential rungs are nonetheless within the catalog-scatter noise floor
of each other — the decisive ladder step is the exponential SFD form
itself over the constant rung (log_mae 0.45 -> 0.04).

LAW_CONSTANTS — frozen, pre-fit on v2 train (slope fixed at 1/2)
---------------------------------------------------------------
- LOG_A = -5.8322   (log10 amplitude; A = 10^LOG_A)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).  The 0.5 magnitude slope is the structural
collisional-equilibrium constant.
"""

import numpy as np

USED_INPUTS = ["H_upper"]
PAPER_REF = "summary_formula_bottke_2002.md"
EQUATION_LOC = (
    "Dohnanyi 1969 collisional-equilibrium SFD: cumulative magnitude "
    "slope 0.5 (dN/dD ∝ D^-3.5), N(<H) = A*10^(0.5*H).  Slope fixed at "
    "1/2; amplitude LOG_A pre-fit on v2 train."
)

LAW_CONSTANTS = {
    "LOG_A": -5.8322,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, LOG_A: float = -5.8322) -> np.ndarray:
    """N(<H) = 10^(LOG_A + 0.5*H); Dohnanyi collisional-equilibrium SFD."""
    H = np.asarray(X[:, 0], dtype=float)
    return 10.0 ** (LOG_A + 0.5 * H)
