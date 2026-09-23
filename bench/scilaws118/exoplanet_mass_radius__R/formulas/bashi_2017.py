"""Bashi et al. (2017) two-segment continuous piecewise power law for the
exoplanet mass-radius relation.

Citation: Bashi et al., A&A 604, A83, Aug 2017, DOI 10.1051/0004-6361/201629922.
PDF pages verified: 1 (abstract), 6 (§4 Results).

Formula (abstract, PDF p. 1; §4 Results, PDF p. 6):
    R_p = C_small * M_p^alpha_small     for M_p <= M_break
    R_p = C_large * M_p^alpha_large     for M_p >  M_break

with continuity at the breakpoint (implicit in the paper's parameterisation):
    R_break = C_small * M_break^alpha_small = C_large * M_break^alpha_large

so the prefactors follow from the four published constants:
    C_small = R_break * M_break^(-alpha_small)
    C_large = R_break * M_break^(-alpha_large)

LAW_CONSTANTS — paper-published values (abstract, PDF p. 1; §4, PDF p. 6):
    alpha_small = 0.55  : M-R slope for small planets (M_p <= 124 M_earth)
    alpha_large = 0.01  : M-R slope for large planets (M_p > 124 M_earth)
    M_break     = 124.0 : breakpoint mass [M_earth]
    R_break     = 12.1  : breakpoint radius [R_earth]

    Direct PDF quotes:
      "R ∝ M^0.55±0.02 and R ∝ M^0.01±0.02 for small and large planets"
      — PDF p. 1 abstract lines 20-21
      "occurs at a mass of 124 ± 7 M_earth and a radius of 12.1 ± 0.5 R_earth"
      — PDF p. 1 abstract lines 20-21; confirmed §4, PDF p. 6 lines 276-278

OTHER_CONSTANTS — structural operator constants (not separate paper fits):
    (none — the continuity prefactors C_small and C_large are computed
     inline from the four LAW_CONSTANTS; no external physical constant needed)

Type designation: Type I — one universal formula for all objects; no per-planet
or per-cluster parameters. LOCAL_FITTABLE is empty ({}).

Column mapping (paper notation -> released CSV):
    M_p (M_earth) -> column "M" (input)
    R_p (R_earth) -> column "R" (target, column 0)

Caveats:
- The paper validates on 274 transiting exoplanets spanning 0.07-6945 M_earth
  (PDF p. 3 §2). The released benchmark CSV extends into the stellar regime
  (~9e6 M_earth); alpha_large ~ 0.01 extrapolation there returns values close
  to R_break for all very-high-mass objects.
- The paper states R ∝ M^0.01 (effectively flat), not R ∝ M^(-0.04) as some
  other literature sources suggest. The benchmark ships this paper's value exactly.
"""

import numpy as np

USED_INPUTS = ["M"]
PAPER_REF = "summary_formula+dataset_bashi_2017.md"
EQUATION_LOC = "Abstract, PDF p. 1; §4 Results, PDF p. 6"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "alpha_small": 0.55,   # PDF p. 1 abstract: "R ∝ M^0.55±0.02 for small planets"
    "alpha_large": 0.01,   # PDF p. 1 abstract: "R ∝ M^0.01±0.02 for large planets"
    "M_break": 124.0,      # PDF p. 1 abstract: "transition at 124 ± 7 M_earth"
    "R_break": 12.1,       # PDF p. 1 abstract: "radius of 12.1 ± 0.5 R_earth"
}

# === OTHER_CONSTANTS — none (empty) ===
OTHER_CONSTANTS = {}       # no external physical constants needed; continuity
                           # prefactors C_small / C_large derived inline from LAW

LOCAL_FITTABLE = {}        # Type I — no per-cluster parameters


def predict(X: np.ndarray, alpha_small: float, alpha_large: float,
            M_break: float, R_break: float) -> np.ndarray:
    """Predicted radius R (R_earth) for each object.

    X: (n, 1) — column M (mass in M_earth).
    Params: LAW_CONSTANTS keys passed as keyword arguments by the harness.

    Continuity-implied prefactors (computed from published LAW_CONSTANTS):
        C_small = R_break * M_break^(-alpha_small)
        C_large = R_break * M_break^(-alpha_large)
    """
    M = np.asarray(X[:, 0], dtype=float)
    # Continuity-implied scale factors
    C_small = R_break * (M_break ** (-alpha_small))
    C_large = R_break * (M_break ** (-alpha_large))
    R_small = C_small * np.power(M, alpha_small)
    R_large = C_large * np.power(M, alpha_large)
    return np.where(M <= M_break, R_small, R_large)
