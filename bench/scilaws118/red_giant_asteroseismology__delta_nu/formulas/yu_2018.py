"""Empirical power-law between Δν and ν_max — Yu et al. (2018), §3.6.

Citation:
    Yu, J. et al. (2018), "Asteroseismology of 16,000 Kepler Red Giants:
    Global Oscillation Parameters, Masses, and Radii,"
    ApJS 236, 42.  DOI: 10.3847/1538-4365/aaaf74.
    PDF: reference/Yu2018.pdf  (§3.6 "Correlation between νmax and Δν",
    PDF p. 9; equation values restated in Fig. 7 caption, PDF p. 10).

Formula (Yu+2018 §3.6, PDF pp. 9–10):
    Δν = A · ν_max^β

with constants from the MCMC fit on the full 16 094-star Kepler red-giant
sample (Yu+2018 Fig. 7a caption, PDF p. 10):
    A    = 0.267 ± 0.002   (μHz)
    β    = 0.764 ± 0.002   (dimensionless)

Physical motivation:
    Combining the two asteroseismic scaling relations
        ν_max ∝ g · T_eff^{-1/2}                          (Yu+2018 Eq. 2)
        Δν   ∝ ρ_mean^{1/2}  ∝  (M / R^3)^{1/2}           (Yu+2018 Eq. 3)
    and eliminating R via R ∝ √(M T_eff^{-1/2} / ν_max), one obtains
        Δν ∝ ν_max^{3/4} · T_eff^{+3/8} · M^{-1/4}
    (Yu+2018 Eq. 7, §3.6 PDF p. 9).  For the narrow red-giant temperature
    and mass range, the M^{-1/4} and T_eff^{+3/8} factors are nearly
    constant; absorbing their mean values into a single prefactor A gives
    the empirical power law above.  The fitted exponent β = 0.764 is
    consistent with the theoretical value 3/4 = 0.750.

LAW_CONSTANTS:
    A    = 0.267   (Yu+2018 §3.6 / Fig. 7a caption, PDF p. 10)
    beta = 0.764   (Yu+2018 §3.6 / Fig. 7a caption, PDF p. 10)

OTHER_CONSTANTS:
    (none — both coefficients are LAW constants of the empirical fit.)

Type designation:
    Type I — each row is an independent Kepler red giant; no per-cluster
    fitted parameters.  The Yu+2018 fit uses a single globally constant
    (A, β) across all 16 094 stars (Fig. 7a).

Column mapping:
    nu_max  (released CSV col)   = ν_max   [μHz]
    Δν      (released CSV col, target) = predicted using A·ν_max^β [μHz]
"""

import numpy as np

USED_INPUTS = ["nu_max"]

PAPER_REF = "summary_formula_dataset_yu_2018.md"

EQUATION_LOC = "§3.6, PDF p. 9; Fig. 7a caption, PDF p. 10 (Yu et al. 2018, ApJS 236, 42)"

# Both coefficients from Yu+2018 §3.6 MCMC fit on the full 16 094-star
# sample.  PDF p. 10, Fig. 7a caption: "fitted using MCMC as Δν = a · (ν_max)^b,
# where α = 0.267 ± 0.002, β = 0.764 ± 0.002".
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "A": 0.267,
    "beta": 0.764,
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}

LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A: float, beta: float) -> np.ndarray:
    """Predict Δν = A · ν_max^β.

    X    : (n_rows, 1) array — column 0 is ν_max [μHz].
    A    : empirical prefactor [μHz];          LAW (Yu+2018 §3.6 MCMC fit).
    beta : empirical exponent [dimensionless]; LAW (Yu+2018 §3.6 MCMC fit).

    The harness supplies A, beta from LAW_CONSTANTS via predict(X, **LAW_CONSTANTS);
    no default arguments (gold style).

    Returns (n_rows,) array of predicted Δν [μHz].
    """
    nu_max = np.asarray(X[:, 0], dtype=float)
    return float(A) * nu_max ** float(beta)
