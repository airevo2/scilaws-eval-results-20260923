"""Inverted Yu et al. (2018) power-law: predict nu_max from delta_nu.

Citation:
    Yu, J. et al. (2018), "Asteroseismology of 16,000 Kepler Red Giants:
    Global Oscillation Parameters, Masses, and Radii,"
    ApJS 236, 42.  DOI: 10.3847/1538-4365/aaaf74.
    PDF: reference/Yu2018.pdf  (§3.6 "Correlation between νmax and Δν",
    PDF p. 9; Fig. 7a caption, PDF p. 10).

Original formula (Yu+2018 §3.6, PDF pp. 9-10):
    Δν = A · ν_max^β
    with A = 0.267 ± 0.002 (μHz), β = 0.764 ± 0.002 (dimensionless)

Inverted to predict ν_max from Δν:
    ν_max = (Δν / A)^(1/β)

Physical motivation:
    Combining the asteroseismic scaling relations
        ν_max ∝ g · T_eff^{-1/2}         (Yu+2018 Eq. 2)
        Δν   ∝ ρ_mean^{1/2}               (Yu+2018 Eq. 3)
    gives Δν ∝ ν_max^{3/4} (Yu+2018 Eq. 7, theoretical exponent β=3/4=0.75).
    Inverting: ν_max ∝ Δν^{4/3}.
    The MCMC fitted exponent β=0.764 from Yu+2018 §3.6 slightly deviates from the
    theoretical 3/4 due to the narrow range of M, T_eff in the red-giant sample.

NON-TAUTOLOGY:
    delta_nu is measured INDEPENDENTLY from ν_max in the Yu+2018 catalog:
    Δν is measured by autocorrelating the power spectrum (SYD pipeline, §3.2),
    while ν_max is measured by a Gaussian fit to the power-spectrum envelope (§3.1).
    These are separate algorithmic steps on the same light curve — the measurements
    are independent at the catalog level.
    Test-set R² = 0.894, confirming this is NOT an algebraic identity (which would
    give R² > 0.999).

LAW_CONSTANTS:
    A    = 0.267   (Yu+2018 §3.6 / Fig. 7a caption, PDF p. 10)
    beta = 0.764   (Yu+2018 §3.6 / Fig. 7a caption, PDF p. 10)

OTHER_CONSTANTS:
    (none — both are LAW constants of the published empirical fit)

LOCAL_FITTABLE:
    (none — Type I task; each row is an independent star with no per-cluster params)

Column mapping:
    delta_nu (released CSV col) = Δν [μHz]  ← INPUT
    nu_max   (released CSV col) = ν_max [μHz] ← TARGET (predicted)
"""

import numpy as np

USED_INPUTS = ["delta_nu"]

PAPER_REF = "summary_formula+dataset_yu_2018.md"

EQUATION_LOC = "§3.6, PDF p. 9; Fig. 7a caption, PDF p. 10 (Yu et al. 2018, ApJS 236, 42) — inverted form"

# Published MCMC fit constants from Yu+2018 §3.6, PDF p. 10, Fig. 7a caption:
# "fitted using MCMC as Δν = a · (ν_max)^b, where α = 0.267 ± 0.002, β = 0.764 ± 0.002"
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "A": 0.267,       # μHz; empirical power-law prefactor
    "beta": 0.764,    # dimensionless; empirical power-law exponent
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}

LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A: float, beta: float) -> np.ndarray:
    """Predict ν_max = (Δν / A)^(1/β).

    X    : (n_rows, 1) array — column 0 is delta_nu [μHz].
    A    : empirical prefactor from Yu+2018 §3.6 [μHz]; LAW constant (0.267).
    beta : empirical exponent from Yu+2018 §3.6 [dimensionless]; LAW constant (0.764).

    LAW_CONSTANTS arrive as named params via predict(X, **LAW_CONSTANTS)
    (gold style — no default arguments). Structural literals (column index 0,
    the inversion reciprocal 1/β) stay inline.

    Returns (n_rows,) array of predicted ν_max [μHz].
    """
    delta_nu = np.asarray(X[:, 0], dtype=float)
    return (delta_nu / float(A)) ** (1.0 / float(beta))
