"""Large frequency separation scaling relation — Kjeldsen & Bedding (1995), Eq. 9.

Citation:
    Kjeldsen, H. & Bedding, T. R. (1995), "Amplitudes of stellar oscillations:
    the implications for asteroseismology,"
    Astronomy & Astrophysics 293, 87–106.
    arXiv: astro-ph/9403015.
    PDF: reference/KjeldsenBedding1995.pdf  (Eq. 9, §4.1, PDF p. 7).

Formula (Kjeldsen & Bedding 1995, Eq. 9, §4.1, PDF p. 7):

    Δν₀ = 134.9 μHz × (M / M☉)^(1/2) × (R / R☉)^(-3/2)

Physical derivation:
    Starting from the acoustic travel-time integral
        Δν₀ ≃ (2 ∫₀^R dr / c_s)⁻¹
    with adiabatic sound speed c_s² ∝ T and mean temperature ⟨T⟩ ∝ M/R
    (virial theorem approximation), one obtains
        Δν₀ ∝ (M/R³)^(1/2) = ρ̄^(1/2)
    Normalising to the observed solar value Δν₀,⊙ = 134.9 μHz
    (Toutain & Fröhlich 1992, cited on PDF p. 6) gives Eq. 9.
    The exponents 1/2 and −3/2 are structurally fixed; only 134.9 is
    an empirical calibration constant anchored to the Sun.

LAW_CONSTANTS:
    None — the only constant 134.9 μHz is the literally-solar Δ₀,⊙
    (Toutain & Fröhlich 1992; the paper plugs in the measured solar value, it
    does NOT fit it as a coefficient). A standard solar reference / structural
    normalisation → OTHER, not a defining coefficient. (The same value is
    filed as OTHER in this bank's sibling baseline brown_1991.) The exponents
    1/2 and -3/2 are structural → inline.

OTHER_CONSTANTS:
    delta_nu_0_solar = 134.9   (μHz; solar Δ₀,⊙ from Toutain & Fröhlich 1992,
                                  cited Kjeldsen & Bedding 1995 PDF p. 6 —
                                  a solar normalisation the formula consumes)

Type designation:
    Type I — every symbol is either an observed input (M, R) or a fixed
    universal constant (134.9, 1/2, −3/2).  No per-cluster or per-star
    fitted parameters.

Column mapping:
    M_solar  (M / M☉)   stellar mass in solar units    — INPUT 0
    R_solar  (R / R☉)   stellar radius in solar units  — INPUT 1
    Δν₀      [μHz]      large frequency separation     — OUTPUT / target

⚠ INPUT-COLUMN MISMATCH — IMPORTANT NOTE:
    The released data/train.csv for `red_giant_asteroseismology__delta_nu`
    does NOT contain columns `M_solar` or `R_solar`.  Those columns were
    deliberately withheld by prep_data.py (see §"Dropped columns") because
    releasing M and R alongside Δν would create an A3 tautology: the
    Kjeldsen & Bedding (and Yu+2018 §3.6) scaling relation allows exact
    recovery of the target from M and R with R² ≈ 0.99998.

    Released input columns are: [nu_max, Teff, FeH, phase].

    Consequence: the harness will NOT find columns `M_solar` / `R_solar`
    when it constructs the X matrix; predict() will raise an IndexError /
    KeyError at runtime.  This behaviour is intentional and expected.
    The formula file is kept for reference and for any variant of the
    benchmark that releases the full stellar-parameters split, but it
    CANNOT be executed against the released data as-is.

    Workaround (not implemented here): M and R can be estimated via the
    inverted asteroseismic scaling relations:
        R ∝ (nu_max / nu_max,☉) × (Δν / Δν₀)^(−2) × sqrt(Teff / 5777)
        M ∝ (nu_max / nu_max,☉)^3 × (Δν / Δν₀)^(−4) × (Teff / 5777)^(3/2)
    but this uses the TARGET Δν in the computation of the inputs, making
    such a proxy circular for SR evaluation purposes.
"""

from typing import Any

import numpy as np

USED_INPUTS = ["M_solar", "R_solar"]

PAPER_REF = "summary_formula_kjeldsen_1995.md"

EQUATION_LOC = "Eq. 9, §4.1, PDF p. 7 (Kjeldsen & Bedding 1995, A&A 293, 87)"

# Solar large frequency separation — a literally-solar reference value the
# formula normalises to (Toutain & Fröhlich 1992 measured Δ₀,⊙, cited on PDF
# p. 6 of Kjeldsen & Bedding 1995). The paper plugs in this known solar value;
# it does not fit it as a coefficient → OTHER (a given), not LAW. The exponents
# 1/2 and -3/2 are structural and stay inline.
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {}   # no fitted defining coefficient; 134.9 is the solar Δ₀,⊙ given (OTHER)

# === OTHER_CONSTANTS — solar normalisation (given) ===
OTHER_CONSTANTS = {
    "delta_nu_0_solar": 134.9,   # μHz; solar Δ₀,⊙ (Toutain & Fröhlich 1992; KjB95 PDF p. 6)
}   # M and R are supplied as dimensionless solar ratios

LOCAL_FITTABLE = {}    # Type I: no per-cluster parameters
# Non-executable on the released CSV (M_solar, R_solar withheld for A3 tautology
# prevention); excluded from the runnable ladder via
# metadata.yaml::references.kjeldsen_bedding_1995.executable=false and the
# all-NaN guard in predict(). See the docstring's INPUT-COLUMN MISMATCH note.


def predict(
    X: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """Predict Δν₀ = delta_nu_0_solar × M_solar^(1/2) × R_solar^(-3/2).

    Parameters
    ----------
    X : np.ndarray, shape (n, 2)
        Columns in USED_INPUTS order:
          0 = M_solar  (M / M☉, dimensionless)
          1 = R_solar  (R / R☉, dimensionless)

        ⚠ The released benchmark CSV does NOT contain these columns.
          When called with the released [nu_max, Teff, FeH, phase] matrix
          (only 4 columns, none of which is M_solar / R_solar), predict
          short-circuits to an all-NaN result instead of raising IndexError,
          so eval harnesses can treat this `reference_only` baseline
          uniformly with runnable baselines.

    delta_nu_0_solar (the solar Δ₀,⊙ = 134.9 μHz) is read from OTHER_CONSTANTS;
    it is a solar reference the formula normalises to, not a fitted coefficient.
    LAW_CONSTANTS is empty (no fitted defining coefficient); the harness passes
    **LAW_CONSTANTS = **{} which the **kwargs absorbs. The exponents 1/2 and
    -3/2 are structural and stay inline.

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted Δν₀ [μHz].  Strictly positive for positive M and R.
        All-NaN when invoked against the released CSV (M_solar / R_solar
        absent), per RUNNABILITY = "reference_only".

    Validation anchors (PDF p. 7, §4.1 and §5.1):
        - Solar (M=1, R=1):        predict → 134.9 μHz  (exact, by construction)
        - α Cen A (M=1.09, R=1.21): predict → ~105.8 μHz
          (paper reports agreement with Edmonds et al. 1992 detailed models
          at 107.9 μHz; the small difference reflects non-solar internal
          structure not captured by the simple scaling).
    """
    X = np.asarray(X)
    if X.ndim < 2 or X.shape[1] != 2:
        # USED_INPUTS = ["M_solar", "R_solar"] requires exactly 2 columns;
        # the released CSV does not include them (A3 tautology prevention),
        # and any other column-count means the caller did not supply the
        # expected M/R pair. Return NaN rather than silently mis-interpret
        # whichever columns happen to be at positions 0–1. See RUNNABILITY.
        return np.full(X.shape[0], np.nan, dtype=float)
    delta_nu_0_solar = OTHER_CONSTANTS["delta_nu_0_solar"]
    M_solar = np.asarray(X[:, 0], dtype=float)   # M / M☉
    R_solar = np.asarray(X[:, 1], dtype=float)   # R / R☉
    return float(delta_nu_0_solar) * (M_solar ** 0.5) * (R_solar ** (-1.5))
