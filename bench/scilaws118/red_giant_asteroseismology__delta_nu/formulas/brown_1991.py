"""Theoretical asymptotic Δν - ν_max scaling, Brown et al. (1991).

Citation:
    Brown, T. M., Gilliland, R. L., Noyes, R. W., & Ramsey, L. W. (1991),
    "Detection of possible p-mode oscillations on Procyon,"
    ApJ 368, 599-609.  DOI: 10.1086/169725.
    PDF: reference/Brown1991.pdf  (PDF p. 1-2; statement of ν_0 scaling
    ∝ ρ^{1/2} and ν_ac ∝ g · T_eff^{-1/2}).

Formula — theoretical Δν - ν_max - T_eff scaling
=================================================
Combining the two foundational scaling laws stated in Brown+1991 (PDF p. 1-2)
and used universally in subsequent asteroseismology (Belkacem+2011 ν_max ∝ ν_ac
underpinning; Sharma+2016 Eq. 5+6; Yu+2018 Eq. 2+3):

    Δν   ∝ ρ_mean^{1/2}     ∝ (M / R^3)^{1/2}         (Brown+1991 PDF p. 1)
    ν_max ∝ g · T_eff^{-1/2} ∝ (M/R^2) · T_eff^{-1/2}  (Brown+1991 PDF p. 2)

Eliminating R between the two yields R ∝ √(M T_eff^{-1/2} / ν_max), hence:

    Δν ∝ ν_max^{3/4} · T_eff^{3/8} · M^{-1/4}                          (*)

Normalising to solar values (Δν_⊙ = 134.9 μHz, ν_max,⊙ = 3090 μHz,
T_eff,⊙ = 5777 K) and absorbing the unknown M^{-1/4} factor by adopting
a canonical red-giant mass M ≈ 1.2 M_⊙ (Yu+2018 §3.6 fig. 7 — population
mean):

    Δν = Δν_⊙ · (ν_max / ν_max,⊙)^{3/4}
                · (T_eff / T_eff,⊙)^{3/8}
                · M_RG^{-1/4}

with M_RG = 1.2 (the population mean of the Kepler red-giant sample;
absorbed below into the prefactor for a single dimensionless expression).

This is the THEORETICAL counterpart to the EMPIRICAL Yu+2018 §3.6 fit
(Δν = 0.267 · ν_max^0.764).  The exponents 3/4 and 3/8 are structural
(derived from the canonical Brown+1991 scaling laws), not fit values; the
prefactor combines Δν_⊙ = 134.9 μHz with the solar normalisations of
ν_max and T_eff and the canonical M_RG^{-1/4}.

Structural derivation of the prefactor:
    Δν = 134.9 · (1/3090)^{3/4} · (1/5777)^{3/8} · 1.2^{-1/4} · ν_max^{3/4} · T_eff^{3/8}
       ≈ 134.9 · 0.002408 · 0.04284 · 0.9554 · ν_max^{3/4} · T_eff^{3/8}
       ≈ 0.01330 · ν_max^{3/4} · T_eff^{3/8}

where ν_max is in μHz and T_eff in K.

LAW_CONSTANTS:
    None — this theoretical baseline has no fitted defining coefficient of its
    own. The power-law exponents 3/4 (on ν_max), 3/8 (on T_eff) and the -1/4
    (on M_RG) are STRUCTURAL — derived from the asymptotic scaling laws
    Δν ∝ ρ^{1/2} and ν_max ∝ g·T_eff^{-1/2}, NOT fit values (see "the exponents
    3/4 and 3/8 are structural ..., not fit values" above). Per the four-field
    contract, structural exponents stay INLINE in predict(), not in LAW.

OTHER_CONSTANTS (solar references / population mass — givens, consumed):
    delta_nu_sun = 134.9    μHz    Solar Δν_⊙ (Toutain & Fröhlich 1992, cited
                                    Kjeldsen & Bedding 1995 PDF p. 6)
    nu_max_sun   = 3090.0   μHz    Solar ν_max,⊙ (Yu+2018 §3.1; Huber+2011)
    Teff_sun     = 5777.0   K      Solar T_eff (IAU 2015 nominal)
    M_RG_canonical = 1.2    M_⊙   Population mean mass of Kepler red giants
                                    (Yu+2018 §3.4 / Stello+2013)

Type designation:
    Type I — no per-cluster parameters; the formula uses two released
    observables (ν_max and Teff) with theoretical exponents.

Column mapping:
    nu_max (released CSV col 1) = ν_max [μHz]   — INPUT 0 in X
    Teff   (released CSV col 2) = T_eff [K]     — INPUT 1 in X
    Δν     (released CSV col 0, target)         — predicted [μHz]

Caveats
-------
1. Absorbing M^{-1/4} into the prefactor with M_RG = 1.2 M_⊙ introduces a
   small systematic for individual stars (population mass spread ~ 0.6–2.5 M_⊙;
   the bracketing range of M^{-1/4} factors is 0.88 to 1.06, ~9% scatter).
   This is the dominant theoretical-vs-empirical scatter source.
2. The exponent on ν_max is 3/4 = 0.75 here (theory); the empirical Yu+2018
   fit recovers 0.764 — a small but real deviation from the asymptotic
   theoretical prediction. The 3/4 baseline is the "pure-physics" prediction;
   yu_2018 is the data-calibrated refinement.
3. T_eff and ν_max are both available on the released input set (FM-K guard:
   inputs are not withheld); this is a FULLY EXECUTABLE alternate to yu_2018.
"""
import numpy as np

USED_INPUTS = ["nu_max", "Teff"]
PAPER_REF   = "summary_formula_brown_1991.md"
EQUATION_LOC = ("PDF p. 1-2 — Δν ∝ ρ^{1/2} and ν_ac ∝ g·T_eff^{-1/2}, "
                "combined to give Δν ∝ ν_max^{3/4} T_eff^{3/8} M^{-1/4}")

# === LAW_CONSTANTS — paper-published / theoretically derived, frozen ===
# The two power-law exponents 3/4 (on ν_max) and 3/8 (on T_eff), together with
# the M^{-1/4} factor, are STRUCTURAL exponents of the form — derived from the
# asymptotic scaling laws (Δν ∝ ρ^{1/2}, ν_max ∝ g·T_eff^{-1/2}), NOT fit values
# (see the module docstring: "The exponents 3/4 and 3/8 are structural ...,
# not fit values"). Per the four-field contract, structural exponents stay
# INLINE in predict(), they are not LAW. This baseline therefore has no
# cross-task defining coefficient of its own; LAW is legitimately empty.
LAW_CONSTANTS = {}

# === OTHER_CONSTANTS — solar references and population mass (givens) ===
OTHER_CONSTANTS = {
    "delta_nu_sun":  134.9,   # μHz; Kjeldsen & Bedding 1995 PDF p. 6 (solar Δν_⊙)
    "nu_max_sun":    3090.0,  # μHz; Yu+2018 §3.1 / Huber+2011 solar reference
    "Teff_sun":      5777.0,  # K;   IAU 2015 nominal solar T_eff
    "M_RG_canonical": 1.2,    # M_⊙; Yu+2018 §3.4 RG population mean
}

LOCAL_FITTABLE = {}


def predict(X: np.ndarray) -> np.ndarray:
    """Predict Δν via the canonical Brown+1991 theoretical scaling.

    Parameters
    ----------
    X : np.ndarray, shape (n, 2)
        Column 0 = ν_max [μHz].
        Column 1 = T_eff [K].

    LAW_CONSTANTS is empty (this theoretical baseline has no fitted defining
    coefficient — its exponents are structural). The solar references and
    canonical RG mass are read from OTHER_CONSTANTS; the structural exponents
    3/4, 3/8, and -1/4 are inline.

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted Δν [μHz].

    Formula:
        Δν = Δν_⊙ · (ν_max / ν_max,⊙)^{3/4} · (T_eff / T_eff,⊙)^{3/8} · M_RG^{-1/4}

    With M_RG^{-1/4} ≈ 0.9554, this is the theoretical baseline complementing
    the Yu+2018 empirical fit.
    """
    delta_nu_sun   = OTHER_CONSTANTS["delta_nu_sun"]
    nu_max_sun     = OTHER_CONSTANTS["nu_max_sun"]
    Teff_sun       = OTHER_CONSTANTS["Teff_sun"]
    M_RG_canonical = OTHER_CONSTANTS["M_RG_canonical"]

    nu_max = np.asarray(X[:, 0], dtype=float)
    Teff   = np.asarray(X[:, 1], dtype=float)

    mass_factor = float(M_RG_canonical) ** (-0.25)
    return (
        float(delta_nu_sun)
        * (nu_max / float(nu_max_sun)) ** (3.0 / 4.0)
        * (Teff   / float(Teff_sun))   ** (3.0 / 8.0)
        * mass_factor
    )
