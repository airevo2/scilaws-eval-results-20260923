"""Zibetti, Charlot & Rix (2009) powerlaw M/L calibration — i-band, g-i colour (Table B1, PDF p. 21).

Citation: Zibetti, Charlot & Rix (2009), MNRAS 400, 1181-1198.
          Appendix B, Eq. B1 and Table B1, PDF page 21.

Formula
-------
Equation B1 (Appendix B, PDF p. 21):

    log10(Upsilon_lambda) = a_lambda + b_lambda * colour

where Upsilon_lambda = M*/L_lambda is the stellar mass-to-light ratio.

Combined with the magnitude-to-luminosity identity, the photometric
stellar-mass equation becomes:

    log10(M*/M_sun) = -0.4*(M_i - M_sun,i) + a_i + b_i*(g - i)

LAW_CONSTANTS (Table B1, PDF p. 21, g-i colour, i-band, Chabrier IMF, CB07 models)
-----------------------------------------------------------------------------------
  a_i = -0.963   — i-band intercept, g-i colour, Chabrier IMF; Table B1 PDF p. 21
  b_i = +1.032   — i-band slope, g-i colour; Table B1 PDF p. 21

OTHER_CONSTANTS
---------------
  c = -0.4      : mathematical magnitude-to-luminosity conversion factor.
  M_sun_i = 4.58 : i-band AB solar absolute magnitude (Taylor et al. 2011,
                   parenthetical after Eq. 7, PDF p. 20).

Type: Type I — globally fixed coefficients; no per-galaxy free parameters.
      LOCAL_FITTABLE is empty.

Column mapping (released CSV -> formula symbols):
  g    -> m_g    (SDSS g-band apparent AB magnitude, Galactic-extinction-corrected)
  i    -> m_i    (SDSS i-band apparent AB magnitude)
  DM   -> mu     (distance modulus; M_i = i - DM)

Caveats
-------
Zibetti et al. (2009) derive M/L ratios from a Monte Carlo library of 50,000
CB07-based SPS models (Charlot & Bruzual 2007, private communication), with
Chabrier (2003) IMF. The slopes in Table B1 are systematically steeper than
those of Bell et al. (2003) (BC03-based) because CB07 models have different
TP-AGB prescriptions. This formula and bell_2003 are expected to disagree by
~0.1 dex even in the absence of measurement scatter. The Chabrier IMF is
consistent with the MPA-JHU lgm_tot_p50 target (Kroupa ~ Chabrier to < 0.03 dex).
No k-corrections are applied (observed-frame proxy for rest-frame g-i).
"""

import numpy as np

USED_INPUTS = ["g", "i", "DM"]
PAPER_REF   = "summary_formula_zibetti_2009.md"
EQUATION_LOC = "Zibetti 2009 Eq. B1 + Table B1, PDF p. 21 (g-i colour, i-band, Chabrier IMF)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a_i": -0.963,   # i-band intercept, g-i colour, Chabrier IMF; Table B1 PDF p. 21
    "b_i":  1.032,   # i-band slope, g-i colour; Table B1 PDF p. 21
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "c":      -0.4,   # magnitude-to-log-luminosity conversion; structural identity
    "M_sun_i": 4.58,  # i-band AB solar absolute magnitude; Fukugita 1996,
                      # same value as Bell 2003 Table A7 footnote (PDF p. 22)
}

LOCAL_FITTABLE = {}   # Type I — no per-galaxy free parameters


def predict(X: np.ndarray, a_i: float, b_i: float) -> np.ndarray:
    """Predict log10(M*/M_sun) from SDSS g, i apparent magnitudes and distance modulus.

    X columns: [g, i, DM] in USED_INPUTS order.
    params: a_i, b_i from LAW_CONSTANTS; c and M_sun_i read from OTHER_CONSTANTS.
    """
    c       = OTHER_CONSTANTS["c"]
    M_sun_i = OTHER_CONSTANTS["M_sun_i"]

    g  = np.asarray(X[:, 0], dtype=float)
    i  = np.asarray(X[:, 1], dtype=float)
    dm = np.asarray(X[:, 2], dtype=float)

    colour = g - i      # observed-frame g-i proxy for rest-frame g-i
    M_i    = i - dm     # absolute i-band magnitude
    return c * (M_i - M_sun_i) + a_i + b_i * colour
