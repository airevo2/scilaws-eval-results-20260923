"""Bell et al. (2003) colour-(M*/L) calibration — i-band, g-r colour (Table A7, PDF p. 22).

Citation: Bell et al. (2003), ApJS 149, 289-312. DOI:10.1086/378847.
          Table A7 ("Stellar M/L Ratio as a Function of Color"), PDF page 22.

Formula
-------
The colour-M/L calibration (Table A7, PDF p. 22, note at bottom):

    log10(M/L_lambda) = a_lambda + b_lambda * Colour

Combined with the magnitude-to-luminosity identity
log10(L_lambda/L_sun,lambda) = -0.4*(M_lambda - M_sun,lambda), the
full photometric stellar-mass equation becomes:

    log10(M*/M_sun) = -0.4*(M_i - M_sun,i) + a_i + b_i*(g - r)

where M_i = m_i - DM is the absolute i-band magnitude.

LAW_CONSTANTS (Table A7, PDF p. 22, diet-Salpeter IMF, g-r colour, i-band)
---------------------------------------------------------------------------
  a_i = -0.222   — intercept for i-band / g-r; diet-Salpeter IMF
  b_i = +0.864   — slope for i-band / g-r

OTHER_CONSTANTS
---------------
  c = -0.4      : mathematical magnitude-to-luminosity conversion factor;
                  structural identity, not a calibrated constant.
  M_sun_i = 4.58 : i-band AB solar absolute magnitude. Bell et al. 2003 used
                   M_sun,i = 4.56 (PEGASE, PDF p. 6) in their own calibration;
                   this benchmark uses the AB-system value 4.58 quoted in
                   Taylor et al. 2011 PDF p. 20. The 0.008 dex difference is
                   far below all other systematics here (intentional unification
                   across baselines, per data_spec §9.6).

Type: Type I — globally fixed coefficients; no per-galaxy free parameters.
      LOCAL_FITTABLE is empty.

Column mapping (released CSV -> formula symbols):
  g           -> m_g   (SDSS g-band apparent AB magnitude, Galactic-extinction-corrected)
  r           -> m_r   (SDSS r-band apparent AB magnitude)
  i           -> m_i   (SDSS i-band apparent AB magnitude)
  DM          -> mu    (distance modulus; M_i = i - DM)
  Colour      -> g - r (observed-frame SDSS g-r proxy for rest-frame g-r)

Caveats
-------
Bell et al. (2003) calibrated on SDSS EDR galaxies at z < 0.1 under a 'diet'
Salpeter IMF. The benchmark target (lgm_tot_p50 from MPA-JHU Kauffmann+2003
pipeline) uses a Kroupa IMF. Diet-Salpeter masses are systematically higher by
approximately 0.15 dex. This systematic offset is a known, documented IMF
calibration difference (PDF p. 22 note: "subtract 0.15 dex for Kennicutt or
Kroupa IMF"). It is intentional calibration-crudeness: the formula is shipped
paper-faithful; the ~0.15 dex IMF bias will appear as a non-zero mean residual.
The formula remains useful as a reference anchor for the benchmark.

No k-corrections are applied (the released CSV provides only observed-frame
magnitudes). At z ~ 0.3 k-corrections are non-negligible; residuals from
k-correction omission are expected to be absorbed by fit drift in the SR method.
"""

import numpy as np

USED_INPUTS = ["g", "r", "i", "DM"]
PAPER_REF   = "summary_formula+dataset_bell_2003.md"
EQUATION_LOC = "Bell 2003 Table A7, PDF p. 22 (g-r colour, i-band column; diet-Salpeter IMF)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a_i": -0.222,   # i-band intercept, g-r colour, diet-Salpeter IMF; Table A7 PDF p. 22
    "b_i":  0.864,   # i-band slope, g-r colour, diet-Salpeter IMF; Table A7 PDF p. 22
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "c":      -0.4,   # magnitude-to-log-luminosity conversion: log(L/L_sun) = -0.4*(M - M_sun)
    "M_sun_i": 4.58,  # i-band AB solar absolute magnitude; Taylor et al. 2011,
                      # parenthetical after Eq. 7, PDF p. 20
}

LOCAL_FITTABLE = {}   # Type I — no per-galaxy free parameters


def predict(X: np.ndarray, a_i: float, b_i: float) -> np.ndarray:
    """Predict log10(M*/M_sun) from SDSS g, r, i apparent magnitudes and distance modulus.

    X columns: [g, r, i, DM] in USED_INPUTS order.
    params: a_i, b_i from LAW_CONSTANTS; c and M_sun_i read from OTHER_CONSTANTS.
    """
    c       = OTHER_CONSTANTS["c"]
    M_sun_i = OTHER_CONSTANTS["M_sun_i"]

    g  = np.asarray(X[:, 0], dtype=float)
    r  = np.asarray(X[:, 1], dtype=float)
    i  = np.asarray(X[:, 2], dtype=float)
    dm = np.asarray(X[:, 3], dtype=float)

    colour = g - r          # observed-frame g-r proxy for rest-frame g-r
    M_i    = i - dm         # absolute i-band magnitude
    return c * (M_i - M_sun_i) + a_i + b_i * colour
