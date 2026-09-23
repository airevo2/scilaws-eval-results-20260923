"""Taylor et al. (2011) GAMA empirical (g-i)-M*/L_i relation (Eqs. 7 & 8, PDF p. 20).

Citation: Taylor et al. (2011), MNRAS 418, 1587-1620. DOI:10.1111/j.1365-2966.2011.19536.x.
          Equations 7 and 8, PDF page 20.

Formula
-------
Equation 7 (PDF p. 20):

    log10(M*/L_i) = -0.68 + 0.70*(g - i)

where L_i is in AB-centric units (log L_i = -0.4*M_i, no solar normalisation).

Equation 8 (PDF p. 20) — the benchmark-ready form:

    log10(M*/M_sun) = -0.68 + 0.70*(g - i) - 0.4*M_i

Derivation: substituting log L_i = -0.4*(M_i - M_i,sun) with M_i,sun = 4.58
(AB; stated parenthetically immediately after Eq. 7, PDF p. 20) into Eq. 7
gives log M* = -0.68 + 0.70*(g-i) - 0.4*(M_i - 4.58) = Eq. 8 form
-0.68 + 0.70*(g-i) - 0.4*M_i + 1.832. This is equivalent to the predict()
body below which uses c*(M_i - M_sun_i) = -0.4*(M_i - 4.58).

LAW_CONSTANTS (Eq. 7, PDF p. 20, Chabrier IMF, BC03 models, GAMA calibration)
-------------------------------------------------------------------------------
  a = -0.68  — empirical intercept of GAMA log(M*/L_i) vs (g-i) fit
  b = +0.70  — empirical slope of the same relation (per AB mag)

OTHER_CONSTANTS
---------------
  c = -0.4      : mathematical magnitude-to-luminosity conversion factor.
  M_sun_i = 4.58 : i-band AB solar absolute magnitude; parenthetical note
                   immediately following Eq. 7, PDF p. 20.

Type: Type I — globally fixed coefficients; no per-galaxy free parameters.
      LOCAL_FITTABLE is empty.

Column mapping (released CSV -> formula symbols):
  g    -> m_g    (SDSS g-band apparent AB magnitude, Galactic-extinction-corrected)
  i    -> m_i    (SDSS i-band apparent AB magnitude)
  DM   -> mu     (distance modulus; M_i = i - DM)

Caveats
-------
Taylor et al. (2011) use restframe (k-corrected) colours. The released benchmark
CSV provides observed-frame apparent magnitudes only (no k-correction column).
At the SDSS redshift range (z up to ~0.3) the k-correction can reach ~0.1-0.2 mag
for g-band and ~0.05 mag for i-band; (g-i) is biased by roughly +0.05 to +0.15 mag.
The SR method is expected to absorb this residual via fit drift on (a, b).
IMF is Chabrier (2003), consistent with the MPA-JHU lgm_tot_p50 target (Kroupa ~
Chabrier to within ~0.03 dex).
"""

import numpy as np

USED_INPUTS = ["g", "i", "DM"]
PAPER_REF   = "summary_formula+dataset_taylor_2011.md"
EQUATION_LOC = "Taylor 2011 Eq. 7 + Eq. 8, PDF p. 20 (M_i,sun = 4.58 from parenthetical after Eq. 7)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a": -0.68,   # empirical intercept of GAMA (g-i)-log(M*/L_i) fit; Eq. 7, PDF p. 20
    "b":  0.70,   # empirical slope; Eq. 7, PDF p. 20
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "c":      -0.4,   # magnitude-to-log-luminosity conversion; structural identity
    "M_sun_i": 4.58,  # i-band AB solar absolute magnitude; parenthetical after Eq. 7, PDF p. 20
}

LOCAL_FITTABLE = {}   # Type I — no per-galaxy free parameters


def predict(X: np.ndarray, a: float, b: float) -> np.ndarray:
    """Predict log10(M*/M_sun) from SDSS g, i apparent magnitudes and distance modulus.

    X columns: [g, i, DM] in USED_INPUTS order.
    params: a, b from LAW_CONSTANTS; c and M_sun_i read from OTHER_CONSTANTS.
    """
    c       = OTHER_CONSTANTS["c"]
    M_sun_i = OTHER_CONSTANTS["M_sun_i"]

    g  = np.asarray(X[:, 0], dtype=float)
    i  = np.asarray(X[:, 1], dtype=float)
    dm = np.asarray(X[:, 2], dtype=float)

    colour = g - i      # observed-frame g-i proxy for rest-frame g-i
    M_i    = i - dm     # absolute i-band magnitude
    return a + b * colour + c * (M_i - M_sun_i)
