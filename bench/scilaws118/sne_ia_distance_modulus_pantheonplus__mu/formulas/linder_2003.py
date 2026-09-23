"""Flat-universe CPL distance modulus — Chevallier-Polarski-Linder parametrization.

Citation: Linder, E.V. (2003). "Exploring the Expansion History of the Universe."
Physical Review Letters 90, 091301. arXiv:astro-ph/0208512. (PDF p. 1–2)

The Chevallier-Polarski-Linder (CPL) parametrization for dark-energy EOS:

    w(a) = w0 + wa * (1 - a)                                      [Eq. 6-7, PDF p. 2]
         = w0 + wa * z / (1 + z)

The Hubble parameter in a flat universe (Eq. 5, PDF p. 1):

    H(z) = H0 * sqrt( Om*(1+z)^3 + (1-Om)*X_DE(z) )

where the dark-energy density factor is:

    X_DE(z) = (1+z)^(3*(1+w0+wa)) * exp(-3*wa*z/(1+z))

(from integrating the CPL EOS through the continuity equation)

The comoving distance and luminosity distance:

    eta(z)  = (c/H0) * integral_0^z dz'/sqrt(Om*(1+z')^3 + (1-Om)*X_DE(z'))
    dL(z)   = (1+z) * eta(z)                          [flat-universe relation]
    mu(z)   = 5 * log10(dL [Mpc]) + 25                [distance modulus]

Flat ΛCDM special case: w0 = -1, wa = 0 → X_DE = 1 → E^2 = Om*(1+z)^3 + (1-Om).

LAW_CONSTANTS — paper-reported cosmological parameters, frozen
--------------------------------------------------------------
H0  = 73.04 km/s/Mpc : Riess et al. 2022 baseline result (PDF p. 1, abstract;
      confirmed on PDF p. 10 Table 4, line "H0 = 73.04 ± 1.01 km/s/Mpc").
      arXiv:2112.04510, ApJ 934:7.
Om  = 0.334 : Brout et al. 2022, FlatΛCDM, Pantheon+ SNe Ia only
      (PDF p. 15, text "we find ΩM = 0.334 ± 0.018"; Table 3, PDF p. 17).
      arXiv:2202.04077, ApJ 938:110.
w0  = -1.0 : ΛCDM cosmological-constant value; Linder 2003 §II.B, PDF p. 2.
wa  = 0.0  : ΛCDM cosmological-constant value; Linder 2003 §II.B, PDF p. 2.

OTHER_CONSTANTS — physical and structural constants
---------------------------------------------------
c_km_s = 299792.458 km/s : speed of light (exact SI). Appears as c in Eq. 1, Linder 2003.

Type designation: Type I — each supernova is an independent row. The CPL formula
maps one scalar input (redshift z) to one scalar output (distance modulus mu).
LOCAL_FITTABLE = {} (no per-cluster parameters).

Column mapping (paper notation → released CSV):
  z  → z_hd   (Hubble-diagram CMB-frame redshift, col 1)
  mu → mu      (standardised distance modulus [mag], col 0 = target)

Caveats:
- Gauss-Legendre quadrature (200 nodes) used for the comoving distance integral;
  numpy-only, no scipy dependency.
- At very low z (< 0.001), the formula underestimates mu because peculiar velocities
  are unaccounted for by pure Hubble-flow cosmology — but there are no SNe below
  z=0.001 in this sample.
- The formula's predictive RMSE on Pantheon+ is ~0.17 mag (train) / ~0.15 mag (test),
  reflecting real SN intrinsic scatter and host-environment corrections not captured
  by the purely cosmological model.
"""

import numpy as np

USED_INPUTS = ["z_hd"]
PAPER_REF   = "summary_formula_linder_2003.md"
EQUATION_LOC = "Eqs. 5-7, PDF pp. 1-2 (Linder 2003, arXiv:astro-ph/0208512)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "H0": 73.04,   # Riess et al. 2022, PDF p. 1 (abstract) + Table 4 p. 10
    "Om": 0.334,   # Brout et al. 2022, PDF p. 15 (FlatΛCDM, Pantheon+ only)
    "w0": -1.0,    # ΛCDM default; Linder 2003 §II.B, PDF p. 2
    "wa":  0.0,    # ΛCDM default; Linder 2003 §II.B, PDF p. 2
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "c_km_s": 299792.458,  # speed of light [km/s], exact SI; used as c in Eq. 5 Linder 2003
}

LOCAL_FITTABLE = {}  # Type I — no per-cluster parameters

# Pre-compute Gauss-Legendre nodes for the comoving-distance integral (200 nodes)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(200)


def _dL_Mpc(z: np.ndarray, H0: float, Om: float, w0: float, wa: float) -> np.ndarray:
    """Luminosity distance in Mpc, vectorised over z (1-D array)."""
    c_km_s = OTHER_CONSTANTS["c_km_s"]
    z_half  = z[:, None] / 2.0                          # (n, 1)
    z_nodes = z_half * (_GL_X[None, :] + 1.0)           # (n, n_gl)  nodes in [0, z_i]
    weights = z_half * _GL_W[None, :]                   # (n, n_gl)

    zp1 = 1.0 + z_nodes
    # CPL dark-energy density factor: rho_DE/rho_DE0 = (1+z)^(3(1+w0+wa)) exp(-3wa*z/(1+z))
    X_DE = zp1 ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z_nodes / zp1)
    E2   = Om * zp1 ** 3 + (1.0 - Om) * X_DE
    E    = np.sqrt(np.maximum(E2, 1e-30))

    # Comoving distance [Mpc]: eta = (c/H0) * integral_0^z dz'/E(z')
    eta_Mpc = np.sum(weights / E, axis=1) * (c_km_s / H0)
    return (1.0 + z) * eta_Mpc  # flat-universe luminosity distance


def predict(X: np.ndarray, H0: float, Om: float,
            w0: float, wa: float) -> np.ndarray:
    """Return distance modulus mu [mag] for each supernova.

    Gold-style: LAW constants arrive as named params via predict(X, **LAW_CONSTANTS)
    (no default values); OTHER constants are read from the OTHER_CONSTANTS dict;
    structural literals (5, 25, the CPL exponents) stay inline.

    Parameters
    ----------
    X : np.ndarray, shape (n, 1)
        Column order per USED_INPUTS: [z_hd] — Hubble-diagram redshift.
    H0 : float
        Hubble constant [km/s/Mpc]. LAW constant (Riess 2022).
    Om : float
        Matter density parameter Omega_m. LAW constant (Brout 2022).
    w0 : float
        CPL dark-energy EOS today. LAW constant (-1 = ΛCDM).
    wa : float
        CPL dark-energy EOS evolution. LAW constant (0 = ΛCDM).

    Returns
    -------
    np.ndarray, shape (n,)
        Distance modulus mu = 5*log10(dL[Mpc]) + 25.
    """
    z     = np.asarray(X[:, 0], dtype=float)
    dL    = _dL_Mpc(z, H0=H0, Om=Om, w0=w0, wa=wa)
    return 5.0 * np.log10(np.maximum(dL, 1e-30)) + 25.0
