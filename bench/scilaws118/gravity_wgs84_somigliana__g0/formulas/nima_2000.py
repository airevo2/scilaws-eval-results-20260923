"""NIMA Technical Report TR8350.2 (2000) — WGS84 Somigliana Normal-Gravity Formula.

National Imagery and Mapping Agency, DoD World Geodetic System 1984,
TR8350.2, Third Edition Amendment 1, January 2000. Equation (4-1),
Section 4.2 "Normal Gravity on the Ellipsoidal Surface", PDF page 42.

Formula (Eq. 4-1, PDF p. 42):
    gamma = gamma_e * (1 + k * sin^2(phi)) / sqrt(1 - e2 * sin^2(phi))

where phi is geodetic latitude in radians.

LAW_CONSTANTS — none. The Somigliana formula declares no FITTED defining
-----------------------------------------------------------------------
coefficient. WGS84's genuine adopted/estimated quantities are its four
DEFINING parameters — semi-major axis a, inverse flattening 1/f, angular
velocity omega, gravitational constant GM (Table 3.1 "WGS 84 Defining
Parameters", PDF p. 35; a "based on estimates from the 1976-1979 time
period determined using laser, Doppler and radar altimeter data") — and
NONE of those appears in the closed Somigliana form. All three constants
the formula consumes (gamma_e, k, e2) are DERIVED from those defining
parameters by closed geodetic identities, so each is a given (OTHER),
not a fitted defining coefficient (LAW). LAW_CONSTANTS is therefore {}.

OTHER_CONSTANTS — derived WGS84 givens the formula consumes (kind b)
-------------------------------------------------------------------
    gamma_e = 9.7803253359 m/s^2
              Theoretical (Normal) Gravity at the Equator (on the Ellipsoid).
              Table 3.4 "Derived Physical Constants", PDF p. 40.
              A DERIVED physical constant (computed from a, f, GM, omega by
              Somigliana gravity theory), not a fitted coefficient.

    k       = 0.00193185265241
              Theoretical (Normal) Gravity Formula Constant.
              Table 3.4 "Derived Physical Constants", PDF p. 40.
              Explicitly DERIVED: k = b*gamma_p/(a*gamma_e) - 1 (the relation
              is written verbatim under Eq. 4-1, PDF p. 42).

    e2      = 0.00669437999014
              Square of the First Ellipsoidal Eccentricity (e^2).
              Table 3.3 "WGS 84 Ellipsoid Derived Geometric Constants", PDF p. 40.
              A DERIVED geometric constant: e^2 = 2f - f^2 from the defining
              1/f = 298.257223563 reproduces 6.69437999014e-3 to < 1e-12.

pi is a structural mathematical constant (inline), not declared.

Type designation: Type I — one global formula applies to every station.
The three WGS84 constants are invariant across all latitudes and stations,
but invariance alone does not make them LAW: each is a derived/standard
given (OTHER), not a paper-fitted defining coefficient. LOCAL_FITTABLE is
empty; no fit() function.

Column mapping:
    phi (geodetic latitude) — released CSV column `latitude` (degrees; converted
                              to radians inside predict).
    USED_INPUTS = ["latitude"]. The released CSV holds only (g0, latitude); the
    free-air gradient has already been folded into the g0 derivation upstream in
    prep_data.py, so elevation is intentionally absent from the released schema.

Caveats:
    The data target g0 is derived from observed gravity plus a free-air
    correction (0.3086 mGal/m * elevation), not pure Somigliana output. For
    low-elevation stations (H < 200 m) the residual is geological scatter
    (~5 mGal std). For high-elevation stations the Bouguer residual can reach
    ~2100 mGal, a known geophysical effect documented in data_raw/PROVENANCE.md.
    The Somigliana formula captures >99.95% of the total g0 signal variance.
    OOD split (data_spec §3.5, Tier 3): train = lat < 60°N, test = lat >= 60°N.
"""

import numpy as np

USED_INPUTS = ["latitude"]
PAPER_REF   = "summary_formula_nima_2000.md"
EQUATION_LOC = "NIMA TR8350.2 Eq. (4-1), Section 4.2, PDF p. 42; constants from Tables 3.3–3.4, PDF p. 40"

# LAW_CONSTANTS — none. The Somigliana form has no fitted defining coefficient;
# WGS84's fitted/adopted quantities are its four DEFINING parameters (a, 1/f,
# GM, omega; Table 3.1), none of which appears in Eq. (4-1).
LAW_CONSTANTS = {}

# OTHER_CONSTANTS — derived WGS84 givens consumed by Eq. (4-1) (kind b: derived
# from the defining parameters), all tabulated on NIMA TR8350.2 PDF page 40.
OTHER_CONSTANTS = {
    "gamma_e": 9.7803253359,      # m/s^2 — equatorial normal gravity (derived), Table 3.4
    "k":       0.00193185265241,  # dimensionless — gravity formula constant (derived k=b*gamma_p/(a*gamma_e)-1), Table 3.4
    "e2":      0.00669437999014,  # dimensionless — first eccentricity squared (derived e^2=2f-f^2), Table 3.3
}

LOCAL_FITTABLE = {}    # Type I — no per-cluster parameters


def predict(X: np.ndarray) -> np.ndarray:
    """Somigliana normal gravity at each station's geodetic latitude.

    X : (n, 1) — column 0 is geodetic latitude in degrees.
    Returns: (n,) array of normal gravity in m/s^2.
    """
    gamma_e = OTHER_CONSTANTS["gamma_e"]
    k       = OTHER_CONSTANTS["k"]
    e2      = OTHER_CONSTANTS["e2"]
    lat_deg = X[:, 0].astype(float)
    phi = np.deg2rad(lat_deg)
    sin2 = np.sin(phi) ** 2
    return gamma_e * (1.0 + k * sin2) / np.sqrt(1.0 - e2 * sin2)
