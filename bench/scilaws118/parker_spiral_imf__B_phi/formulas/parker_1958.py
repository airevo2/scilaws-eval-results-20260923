"""Parker (1958) azimuthal IMF baseline — the Archimedean spiral relation.

Parker, E. N. (1958). "Dynamics of the Interplanetary Gas and Magnetic
Fields." Astrophysical Journal, 128, 664-676. DOI: 10.1086/146579.
Equation (26), PDF page 10:

    B_r(r, theta, phi) = B(b, phi0) * (b/r)^2
    B_theta = 0
    B_phi(r, theta, phi) = B(b, phi0) * (b/r) * (r - b) * (omega * sin(theta) / v_m)

For r >> b (the heliosphere well beyond the source surface) and the ecliptic
plane (theta = pi/2, sin(theta) = 1), this reduces to:

    B_phi / B_r ≈ -omega * r / v_sw

or equivalently:

    B_phi = -B_r * Omega_sun * r / v_sw

where the sign reflects the trailing sense of the Parker spiral.

Parker (1958) states on PDF page 11: "The angular velocity of the sun is
approximately omega ~ 2.7 × 10^-6." This is an approximation, not a precise
value. The precise value 2.866e-6 rad/s is computed from the Carrington
sidereal rotation period (≈ 25.38 days), which is an IAU / solar-physics
standard NOT stated on any page of the Parker (1958) or Owens & Forsyth
(2013) PDFs. Per data_spec §0.2.4, a constant computed from an external
standard is not "in the paper" and must be declared OTHER_CONSTANTS with the
external source named, not LAW_CONSTANTS. Accordingly LAW_CONSTANTS = {}.

LAW_CONSTANTS (paper's scientific claim — scored by constant-judge)
-------------------------------------------------------------------
(empty) — the scientific claim of Parker (1958) is the functional form
    B_phi = -B_r * omega * r / v_sw, not a specific numeric value for
    omega. The paper only states an approximate "omega ~ 2.7 × 10^-6"
    (PDF p. 11). The precise value used below is an external standard.

OTHER_CONSTANTS (universal physics / unit factors / external standards — not scored)
-----------------------------------------------------------------------------------
Omega_sun = 2.866e-6 rad/s  — Carrington sidereal solar rotation rate at
    the solar equator. External source: IAU / Carrington standard sidereal
    rotation period ≈ 25.38 days ⇒ Ω = 2π / (25.38 × 86400 s) =
    2.8657e-6 rad/s ≈ 2.866e-6 rad/s. This value is NOT stated on any page
    of the Parker (1958) or Owens & Forsyth (2013) PDFs; it is computed from
    the Carrington period, an external solar-physics standard.
AU_m = 1.495978707e+11 m  — 1 astronomical unit, IAU 2012 Resolution B2
    (exact, by definition). Unit conversion factor r [AU] -> r [m].
km_to_m = 1000.0           — Unit conversion v_sw [km/s] -> [m/s].

Type designation: TYPE I.
  - All rows are independent hourly-averaged daily measurements at L1.
  - No cluster structure; no per-cluster fitted parameters.
  - LOCAL_FITTABLE = {} (empty).

Column mapping (released CSV → formula variables):
  col 0  B_phi_nT  — target; not an input to predict()
  col 1  B_r_nT    — B_r [nT] (radial IMF, +outward from Sun)
  col 2  v_sw_km_s — v_sw [km/s] (solar wind bulk speed)
  col 3  r_AU      — r [AU] (heliocentric distance; 1.0 for OMNI/L1)

Caveats:
  (a) At daily cadence on OMNI 2 L1 data the bare Parker relation achieves
      R² ~ 0.30-0.40; the dominant residual is Alfvenic turbulence and
      transient events (CMEs, SIRs), not a coefficient error.
  (b) The sign of B_phi is negative for a trailing ("Parker-spiral") sector
      where B_r > 0 (outward polarity) — verified by the mean(B_r * B_phi) < 0
      test on the OMNI dataset.
  (c) r_AU = 1.0 for all L1 rows; the column is retained so the formula
      generalises to multi-r data (Helios, Parker Solar Probe).
"""

import numpy as np

USED_INPUTS = ["B_r_nT", "v_sw_km_s", "r_AU"]
PAPER_REF   = "summary_formula_parker_1958.md"
EQUATION_LOC = "Eq. 26, PDF p. 10; Omega_sun ~ 2.7e-6 stated PDF p. 11"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {}   # functional form is the claim; omega value is an external standard.

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    # Carrington sidereal solar rotation rate. External source: IAU / Carrington
    # standard sidereal period ≈ 25.38 days → Ω = 2π/(25.38×86400) = 2.866e-6 rad/s.
    # NOT stated on any page of the Parker (1958) or Owens & Forsyth (2013) PDFs.
    "Omega_sun": 2.866e-6,   # rad/s  (IAU/Carrington external standard)
    # IAU 2012 Resolution B2 (exact by definition).
    "AU_m":    1.495978707e11,  # m per astronomical unit
    "km_to_m": 1000.0,         # m per km (unit conversion)
}

LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters; no fit().


def predict(X: np.ndarray) -> np.ndarray:
    """Predicted azimuthal IMF B_phi [nT] from the Parker (1958) spiral.

    Parameters
    ----------
    X : np.ndarray, shape (n, 3)
        Column 0 = B_r_nT    [nT]    radial IMF (+outward)
        Column 1 = v_sw_km_s [km/s]  solar wind bulk speed
        Column 2 = r_AU      [AU]    heliocentric distance

    Omega_sun, AU_m, km_to_m are read from OTHER_CONSTANTS (module namespace),
    not passed as kwargs (data_spec §5.1). LAW_CONSTANTS = {} — the form is
    the paper's scientific claim; the numeric Omega_sun is an external standard.

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted B_phi [nT].
    """
    Omega_sun = OTHER_CONSTANTS["Omega_sun"]
    AU_m      = OTHER_CONSTANTS["AU_m"]
    km_to_m   = OTHER_CONSTANTS["km_to_m"]

    X = np.asarray(X, dtype=float)
    B_r      = X[:, 0]           # nT
    v_sw_ms  = X[:, 1] * km_to_m  # km/s -> m/s
    r_m      = X[:, 2] * AU_m     # AU   -> m

    # Parker (1958) Eq. 26, ecliptic / large-r limit:
    #   B_phi = -B_r * omega * r / v_sw
    return -B_r * Omega_sun * r_m / v_sw_ms
