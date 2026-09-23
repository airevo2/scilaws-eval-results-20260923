"""Parker (1958) azimuthal IMF — paper-stated approximate omega value.

Parker, E. N. (1958). "Dynamics of the Interplanetary Gas and Magnetic
Fields." Astrophysical Journal, 128, 664-676. DOI: 10.1086/146579.

Same functional form as parker_1958.py (Eq. 26, PDF p. 10):

    B_phi = -B_r * omega * r / v_sw

but here the solar rotation rate Omega_sun is taken directly from Parker's
own approximate statement on PDF page 11:
    "The angular velocity of the sun is approximately omega ~ 2.7 × 10^-6."

This value (2.7e-6 rad/s) is explicitly stated in the Parker (1958) paper
itself (PDF p. 11), in contrast to the precise IAU/Carrington sidereal
value (2.866e-6 rad/s) used in parker_1958.py which is an external standard.
The "~" acknowledges it is an approximation. This baseline implements the
formula exactly as it would be applied using only values given in the
Parker (1958) paper, without resort to external standards.

Compared to parker_1958.py:
  - parker_1958.py uses Omega_sun = 2.866e-6 (IAU/Carrington external standard)
  - This module uses Omega_sun = 2.7e-6  (Parker's own stated value)
Both file Omega in OTHER_CONSTANTS — it is a standard physical constant (the
Sun's angular rotation rate) that the kinematic Parker spiral merely consumes
as a given, not a defining coefficient fit to the IMF data. The two baselines
differ only in which value of that physical given they plug in.

LAW_CONSTANTS (paper's scientific claim — scored by constant-judge)
-------------------------------------------------------------------
(empty) — the scientific claim of Parker (1958) is the functional form
    B_phi = -B_r * omega * r / v_sw, not a specific numeric value for omega.
    Omega is the Sun's angular velocity, a known physical property the spiral
    consumes as a given (Parker only *quotes* an approximate value to compute
    his 2.5 AU example, PDF p. 11), so it is OTHER, not a fitted coefficient.

OTHER_CONSTANTS (universal physics / unit factors / paper-stated givens)
------------------------------------------------------------------------
Omega_sun_stated = 2.7e-6 rad/s  — the Sun's angular rotation rate, taken at
    Parker's own stated value. Parker (1958) PDF p. 11: "The angular velocity
    of the sun is approximately omega ~ 2.7 × 10^-6." A standard physical
    constant the kinematic spiral plugs in (not a defining coefficient).
AU_m    = 1.495978707e11 m  — IAU 2012 Resolution B2, unit conversion r [AU] -> [m].
km_to_m = 1000.0           — unit conversion v_sw [km/s] -> [m/s].

Type designation: TYPE I.
  - All rows are independent daily measurements at L1.
  - No cluster structure; no per-cluster fitted parameters.
  - LOCAL_FITTABLE = {} (empty).

Column mapping (released CSV -> formula variables):
  col 0  B_phi_nT  — target
  col 1  B_r_nT    — B_r [nT]
  col 2  v_sw_km_s — v_sw [km/s]
  col 3  r_AU      — r [AU] (= 1.0 for all L1 rows)
"""

import numpy as np

USED_INPUTS = ["B_r_nT", "v_sw_km_s", "r_AU"]
PAPER_REF   = "summary_formula_parker_1958.md"
EQUATION_LOC = "Eq. 26, PDF p. 10 (functional form); Omega_sun ~ 2.7e-6 stated PDF p. 11"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {}   # functional form is the claim; omega is a standard physical given.

# === OTHER_CONSTANTS — universal physics factors / paper-stated givens ===
OTHER_CONSTANTS = {
    # Parker (1958) PDF p. 11: "approximately omega ~ 2.7 × 10^-6". The Sun's
    # angular rotation rate (a standard physical constant the spiral consumes),
    # at Parker's own stated approximate value.
    "Omega_sun_stated": 2.7e-6,   # rad/s — paper-stated value of the solar rotation rate
    "AU_m":    1.495978707e11,  # m per AU, IAU 2012 Resolution B2
    "km_to_m": 1000.0,         # m per km
}

LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray) -> np.ndarray:
    """Parker (1958) B_phi using the paper's own stated omega ~ 2.7e-6.

    X : (n, 3)
        Column 0 = B_r_nT    [nT]
        Column 1 = v_sw_km_s [km/s]
        Column 2 = r_AU      [AU]

    Omega_sun_stated, AU_m and km_to_m are all read from OTHER_CONSTANTS
    (module namespace); LAW_CONSTANTS = {} so the harness passes no kwargs
    (gold style: predict(X)).

    Returns: (n,) predicted B_phi [nT].
    """
    Omega_sun_stated = OTHER_CONSTANTS["Omega_sun_stated"]
    AU_m             = OTHER_CONSTANTS["AU_m"]
    km_to_m          = OTHER_CONSTANTS["km_to_m"]

    X = np.asarray(X, dtype=float)
    B_r      = X[:, 0]
    v_sw_ms  = X[:, 1] * km_to_m
    r_m      = X[:, 2] * AU_m

    return -B_r * Omega_sun_stated * r_m / v_sw_ms
