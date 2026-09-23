"""Alerstam et al. (2007) lift-equilibrium equivalent airspeed formula.

Alerstam T, Rosén M, Bäckman J, Ericson PGP, Hellgren O (2007)
"Flight speeds among bird species: Allometric and phylogenetic effects."
PLoS Biology 5(8): e197. doi:10.1371/journal.pbio.0050197

Equation (1), PDF page 1, Introduction section (second paragraph):

    L = (1/2) * rho * C_L * S * U^2

At steady horizontal cruising flight, lift equals body weight (L = m*g).
Using rho = rho_0 (sea-level ISA air density) to obtain the equivalent
airspeed Ue (airspeed corrected to sea-level air density):

    m * g = (1/2) * rho_0 * C_L * S * Ue^2

Solving for Ue:

    Ue = sqrt(2 * m * g / (rho_0 * S * C_L))

This formula can also be written in terms of wing loading Q = m*g/S:

    Ue = sqrt(2 * Q / (rho_0 * C_L))

Paper citation of the form (Introduction, page 1, immediately after Eq. 1):
"...it follows that cruising flight speed among bird species is expected to
scale with body mass and wing loading (Q = m*g/S) as U ~ m^{1/6} and
U ~ Q^{1/2}."

LAW_CONSTANTS
-------------
(none) — The paper does not tabulate a C_L value. The dynamical-similarity
assumption (C_L approximately equal across species) is the scientific claim,
but the numeric value 0.4195 is dataset-derived (see OTHER_CONSTANTS). Per
data_spec §0.2.4 a constant computed from data and NOT stated in the paper
belongs in OTHER_CONSTANTS, not LAW_CONSTANTS.

OTHER_CONSTANTS
---------------
- C_L = 0.4195 (dimensionless) — dataset-derived global mean of the lift
  coefficient across the released species, from Alerstam 2007 Protocol S1.
  Computed as mean(2*m*g/(rho_0*S*Ue^2)) over the 131 species with complete
  mass and wing-area measurements. The paper does not tabulate this value.
  Per data_spec §0.2.4 this is OTHER_CONSTANTS, not LAW_CONSTANTS.
- g = 9.81 m/s^2 — standard gravitational acceleration (CODATA / ISO 80000).
  Universal physical constant, not a scientific discovery target.
- rho_0 = 1.225 kg/m^3 — ISA sea-level air density (ICAO standard atmosphere,
  ISO 2533:1975). Used to convert true airspeed at altitude to equivalent
  airspeed at sea level. Not a discovery target.

Type designation: Type I — the formula applies universally to every species
with a single global C_L. No per-species parameter refit. criteria (a), (b),
(c) of data_spec §1.1 all fail: no per-cluster fitted parameters; each species
is an independent observation under the same physical law; no reference formula
declares per-cluster recovery.

Column mapping (paper notation -> released CSV columns):
  m  (body mass)     -> mass_kg     [kg]
  S  (wing area)     -> wing_area_m2 [m^2]
  Ue (equiv. speed)  -> Ue_ms       [m/s]  (target, column 0)

Caveats:
- The paper's main finding is that the empirical scaling exponent of Ue vs Q
  is 0.31, significantly less than the theoretical 0.5 from this formula.
  Therefore the formula systematically over-predicts the speed range. Negative
  R2 on the test set is expected and documents the paper's scientific claim
  (departure from aerodynamic scaling). Documented in metadata.yaml note block
  per data_spec §9.18.
- C_L = 0.4195 is the global mean from the Protocol S1 dataset; the paper does
  not report a single fitted C_L value.
"""

import numpy as np

USED_INPUTS = ["mass_kg", "wing_area_m2"]
PAPER_REF = "summary_formula+dataset_alerstam_2007.md"
EQUATION_LOC = "Eq. (1), PDF page 1, Introduction section; see also Introduction text p.1 for equilibrium derivation"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {}   # no paper-tabulated constants; see OTHER_CONSTANTS for C_L

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "C_L":   0.4195,  # dimensionless — dataset-derived global mean of the lift coefficient
                      # across the released species, from Alerstam 2007 Protocol S1.
                      # Not tabulated in the paper; per data_spec §0.2.4 → OTHER_CONSTANTS.
    "g":     9.81,    # m/s^2 — standard gravitational acceleration (CODATA)
    "rho_0": 1.225,   # kg/m^3 — ISA sea-level air density (ICAO ISO 2533:1975)
}

LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray) -> np.ndarray:
    """Equivalent airspeed (m/s) for each species under the lift-equilibrium formula.

    X: (n, 2) — columns [mass_kg, wing_area_m2] in USED_INPUTS order.
    C_L, g, rho_0: all read from OTHER_CONSTANTS module namespace (no kwargs).

    Returns Ue_ms array of shape (n,).
    """
    C_L   = OTHER_CONSTANTS["C_L"]
    g     = OTHER_CONSTANTS["g"]
    rho_0 = OTHER_CONSTANTS["rho_0"]
    mass_kg = np.asarray(X[:, 0], dtype=float)
    wing_area_m2 = np.asarray(X[:, 1], dtype=float)
    Ue = np.sqrt(2.0 * mass_kg * g / (rho_0 * wing_area_m2 * C_L))
    return Ue
