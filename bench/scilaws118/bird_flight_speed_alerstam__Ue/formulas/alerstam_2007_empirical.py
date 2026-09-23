"""alerstam_2007_empirical.py — bird_flight_speed_alerstam__Ue

Alerstam T, Rosén M, Bäckman J, Ericson PGP, Hellgren O (2007)
"Flight speeds among bird species: Allometric and phylogenetic effects."
PLoS Biology 5(8): e197. doi:10.1371/journal.pbio.0050197

Empirical reduced-major-axis (RMA) regression formula from Table 1 and
Figure 1 caption (PDF p. 3–4):

    Ue = a × Q^c

where Q = m × g / S is the wing loading (N/m²), and the coefficients
are from the "All species" row (n=129) of Table 1 (Ue vs. wing loading):

    a = 4.3   (amplitude coefficient; 95% CI: 4.0–4.6; Table 1, PDF p. 4)
    c = 0.31  (scaling exponent; 95% CI: 0.27–0.35; Table 1, PDF p. 4)

This is the empirical scaling counterpart to the theoretical
lift-equilibrium prediction (Ue ~ Q^0.5) in alerstam_2007.py.
The paper's central finding is that this observed exponent (0.31) is
significantly smaller than the theoretical aerodynamic prediction (0.50).

Figure 1 caption (PDF p. 3): "The lines show the scaling relationships
Ue = 15.9 × (mass)^0.13 and Ue = 4.3 × (wing loading)^0.31 as calculated
by reduced major axis regression for all species (Table 1)."

LAW_CONSTANTS (Table 1, PDF pp. 3–4; Figure 1 caption, PDF p. 3):
    a = 4.3    — amplitude of the Ue–Q power law (m/s per (N/m²)^c)
    c = 0.31   — empirical RMA scaling exponent of Ue vs. wing loading Q

OTHER_CONSTANTS (universal physics):
    g = 9.81   m/s² — standard gravitational acceleration (CODATA / ISO 80000)

Note: the paper uses wing loading Q = mg/S (N/m²), which requires g to
combine the measured mass and wing area into the force-per-area quantity.
g is a universal physical constant, not a scientific discovery.

Type designation: Type I — one universal empirical scaling law for all
species; no per-species or per-group parameters. LOCAL_FITTABLE = {}.

Column mapping (paper notation → released CSV):
    m   (body mass, kg)       → mass_kg        [kg]
    S   (wing area, m²)       → wing_area_m2   [m²]
    Ue  (equiv. airspeed, m/s) → Ue_ms         [m/s]  (target, column 0)

Caveats:
- The Table 1 coefficient (a=4.3, c=0.31) is for "All species" (n=129)
  at the species level using reduced major axis regression.
- This formula is purely empirical (fitted to the Alerstam 2007 data);
  the theoretical lift-equilibrium formula (Ue = sqrt(2Q/(rho_0*C_L)))
  is in alerstam_2007.py. Both formulae are from the same paper.
- The empirical exponent c=0.31 (vs. theoretical 0.50) documents the
  paper's key finding: aerodynamic scaling predictions overestimate the
  speed–wing-loading relationship in real birds.
"""

import numpy as np

USED_INPUTS = ["mass_kg", "wing_area_m2"]
PAPER_REF = "summary_formula+dataset_alerstam_2007.md"
EQUATION_LOC = "Table 1 (Ue vs. wing loading, All species row), PDF p. 4; Figure 1 caption, PDF p. 3"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a": 4.3,   # amplitude coefficient (m/s per (N/m²)^c); Table 1, PDF p. 4
    "c": 0.31,  # empirical RMA exponent of Ue vs. Q; Table 1, PDF p. 4
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "g": 9.81,  # m/s² — standard gravitational acceleration (CODATA / ISO 80000)
}

LOCAL_FITTABLE = {}  # Type I — no per-species parameters


def predict(X: np.ndarray, a: float, c: float) -> np.ndarray:
    """Empirical equivalent airspeed (m/s) from wing loading power law.

    Ue = a * (m * g / S)^c

    where Q = m * g / S is the wing loading in N/m².

    X: (n, 2) — columns [mass_kg, wing_area_m2] in USED_INPUTS order.
    a, c: empirical regression coefficients from Alerstam 2007 Table 1
        (arrive via predict(X, **LAW_CONSTANTS); gold style — no defaults).
    g: fixed at 9.81 m/s² (universal; read from OTHER_CONSTANTS).

    Returns Ue_ms array of shape (n,).
    """
    g = OTHER_CONSTANTS["g"]
    mass_kg = np.asarray(X[:, 0], dtype=float)
    wing_area_m2 = np.asarray(X[:, 1], dtype=float)
    Q = mass_kg * g / wing_area_m2   # wing loading, N/m²
    return a * Q**c
