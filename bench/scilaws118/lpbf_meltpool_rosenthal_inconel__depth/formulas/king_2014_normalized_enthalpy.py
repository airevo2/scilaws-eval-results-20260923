"""King et al. 2014 — Normalized-enthalpy scaling for LPBF melt-pool depth.

King, W. E., Barth, H. D., Castillo, V. M., Gallegos, G. F., Gibbs, J. W.,
Hahn, D. E., Kamath, C., & Rubenchik, A. M. (2014). Observation of
keyhole-mode laser melting in laser powder-bed fusion additive
manufacturing. Journal of Materials Processing Technology, 214(12),
2915-2925. DOI: 10.1016/j.jmatprotec.2014.06.005.

=== Formula ===
King 2014, Eq. 2.6 (PDF pp. 7-8): the normalized enthalpy is

    DeltaH/hs = A_abs * P / (hs * sqrt(pi * D_diff * u * sigma^3))

where hs = rho * c * Tm is the volumetric enthalpy at melting (J/m^3),
equivalently hs = rho * hs_specific where hs_specific is the specific
enthalpy at melting (J/kg), D_diff = k/(rho*c) is the thermal diffusivity,
A_abs is absorptivity, u is scan speed (m/s), sigma is the beam 1/e^2 radius.

The formula with explicit rho and hs_specific (from Table 3, PDF p.36):
    DeltaH/hs = A_abs * P / (rho * hs_specific * sqrt(pi * D_diff * u * sigma^3))

Dimensional check: [W] / ([kg/m³]*[J/kg]*sqrt([m²/s]*[m/s]*[m³]))
  = [J/s] / ([J/m³]*[m³/s]) = dimensionless. ✓

King 2014 Section 5.3 (PDF pp. 15-17, Fig. 8 description): depth normalized
by beam radius d/sigma collapses onto a band about a line when plotted
against DeltaH/hs. From the paper: "the measured melt pool depths normalized
to the beam size is zero below a normalized enthalpy of ~10. Above this
value, they increase gradually to a value of ~3 at normalized enthalpy
of ~26" (PDF p. 15). Separator: "two lines for D4sigma=52 and 130 um
have about the same slope but are separated by d/sigma = 0.2" (PDF p. 17).

Approximate linear relationship from graphical description:
    d / sigma = max(0, B_slope * (DeltaH/hs - E_threshold))

with B_slope ≈ (3.0)/(26.0 - 10.0) = 0.1875 and E_threshold ≈ 10.0.

NOTE: King 2014 does NOT tabulate explicit slope/intercept values; the
values above are read graphically from the figure description in §5.3. The
paper's headline contribution is the dimensionless framework (Eq. 2.6), but
B_slope and E_threshold are nonetheless the *defining coefficients* of the
depth relationship (the empirical slope + onset that map DeltaH/hs → depth),
i.e. the formula's discovery target → they are LAW_CONSTANTS, not OTHER.

=== LAW_CONSTANTS ===
Defining coefficients of the depth law (slope + onset of the
d/sigma vs DeltaH/hs line, the formula's discovery target):
    B_slope     = 0.1875  — slope of d/sigma vs DeltaH/hs in conduction regime
    E_threshold = 10.0    — onset of measurable melt depth (DeltaH/hs units)

These two are read graphically from King 2014 Section 5.3 (PDF pp. 15-17):
"the measured melt pool depths normalized to the beam size is zero below a
normalized enthalpy of ~10. Above this value, they increase gradually to a
value of ~3 at normalized enthalpy of ~26" → E_threshold ≈ 10 and
B_slope ≈ (3.0)/(26.0 - 10.0) = 0.1875. King does NOT tabulate them, but
they are the empirical slope/intercept that convert the dimensionless
DeltaH/hs into a depth — i.e. the depth law's defining coefficients (LAW),
NOT a universal/derived/structural given (OTHER). They are read from King's
own figure, NOT refit on this benchmark's train split.

=== OTHER_CONSTANTS ===
Material constants for 316L stainless steel from Table 3 (PDF p. 36),
captioned "Values of constants used in calculation of the normalized
enthalpy" — i.e. known/handbook material properties the formula merely
*consumes* (not coefficients King fitted). Each is a standard 316L value
(the enthalpy is explicitly literature-cited, "Rai et al., 2007"), so per
the bidirectional field-classification rule (a handbook/standard GIVEN a
formula plugs in → OTHER) they live in OTHER and predict() reads them from
the OTHER_CONSTANTS dict (gold/weizsacker e2 style):
    A_abs_316L   = 0.4         — absorptivity (assumed/handbook) [Table 3, PDF p.36]
    rho_316L     = 7980.0      — density, kg/m^3 (standard 316L ~7980-8000) [Table 3]
    hs_J_kg_316L = 1.2e6       — specific enthalpy at melting, J/kg (Rai et al. 2007) [Table 3]
    D_diff_316L  = 5.38e-6     — thermal diffusivity, m^2/s (handbook 316L) [Table 3]

=== Type designation ===
Type I — universal formula, no per-cluster refit. LOCAL_FITTABLE = {}.
King 2014 Section 5.3 claims the scaling collapses data across different
powers, speeds, and beam sizes into one curve.

=== Column mapping ===
    laser_power_W         → P (W)
    scan_velocity_mm_per_s → u (converted mm/s → m/s)
    beam_diameter_um      → D4sigma (converted um → m); sigma = D4sigma/4

=== Caveats ===
1. Slope/threshold are graphical readings from King 2014 Fig. 5 / §5.3,
   NOT tabulated in the paper. They are the depth line's defining
   coefficients → LAW_CONSTANTS (the SR discovery target).
2. Material constants (Table 3) are for 316L SS. The benchmark trains on
   IN718 and tests on IN625 — cross-material application tests universality.
3. Formula is only valid in conduction mode (DeltaH/hs < ~30). In keyhole
   mode the depth jumps sharply and the linear approximation fails.
4. sigma = D4sigma/4: in King 2014's convention D4sigma = 4*sigma (beam
   diameter at 2-sigma) so sigma = D4sigma/4 (1/e^2 radius).
"""

import numpy as np

USED_INPUTS  = ["laser_power_W", "scan_velocity_mm_per_s", "beam_diameter_um"]
PAPER_REF    = "summary_formula_king_2014.md"
EQUATION_LOC = ("Eq. 2.6, PDF pp. 7-8; Table 3, PDF p. 36; "
                "Fig. 8b description, PDF pp. 15-17")

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    # Defining coefficients of the depth law d/sigma = B_slope*(DeltaH/hs - E_threshold).
    # Read graphically from King 2014 Fig. 5 / Section 5.3 (PDF pp. 15-17): the slope and
    # onset of the conduction-regime depth line are the only coefficients that turn the
    # dimensionless DeltaH/hs into a depth prediction — i.e. the formula's discovery target,
    # not a universal/derived/structural given. They are NOT fit on this benchmark's train.
    "B_slope":     0.1875,      # slope d/sigma per unit DeltaH/hs (3/(26-10), §5.3)
    "E_threshold": 10.0,        # DeltaH/hs onset of measurable melt depth (§5.3, "zero below ~10")
}
# === OTHER_CONSTANTS — known/handbook 316L material givens the formula consumes ===
# King 2014 Table 3 (PDF p. 36), "Values of constants used in calculation of the
# normalized enthalpy" — standard 316L material properties plugged into Eq. 2.6, NOT
# coefficients King fitted (hs is explicitly literature-cited, Rai et al. 2007). Per the
# bidirectional field rule these are GIVENs → OTHER; predict reads them from this dict.
OTHER_CONSTANTS = {
    "A_abs_316L":   0.4,        # absorptivity, dimensionless (assumed/handbook) [Table 3]
    "rho_316L":     7980.0,     # density, kg/m^3 (standard 316L ~7980-8000)     [Table 3]
    "hs_J_kg_316L": 1.2e6,      # specific enthalpy at melt, J/kg (Rai et al. 2007) [Table 3]
    "D_diff_316L":  5.38e-6,    # thermal diffusivity, m^2/s (handbook 316L)     [Table 3]
}
LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray,
            B_slope: float,
            E_threshold: float) -> np.ndarray:
    """Predict melt-pool depth (um) via King 2014 normalized-enthalpy scaling.

    X: (n, 3) — columns [laser_power_W, scan_velocity_mm_per_s, beam_diameter_um].
    Returns depth in micrometers.
    """
    # 316L material givens (handbook / Table 3) read from OTHER_CONSTANTS (gold style)
    A_abs_316L   = OTHER_CONSTANTS["A_abs_316L"]
    rho_316L     = OTHER_CONSTANTS["rho_316L"]
    hs_J_kg_316L = OTHER_CONSTANTS["hs_J_kg_316L"]
    D_diff_316L  = OTHER_CONSTANTS["D_diff_316L"]

    P       = np.asarray(X[:, 0], dtype=float)            # W
    u       = np.asarray(X[:, 1], dtype=float) * 1.0e-3   # mm/s → m/s
    D4sigma = np.asarray(X[:, 2], dtype=float) * 1.0e-6   # um → m
    sigma   = D4sigma / 4.0                                # 1/e^2 radius, m
    sigma_um = np.asarray(X[:, 2], dtype=float) / 4.0     # 1/e^2 radius, um

    # Eq. 2.6 corrected: DeltaH/hs = A_abs * P / (rho * hs_specific * sqrt(pi * D * u * sigma^3))
    sigma3  = np.power(np.maximum(sigma, 1.0e-15), 3.0)
    u_safe  = np.where(u > 0, u, 1.0e-30)
    arg     = np.pi * D_diff_316L * u_safe * sigma3
    arg     = np.maximum(arg, 1.0e-60)
    denom   = rho_316L * hs_J_kg_316L * np.sqrt(arg)

    E_star  = A_abs_316L * P / denom

    # Linear model: d/sigma = max(0, B_slope * (E_star - E_threshold))
    d_over_sigma = np.maximum(0.0, B_slope * (E_star - E_threshold))

    return sigma_um * d_over_sigma
