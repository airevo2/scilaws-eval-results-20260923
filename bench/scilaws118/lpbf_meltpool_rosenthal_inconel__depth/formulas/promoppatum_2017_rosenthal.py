"""Promoppatum et al. 2017 — Rosenthal moving-point-source melt-pool depth.

Promoppatum, P., Yao, S.-C., Pistorius, P. C., & Rollett, A. D. (2017).
A Comprehensive Comparison of the Analytical and Numerical Prediction of
the Thermal History and Solidification Microstructure of Inconel 718
Products Made by Laser Powder-Bed Fusion. Engineering, 3(5), 685-694.
DOI: 10.1016/J.ENG.2017.05.023. CC BY-NC-ND 4.0.

=== Formula ===
Rosenthal (1946) moving point-source temperature field (Eq. 4, PDF p. 4):

    T = T0 + (lambda * P / (2*pi*k*r)) * exp(-V*(r + xi)/(2*alpha))

where xi = x - V*t (moving frame), r = sqrt(xi^2 + y^2 + z^2),
alpha = k / (rho*CP) is the thermal diffusivity.

Setting dy/dxi = 0 at the melt pool boundary for materials with
low thermal diffusivity, Tang et al. (cited in Promoppatum) show the
melt pool width W satisfies (Eq. 6, PDF p. 4):

    W ≈ sqrt(8 * lambda * P / (pi * e * rho * CP * V * (Tm - T0)))

Promoppatum 2017 (PDF p. 5, below Eq. 6): "the calculated melt pool depth
is half of the melt pool width" (semi-circular cross-section assumption of
the Rosenthal point-source). Therefore:

    D = W / 2 = sqrt(2 * lambda * P / (pi * e * rho * CP * V * (Tm - T0)))

Converting to micrometers:
    D_um = D_m * 1e6

=== LAW_CONSTANTS ===
The one coefficient Promoppatum 2017 *fitted* for the Rosenthal equation:
    absorptivity = 0.4  — Promoppatum varied absorptivity over the literature
                          range 0.3-0.87 (Table 1/Table 2) and reports that
                          "the fitted absorptivities that yield the closest
                          agreement with the experiment are found to be 0.4
                          for the Rosenthal equation" (PDF p. 5, txt L422).
                          This is the paper's calibrated/fitted coefficient,
                          not a handbook given → LAW.

=== OTHER_CONSTANTS ===
Known/handbook material GIVENs the formula consumes (NOT fitted by the
paper). Table 2 (PDF p. 4) is captioned "Room-temperature thermal properties
of Inconel 718 used in the Rosenthal equation"; every value is literature-
cited ([11]). Per the bidirectional field rule a standard/handbook material
property a formula plugs in → OTHER (predict reads them from this dict,
gold/weizsacker e2 style):
    rho_kg_m3 = 8220  kg/m^3   — IN718 density [Table 2, ref [11]]
    CP_J_kgK  = 435   J/(kg·K) — IN718 specific heat [Table 2, ref [11]]
    Tm_C      = 1340  deg C    — IN718 avg melting temp (1613 K [16] → °C),
                                 PDF p. 3 Eq. (1) context (txt L173)
    e_euler   = 2.71828182845904523536  — Euler's number e (structural
              prefactor from the Rosenthal/Tang derivation, not a material
              property; implicit in Eq. 6 of Promoppatum 2017)
    T0_C = 25.0  — baseline substrate temperature (ambient, deg C);
                   constant for all rows in the released dataset.

=== Type designation ===
Type I — each row is an independent single laser track; no per-cluster
refit needed. The Rosenthal formula is universal: given fixed material
constants and process parameters, depth is predicted without any
per-specimen or per-machine fit parameter. LOCAL_FITTABLE = {}.

=== Column mapping ===
Paper notation → released CSV:
    P (laser power)          → laser_power_W
    V (scanning velocity)    → scan_velocity_mm_per_s  [converted to m/s]
    D4sigma (beam diameter)  → beam_diameter_um  [not directly used here;
                                Rosenthal is a point-source model, beam
                                geometry enters only implicitly via the
                                absorptivity's regime]

=== Caveats ===
1. The Rosenthal equation assumes a point heat source, temperature-
   independent properties, no convection, and negligible surface losses.
   It is known to overestimate width at high heat inputs (keyhole regime),
   as noted in Promoppatum 2017 PDF p. 5.
2. Material constants are for IN718; they are used for both IN718 (train)
   and IN625 (test). IN625 differs: k~9.8 W/(m·K), rho~8440 kg/m³,
   CP~410 J/(kg·K), Tm~1290 C (from baselines.py documentation). This
   cross-material application is intentional for the OOD split.
3. The D = W/2 relation assumes a perfectly semi-circular cross-section,
   which is an approximation.
4. beam_diameter_um is listed in USED_INPUTS for schema completeness
   but is not consumed in the computation (point-source model).
"""

import numpy as np

USED_INPUTS = ["laser_power_W", "scan_velocity_mm_per_s", "beam_diameter_um"]
PAPER_REF   = "summary_formula_promoppatum_2017.md"
EQUATION_LOC = "Eq. 6 + depth=W/2 note, Promoppatum 2017, PDF pp. 4-5"

# === LAW_CONSTANTS — paper-fitted coefficient, frozen ===
LAW_CONSTANTS = {
    "absorptivity": 0.4,   # dimensionless — paper-FITTED for Rosenthal ("closest agreement", PDF p.5 txt L422)
}
# === OTHER_CONSTANTS — known/handbook material givens + universal physics factors ===
# IN718 material properties (Table 2, ref [11]) are literature-cited GIVENs the formula
# consumes, not coefficients the paper fitted → OTHER (bidirectional field rule).
OTHER_CONSTANTS = {
    "rho_kg_m3": 8220.0,  # kg/m^3  — IN718 density, Table 2 PDF p.4 (ref [11])
    "CP_J_kgK":  435.0,   # J/(kg·K)— IN718 specific heat, Table 2 PDF p.4 (ref [11])
    "Tm_C":      1340.0,  # deg C   — IN718 avg melting temp (~1613 K [16]→°C), PDF p.3 Eq.1 context
    "e_euler": 2.71828182845904523536,  # Euler's number e — structural prefactor
    "T0_C":    25.0,                    # deg C — ambient substrate temp, constant all rows
    "k_W_mK":  11.4,                    # W/(m·K) IN718 thermal conductivity — cited in Promoppatum Table 2 PDF p.4 BUT not consumed by Eq.6 (cancels in the derivation; retained as reference/citation companion only)
}
LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray,
            absorptivity: float) -> np.ndarray:
    """Predict melt-pool depth (um) via Rosenthal Eq. 6 + D=W/2.

    X: (n, 3) — columns [laser_power_W, scan_velocity_mm_per_s, beam_diameter_um].
    Returns depth in micrometers.
    """
    e = OTHER_CONSTANTS["e_euler"]
    T0 = OTHER_CONSTANTS["T0_C"]
    # IN718 material givens (handbook / Table 2) read from OTHER_CONSTANTS (gold style)
    rho_kg_m3 = OTHER_CONSTANTS["rho_kg_m3"]
    CP_J_kgK  = OTHER_CONSTANTS["CP_J_kgK"]
    Tm_C      = OTHER_CONSTANTS["Tm_C"]

    P   = np.asarray(X[:, 0], dtype=float)   # W
    V   = np.asarray(X[:, 1], dtype=float) * 1.0e-3   # mm/s → m/s
    # beam_diameter_um is column 2 but not used in point-source Rosenthal
    # (index kept for schema alignment)

    dT = np.maximum(Tm_C - T0, 1.0)  # temperature rise to melt, deg C = K

    # Eq. 6 Promoppatum 2017 (PDF p.4):
    # W = sqrt(8 * absorptivity * P / (pi * e * rho * CP * V * dT))
    numerator   = 8.0 * absorptivity * P
    denominator = np.pi * e * rho_kg_m3 * CP_J_kgK * V * dT
    denominator = np.where(denominator > 0, denominator, 1.0e-30)

    W_m = np.sqrt(numerator / denominator)   # melt pool width, metres

    # D = W/2 (semi-circular cross-section; PDF p.5 below Eq.6)
    D_m = W_m / 2.0
    return D_m * 1.0e6   # metres → micrometers
