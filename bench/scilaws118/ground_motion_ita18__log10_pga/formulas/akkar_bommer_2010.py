"""Akkar & Bommer (2010) Ground Motion Prediction Equation (GMPE) for PGA.

Citation:
    Akkar, S., and Bommer, J. J. (2010).
    Empirical Equations for the Prediction of PGA, PGV, and Spectral Accelerations
    in Europe, the Mediterranean Region, and the Middle East.
    Seismological Research Letters, 81(2), 195-206.
    DOI: 10.1785/gssrl.81.2.195

Formula (Eq. 1, p. 198; Table 1 PGA row, pp. 200-201):

    log10(Y) = b1 + b2*M + b3*M^2
               + (b4 + b5*M) * log10(sqrt(Rjb^2 + b6^2))
               + b7*Ss + b8*Sa
               + b9*Fn + b10*Fr

where Y is PGA in cm/s^2, M is moment magnitude (Mw),
Rjb is Joyner-Boore distance (km),
Ss = 1 for soft soil (Vs30 < 360 m/s, AB-2010 class),
Sa = 1 for stiff soil (Vs30 360-750 m/s, AB-2010 class),
rock (Vs30 > 750 m/s) is the reference class (Ss=Sa=0),
Fn = 1 for normal faulting, Fr = 1 for reverse/thrust faulting,
strike-slip (and unknown) is the reference class (Fn=Fr=0).

LAW_CONSTANTS — verbatim from AB-2010 Table 1 PGA row, pp. 200-201
(independently verified against GEM/OpenQuake implementation at
https://raw.githubusercontent.com/gem/oq-engine/master/openquake/hazardlib/gsim/akkar_bommer_2010.py):
    b1  =  1.43525   Table 1, PGA row
    b2  =  0.74866   Table 1, PGA row
    b3  = -0.06520   Table 1, PGA row
    b4  = -2.72950   Table 1, PGA row
    b5  =  0.25139   Table 1, PGA row
    b6  =  7.74959   Table 1, PGA row (distance saturation parameter, km)
    b7  =  0.08320   Table 1, PGA row (soft soil Ss coefficient)
    b8  =  0.00766   Table 1, PGA row (stiff soil Sa coefficient)
    b9  = -0.05823   Table 1, PGA row (normal faulting Fn coefficient)
    b10 =  0.07087   Table 1, PGA row (reverse faulting Fr coefficient)

OTHER_CONSTANTS — none; AB-2010 Eq. 1 has no external structural constants
    (all structural terms — b6 as the saturation distance — are captured
    in LAW_CONSTANTS; Eq. 1 is dimensionally self-contained).

Type I — each station-event recording is an independent row. The GMPE has
no per-cluster fitted parameters; all coefficients are globally published
constants from Table 1. LOCAL_FITTABLE is empty; no fit() needed.

Column mapping (paper notation → released CSV columns, USED_INPUTS order):
    M     → Mw         moment magnitude
    Rjb   → R_km       Joyner-Boore distance (km); confirmed in metadata.yaml
    Ss    → ec8_C + ec8_D + ec8_E  (Vs30 < 360 m/s — soft, medium-soft, special-soft)
    Sa    → ec8_B                  (Vs30 360-800 m/s — stiff soil, matches AB-2010 360-750)
    rock  → ec8_A = 1              reference class (Ss=Sa=0)
    Fn    → sof_NF
    Fr    → sof_TF
    SS/unknown → sof_SS / none     reference class (Fn=Fr=0)

Spot-check verification (orchestrator + W117):
  M=6.0, Rjb=10.0, rock (Ss=Sa=0), unknown faulting (Fn=Fr=0):
  b1+b2*M+b3*M^2 = 1.43525+4.49196-2.34720 = 3.58001
  log10(sqrt(100+60.056)) = log10(12.651) = 1.10214
  (b4+b5*M)*log10(...) = (-1.22116)*1.10214 = -1.34588
  log10(PGA) = 3.58001 - 1.34588 = 2.23413 ≈ 2.234 [PASS]

OA source used: GEM/OpenQuake implementation (coefficients confirmed identical
to Table 1 PGA row); W117 sweep identified this baseline as the S2-P7 gap fix.
"""

import numpy as np

USED_INPUTS = [
    "Mw", "R_km",
    "ec8_B", "ec8_C", "ec8_D", "ec8_E",
    "sof_NF", "sof_TF",
]
PAPER_REF = "akkar_bommer_2010"
EQUATION_LOC = "Eq. 1, p. 198; Table 1 (PGA row), pp. 200-201"

# All 10 values from Table 1, PGA row, pp. 200-201 of AB-2010 SRL 81(2)
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "b1":  1.43525,   # Table 1, PGA row — constant offset
    "b2":  0.74866,   # Table 1, PGA row — linear magnitude scaling
    "b3": -0.06520,   # Table 1, PGA row — quadratic magnitude scaling
    "b4": -2.72950,   # Table 1, PGA row — geometric spreading slope
    "b5":  0.25139,   # Table 1, PGA row — magnitude-dependent spreading
    "b6":  7.74959,   # Table 1, PGA row — distance saturation (km)
    "b7":  0.08320,   # Table 1, PGA row — soft soil Ss coefficient
    "b8":  0.00766,   # Table 1, PGA row — stiff soil Sa coefficient
    "b9": -0.05823,   # Table 1, PGA row — normal faulting coefficient
    "b10": 0.07087,   # Table 1, PGA row — reverse faulting coefficient
}

# No non-LAW structural constants in AB-2010 Eq. 1 for PGA
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}

LOCAL_FITTABLE = {}  # Type I — no per-cluster parameters


def predict(
    X: np.ndarray,
    b1: float, b2: float, b3: float,
    b4: float, b5: float, b6: float,
    b7: float, b8: float,
    b9: float, b10: float,
) -> np.ndarray:
    """Predict log10(PGA [cm/s^2]) for each station-event row.

    X.shape = (n, 8) — columns in USED_INPUTS order:
        0: Mw
        1: R_km  (Joyner-Boore distance)
        2: ec8_B (stiff soil Sa=1 if ec8_B==1)
        3: ec8_C (soft soil component)
        4: ec8_D (soft soil component)
        5: ec8_E (soft soil component)
        6: sof_NF (normal faulting Fn)
        7: sof_TF (reverse/thrust faulting Fr)

    Ss = ec8_C + ec8_D + ec8_E  (AB-2010 soft soil class: Vs30 < 360 m/s)
    Sa = ec8_B                  (AB-2010 stiff soil class: Vs30 360-750 m/s)
    Rock (ec8_A) is reference: Ss=Sa=0
    Strike-slip (sof_SS) is reference: Fn=Fr=0
    """
    M    = X[:, 0]
    Rjb  = X[:, 1]
    Sa   = X[:, 2]                     # ec8_B — stiff soil
    Ss   = X[:, 3] + X[:, 4] + X[:, 5]  # ec8_C + ec8_D + ec8_E — soft soil
    Fn   = X[:, 6]                     # sof_NF — normal faulting
    Fr   = X[:, 7]                     # sof_TF — reverse/thrust faulting

    # Eq. 1: magnitude scaling terms
    mag_term = b1 + b2 * M + b3 * M**2

    # Eq. 1: distance scaling with saturation
    dist_eff = np.sqrt(Rjb**2 + b6**2)
    dist_term = (b4 + b5 * M) * np.log10(dist_eff)

    # Eq. 1: site amplification (rock = reference; Ss=Sa=0)
    site_term = b7 * Ss + b8 * Sa

    # Eq. 1: style-of-faulting (SS + unknown = reference; Fn=Fr=0)
    fault_term = b9 * Fn + b10 * Fr

    return mag_term + dist_term + site_term + fault_term
