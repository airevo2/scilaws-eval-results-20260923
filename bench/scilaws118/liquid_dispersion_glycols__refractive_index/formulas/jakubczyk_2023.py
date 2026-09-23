"""Jakubczyk (2023) two-pole Sellmeier + thermal dispersion model — n(lambda, T).

Jakubczyk D, Derkachov G, Nyandey K, Alikhanzadeh-Arani S, Derkachova A
(2023). "Chromatic dispersion and thermal coefficients of hygroscopic liquids:
5 glycols and glycerol." Scientific Data 10:894.
DOI: 10.1038/s41597-023-02819-3.

Three-equation model (Eqs. 1-3, PDF pp. 3, 8):

  Eq. 3 - Two-pole Sellmeier (dispersion at 20 degC reference):
      n^2(lambda, 20) = A + B_IR * lambda^2 / (lambda^2 - C_IR)
                          + B_UV * lambda^2 / (lambda^2 - C_UV)

  Eq. 2 - Rational thermal coefficient:
      dn/dT(lambda) = A_T + B_T / (lambda - C_T)

  Eq. 1 - Linear temperature shift:
      n(lambda, T) = n(lambda, 20) + dn/dT(lambda) * (T - 20)

All lambda values in MICROMETRES (um), as stated explicitly in PDF p. 8
caption of Tables 2 and 3.

Type designation: Type I.
LAW_CONSTANTS are published values for ethylene glycol (EG / MEG) from
Jakubczyk (2023) Table 2 (Sellmeier, PDF p. 8) and Table 3 (thermal, PDF p. 8).
These are paper-frozen values for a single liquid; no fitting is performed.

The benchmark uses EG-only data (single liquid) for TypeI validity. Using all 6
glycols without liquid-identity information would yield inter-liquid variance > 70%
of total variance, making any TypeI formula achieve R2 < 0.30.

Note on poles (FM-J4 check):
  C_IR = 3.12 um^2 -> IR pole at sqrt(3.12) ~ 1.766 um (well above data max 1.071 um -> SAFE)
  C_UV = 0.01332 um^2 -> UV pole at sqrt(0.01332) ~ 0.1154 um (well below data min 0.394 um -> SAFE)
  C_T = 0.17 um -> thermal pole at lambda = 0.17 um (well below data min 0.394 um -> SAFE)
  All poles are outside the measurement range [0.394, 1.071] um. Predict() clips
  n^2 >= 0 as an additional numerical safety guard.

LAW_CONSTANTS from Jakubczyk (2023), ethylene glycol (EG / MEG) row:
  Table 2 (PDF p. 8) - Sellmeier coefficients:
    A     = 1.34238
    B_IR  = 0.0137
    C_IR  = 3.12  (um^2)
    B_UV  = 0.68263
    C_UV  = 0.01332  (um^2)
  Table 3 (PDF p. 8) - Thermal coefficients:
    A_T   = -2.643e-4  (K^-1)
    B_T   = -7.5e-6    (K^-1 um)
    C_T   = 0.17       (um)
"""

import numpy as np

USED_INPUTS = ["lambda_um", "temperature_C"]
PAPER_REF = "summary_formula_dataset_jakubczyk_2023.md"
EQUATION_LOC = (
    "Jakubczyk (2023) Eq. 3 (Sellmeier, PDF p. 3/8), "
    "Eq. 2 (thermal coefficient, PDF p. 3), "
    "Eq. 1 (linear T shift, PDF p. 3). "
    "LAW_CONSTANTS from Table 2 (Sellmeier) and Table 3 (thermal), "
    "ethylene glycol (EG/MEG) row, PDF p. 8."
)

# LAW_CONSTANTS - Jakubczyk (2023) Table 2 and Table 3, ethylene glycol (EG/MEG) row, PDF p. 8
LAW_CONSTANTS = {
    "A":    1.34238,    # Sellmeier constant term (dimensionless); Table 2 EG row
    "B_IR": 0.0137,     # IR resonance strength (dimensionless); Table 2 EG row
    "C_IR": 3.12,       # IR resonance pole = lambda_IR^2 (um^2); Table 2 EG row
    "B_UV": 0.68263,    # UV resonance strength (dimensionless); Table 2 EG row
    "C_UV": 0.01332,    # UV resonance pole = lambda_UV^2 (um^2); Table 2 EG row
    "A_T":  -2.643e-4,  # thermal coeff constant term (K^-1); Table 3 EG row
    "B_T":  -7.5e-6,    # thermal coeff resonance (K^-1 um); Table 3 EG row
    "C_T":  0.17,       # thermal resonance wavelength shift (um); Table 3 EG row
}

# Reference temperature for Eq. 1's linear-T expansion n(lambda,T) = n(lambda,20) + dn/dT*(T - T_ref).
# Structural protocol value (Jakubczyk 2023, PDF p. 3) — NOT a discoverable law constant, so it lives in
# OTHER_CONSTANTS, and it is exposed to the SR system as the `reference_temperature_sellmeier` candidate
# prior (metadata.yaml). The Type I harness calls predict(X, **LAW_CONSTANTS) only (OTHER_CONSTANTS is not
# passed), so predict() reads T_ref from this dict directly.
OTHER_CONSTANTS = {"T_ref": 20.0}    # degC

LOCAL_FITTABLE = {}     # Type I - no per-cluster parameters.


def predict(X: np.ndarray,
            A: float, B_IR: float, C_IR: float,
            B_UV: float, C_UV: float,
            A_T: float, B_T: float, C_T: float) -> np.ndarray:
    """Full Jakubczyk n(lambda, T) = n(lambda,20) + dn/dT*(T-20).

    Implements Eqs. 1-3 from Jakubczyk (2023) with paper-published MEG constants.

    X: (n, 2) - columns [lambda_um, temperature_C].
    Returns: (n,) array of predicted refractive index.

    FM-J4 note: poles are at sqrt(C_IR) ~ 1.766 um (IR) and sqrt(C_UV) ~ 0.1154 um (UV),
    both well outside the measured range [0.394, 1.071] um. Thermal pole at C_T = 0.17 um
    is also well below the data minimum (0.394 um). All poles are safe. Clip n^2 >= 0.
    """
    lam = np.asarray(X[:, 0], dtype=float)
    T   = np.asarray(X[:, 1], dtype=float)

    # Eq. 3 - two-pole Sellmeier at the 20 degC reference (T_ref)
    lam2 = lam * lam
    n2 = A + B_IR * lam2 / (lam2 - C_IR) + B_UV * lam2 / (lam2 - C_UV)
    n20 = np.sqrt(np.clip(n2, 1e-6, None))   # clip for numerical safety

    # Eq. 2 - rational thermal coefficient
    # C_T is a wavelength-shift parameter; lam - C_T should not be zero in valid range
    denom = lam - C_T
    # Safety clip: if denom approaches zero (lambda near C_T pole), clamp to avoid singularity.
    # For MEG C_T = 0.17 um: all data at lambda in [0.394, 1.071] um -> denom in [0.224, 0.901] um.
    safe_denom = np.where(np.abs(denom) > 1e-6, denom, 1e-6)
    dndt = A_T + B_T / safe_denom

    # Eq. 1 - linear temperature shift (T_ref = structural reference temperature, from OTHER_CONSTANTS)
    T_ref = OTHER_CONSTANTS["T_ref"]
    n = n20 + dndt * (T - T_ref)

    # Physical clip: n must be positive (condensed optical medium)
    return np.clip(n, 1e-3, None)
