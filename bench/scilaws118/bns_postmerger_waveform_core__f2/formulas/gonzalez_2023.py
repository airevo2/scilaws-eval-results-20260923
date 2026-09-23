"""Quasi-universal post-merger peak-frequency fit — Gonzalez et al. 2023.

Citation: A. Gonzalez et al., Class. Quantum Grav. 40 (2023) 085011 (arXiv:2210.16366).
Equations (23)-(24), PDF p. 21; coefficients from Table 2 (Mf2 row), PDF p. 22.

Formula
-------
    M*f2 = a0 * Q^M(X) * Q^S(S_hat, X) * Q^T(kappa_2^T, X)

where X = 1 - 4*nu, nu = m1*m2 / (m1+m2)^2 (symmetric mass ratio):

    Q^M(X)        = 1 + aM_1 * X
    Q^S(S_hat, X) = 1 + aS_1 * (1 + bS_1 * X) * S_hat  [set to 1 — see Caveats]
    Q^T(kT, X)    = (1 + pT_1*kT + pT_2*kT^2) / (1 + pT_3*kT + pT_4*kT^2)
    pT_k          = aT_k * (1 + bT_k * X)

LAW_CONSTANTS — Table 2, Mf2 row, PDF p. 22 (chi^2 = 0.067, 1sigma error 3.6%, R^2 = 0.958)
   a0   = 8.99e-2   (line 1678 of .txt)
   aM_1 = 31.02     (line 1746 of .txt)
   aT_1 = 2.94e-2   (line 1726 of .txt)
   bT_1 = 1.13      (line 1739 of .txt)
   aT_2 = 3.78e-5   (line 1727 of .txt)
   bT_2 = -0.99     (line 1740 of .txt)
   aT_3 = 5.75e-2   (line 1728 of .txt)
   bT_3 = 39.99     (line 1741 of .txt)
   aT_4 = 2.77e-4   (line 1729 of .txt)
   bT_4 = 27.77     (line 1742 of .txt)

OTHER_CONSTANTS
   GM_SUN_C3 = 4.925490947e-6 s — geometric-units factor G*M_sun/c^3 (CODATA).
   Unit conversion: f[kHz] = (M*f2) / (M[M_sun] * GM_SUN_C3) / 1000.

Type designation: Type I. All 10 coefficients are globally calibrated on the
full CoRe Release 2 catalogue (590 NR simulations). No per-cluster refit.
LOCAL_FITTABLE is empty.

Column mapping (paper -> released-CSV columns):
    M = mass   [M_sun] — total binary gravitational mass
    m1 = m1    [M_sun] — primary (heavier) neutron star mass
    m2 = m2    [M_sun] — secondary (lighter) neutron star mass
    kappa_2^T = kap2t  [dimensionless] — combined tidal coupling constant

Caveats:
    The released benchmark CSV does not carry component-spin columns (chiAz,
    chiBz are dropped as near-zero and not used by any baseline). We evaluate
    with S_hat = 0, so Q^S = 1 identically. This is the correct projection for
    the available inputs; the paper documents the spin correction as a secondary
    effect within the fit uncertainty.
"""

import numpy as np

USED_INPUTS = ["mass", "m1", "m2", "kap2t"]
PAPER_REF   = "summary_formula+dataset_gonzalez_2023.md"
EQUATION_LOC = "Eq. (23)-(24), PDF p. 21; Table 2 (Mf2 row), PDF p. 22"

# All 10 tidal/mass-ratio coefficients from Table 2, Mf2 row, PDF p. 22.
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a0":   8.99e-2,   # overall amplitude; txt line 1678
    "aM_1": 31.02,     # linear mass-ratio correction; txt line 1746
    "aT_1": 2.94e-2,   # tidal numerator coeff 1; txt line 1726
    "bT_1": 1.13,      # tidal X-coupling coeff 1; txt line 1739
    "aT_2": 3.78e-5,   # tidal numerator coeff 2; txt line 1727
    "bT_2": -0.99,     # tidal X-coupling coeff 2; txt line 1740
    "aT_3": 5.75e-2,   # tidal denominator coeff 1; txt line 1728
    "bT_3": 39.99,     # tidal X-coupling coeff 3; txt line 1741
    "aT_4": 2.77e-4,   # tidal denominator coeff 2; txt line 1729
    "bT_4": 27.77,     # tidal X-coupling coeff 4; txt line 1742
}

# GM_SUN_C3: CODATA geometric-units constant G*M_sun/c^3 = 4.925490947e-6 s.
# Not a discovery target — a universal unit conversion factor.
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "GM_SUN_C3": 4.925490947e-6,  # seconds; CODATA
}

LOCAL_FITTABLE = {}  # Type I — no per-cluster parameters; no fit()


def predict(
    X: np.ndarray,
    a0: float,
    aM_1: float,
    aT_1: float,
    bT_1: float,
    aT_2: float,
    bT_2: float,
    aT_3: float,
    bT_3: float,
    aT_4: float,
    bT_4: float,
) -> np.ndarray:
    """Predict f2 [kHz] for each BNS merger row.

    X: (n, 4) array — columns [mass, m1, m2, kap2t] per USED_INPUTS.
    Returns: (n,) array of f2 predictions in kHz.
    """
    GM_SUN_C3 = OTHER_CONSTANTS["GM_SUN_C3"]

    X = np.asarray(X, dtype=float)
    mass  = X[:, 0]
    m1    = X[:, 1]
    m2    = X[:, 2]
    kap2t = X[:, 3]

    # Symmetric mass ratio and mass-ratio combination X = 1 - 4*nu
    nu   = m1 * m2 / (mass * mass)
    Xv   = 1.0 - 4.0 * nu

    # Q^M factor: mass-ratio correction (Eq. 23)
    Q_M = 1.0 + aM_1 * Xv

    # Q^S factor: spin correction — set to 1 (no spin column; S_hat = 0)
    Q_S = 1.0

    # Q^T factor: rational tidal polynomial (Eq. 24)
    pT1 = aT_1 * (1.0 + bT_1 * Xv)
    pT2 = aT_2 * (1.0 + bT_2 * Xv)
    pT3 = aT_3 * (1.0 + bT_3 * Xv)
    pT4 = aT_4 * (1.0 + bT_4 * Xv)
    kT2 = kap2t * kap2t
    Q_T = (1.0 + pT1 * kap2t + pT2 * kT2) / (1.0 + pT3 * kap2t + pT4 * kT2)

    # Dimensionless mass-scaled frequency M*f2
    Mf2 = a0 * Q_M * Q_S * Q_T

    # Convert to kHz: f[kHz] = Mf2 / (mass * GM_SUN_C3) / 1000
    return Mf2 / (mass * GM_SUN_C3) / 1000.0
