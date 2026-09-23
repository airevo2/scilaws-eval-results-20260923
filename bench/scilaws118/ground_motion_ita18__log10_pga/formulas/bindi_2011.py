"""Bindi et al. (2011) Ground Motion Prediction Equation (GMPE) for PGA — ITA10 model.

Citation:
    Bindi, D., Pacor, F., Luzi, L., Puglia, R., Massa, M., Ameri, G., Paolucci, R. (2011).
    Ground Motion Prediction Equations Derived from the Italian Strong Motion Database.
    Bulletin of Earthquake Engineering, 9(6), 1899-1920.
    DOI: 10.1007/s10518-011-9313-z
    OA copy: https://gfzpublic.gfz-potsdam.de/pubman/item/item_244399

Formula (Equations 1-3, PDF p. 6; Table 5 PGA_GeoH column, PDF p. 22):

    log10(PGA) = e1 + FD(R, M) + FM(M) + FS + Fsof

    FD(R, M) = (c1 + c2*(M - Mref)) * log10(sqrt(R^2 + h^2) / Rref)
               - c3 * (sqrt(R^2 + h^2) - Rref)

    FM(M) = b1*(M - Mh) + b2*(M - Mh)^2    for M <= Mh
    FM(M) = 0                                for M > Mh  [b3=0, fixed]

    FS   = sB*ec8_B + sC*ec8_C + sD*ec8_D + sE*ec8_E   (sA=0, reference class A)
    Fsof = fN*sof_NF + fR*sof_TF + fSS*sof_SS           (fU=0, reference class U)

where PGA is in cm/s^2, R is Joyner-Boore distance (or epicentral if Rjb unavailable,
typically for M < 5.5), and M is moment magnitude (Mw).

LAW_CONSTANTS — from Table 5, PGA_GeoH column, PDF p. 22 (Table 5 caption p. 21):
    e1   =  3.672       constant offset (log10 cm/s2)
    c1   = -1.940       geometric spreading slope
    c2   =  0.413       magnitude-dependent geometric spreading
    h    = 10.322       effective depth / pseudo-depth (km)
    c3   =  1.34e-4     anelastic attenuation coefficient (km^-1)
    b1   = -0.262       linear magnitude scaling (M <= Mh)
    b2   = -0.0707      quadratic magnitude scaling (M <= Mh)
    sB   =  0.162       EC8 site class B amplification
    sC   =  0.240       EC8 site class C amplification
    sD   =  0.105       EC8 site class D amplification
    sE   =  0.570       EC8 site class E amplification
    fN   = -0.0503      normal faulting correction (f1 in paper)
    fR   =  0.1050      reverse/thrust faulting correction (f2 in paper)
    fSS  = -0.0544      strike-slip faulting correction (f3 in paper)

OTHER_CONSTANTS — fixed structural constants (PDF p. 7, "After some trial regressions
and after Boore and Atkinson (2008), the following variables have been fixed:
Rref = 1 km; Mref = 5; Mh = 6.75; b3 = 0."):
    Mref =  5.0   reference magnitude (km)
    Mh   =  6.75  hinge magnitude (piecewise FM branch)
    Rref =  1.0   reference distance (km)

Type I — each station-event recording is an independent row. The GMPE has no
per-cluster fitted parameters; all coefficients in LAW_CONSTANTS are globally published
constants from Table 5. LOCAL_FITTABLE is empty; no fit() needed.

Column mapping (paper notation → released CSV columns, USED_INPUTS order):
    M      → Mw        (col 1)
    R/Rjb  → R_km      (col 2)
    depth  → depth_km  (col 3) — SR covariate only; formula uses fixed h=10.322
    CA     → ec8_A     (col 4) — dummy 0/1; reference class (sA=0, not used)
    CB     → ec8_B     (col 5) — dummy 0/1
    CC     → ec8_C     (col 6) — dummy 0/1
    CD     → ec8_D     (col 7) — dummy 0/1
    CE     → ec8_E     (col 8) — dummy 0/1
    EN     → sof_NF    (col 9)  — dummy 0/1 (Normal fault)
    ER     → sof_TF    (col 10) — dummy 0/1 (Reverse/Thrust fault)
    ESS    → sof_SS    (col 11) — dummy 0/1 (Strike-slip fault)

Note on RotD50 vs GeoH alignment: Bindi 2011 uses geometric mean of horizontal
components (GeoH); the ITA18 dataset provides RotD50. Boore (2010) establishes
RotD50 ≈ GeoH to within ~1-3% for PGA. The systematic offset is small relative to
total sigma = 0.337 log10 units. Benchmark RMSE ~0.37 log10 units is consistent.

Note on Lanzano 2019 (BSSA 109(2):525-540, DOI 10.1785/0120180210) omission:
The ITA18 GMPE calibrated on this exact dataset is NOT shipped for two reasons:
(1) SCHEMA mismatch — ITA18 uses a CONTINUOUS Vs30 site term
(FS = k*log10(min(Vs30,1500)/800); OpenQuake LanzanoEtAl2019
REQUIRES_SITES_PARAMETERS={'vs30'}), incompatible with this task's EC8 site-CLASS
dummy schema, so it is not implementable on the released columns without adding a
Vs30 column; (2) the BSSA PDF is paywalled (HTTP 403, GeoScienceWorld). The shipped
baselines are Bindi 2011 (this file) and Akkar-Bommer 2010. Adding ITA18 would
require a schema change (release Vs30, re-prep, continuous-Vs30 site term). See VERDICT.md.
"""

import numpy as np

USED_INPUTS = [
    "Mw", "R_km",
    "depth_km",
    "ec8_A", "ec8_B", "ec8_C", "ec8_D", "ec8_E",
    "sof_NF", "sof_TF", "sof_SS",
]
PAPER_REF = "summary_formula_bindi_2011.md"
EQUATION_LOC = "Eqs. 1-3, PDF p. 6; Table 5 (PGA_GeoH column), PDF p. 22"

# All values from Table 5, PGA_GeoH column, PDF p. 22
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "e1":  3.672,      # Table 5 PGA_GeoH, PDF p. 22
    "c1": -1.940,      # Table 5 PGA_GeoH, PDF p. 22
    "c2":  0.413,      # Table 5 PGA_GeoH, PDF p. 22
    "h":  10.322,      # Table 5 PGA_GeoH, PDF p. 22 (km)
    "c3":  1.34e-4,    # Table 5 PGA_GeoH, PDF p. 22 (km^-1)
    "b1": -0.262,      # Table 5 PGA_GeoH, PDF p. 22
    "b2": -0.0707,     # Table 5 PGA_GeoH, PDF p. 22
    "sB":  0.162,      # Table 5 PGA_GeoH, PDF p. 22
    "sC":  0.240,      # Table 5 PGA_GeoH, PDF p. 22
    "sD":  0.105,      # Table 5 PGA_GeoH, PDF p. 22
    "sE":  0.570,      # Table 5 PGA_GeoH, PDF p. 22
    "fN": -0.0503,     # Table 5 PGA_GeoH f1, PDF p. 22
    "fR":  0.1050,     # Table 5 PGA_GeoH f2, PDF p. 22
    "fSS": -0.0544,    # Table 5 PGA_GeoH f3, PDF p. 22
}

# Fixed structural anchors (PDF p. 7; Boore & Atkinson 2008 precedent)
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "Mref": 5.0,    # reference magnitude; fixed in regression, PDF p. 7
    "Mh":   6.75,   # hinge magnitude; fixed, PDF p. 7
    "Rref": 1.0,    # reference distance (km); fixed, PDF p. 7
    # sA = 0 (EC8 class A reference), fU = 0 (unknown SOF reference): structural zeros
}

LOCAL_FITTABLE = {}  # Type I — no per-cluster parameters


def predict(
    X: np.ndarray,
    e1: float, c1: float, c2: float, h: float, c3: float,
    b1: float, b2: float,
    sB: float, sC: float, sD: float, sE: float,
    fN: float, fR: float, fSS: float,
) -> np.ndarray:
    """Predict log10(PGA [cm/s^2]) for each station-event row.

    X.shape = (n, 11) — columns in USED_INPUTS order:
        0: Mw, 1: R_km, 2: depth_km (unused by formula, SR covariate),
        3: ec8_A, 4: ec8_B, 5: ec8_C, 6: ec8_D, 7: ec8_E,
        8: sof_NF, 9: sof_TF, 10: sof_SS
    """
    Mref = OTHER_CONSTANTS["Mref"]
    Mh   = OTHER_CONSTANTS["Mh"]
    Rref = OTHER_CONSTANTS["Rref"]

    M    = X[:, 0]
    R    = X[:, 1]
    # depth_km at X[:, 2] is a SR covariate only; formula uses fixed h
    ec8_B_ = X[:, 4]
    ec8_C_ = X[:, 5]
    ec8_D_ = X[:, 6]
    ec8_E_ = X[:, 7]
    sof_NF_ = X[:, 8]
    sof_TF_ = X[:, 9]
    sof_SS_ = X[:, 10]

    R_eff = np.sqrt(R**2 + h**2)

    # Distance scaling (Eq. 2, PDF p. 6)
    FD = (c1 + c2 * (M - Mref)) * np.log10(R_eff / Rref) - c3 * (R_eff - Rref)

    # Magnitude scaling (Eq. 3, PDF p. 6) — piecewise at hinge
    dM = M - Mh
    FM_low  = b1 * dM + b2 * dM**2
    FM_high = np.zeros_like(M)        # b3 = 0
    FM = np.where(M <= Mh, FM_low, FM_high)

    # Site amplification (PDF p. 7) — EC8 class A is reference (sA = 0)
    FS = sB * ec8_B_ + sC * ec8_C_ + sD * ec8_D_ + sE * ec8_E_

    # Style-of-faulting (PDF p. 7) — Unknown (U) is reference (fU = 0)
    Fsof = fN * sof_NF_ + fR * sof_TF_ + fSS * sof_SS_

    return e1 + FD + FM + FS + Fsof
