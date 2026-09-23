"""Burton, McPherron & Russell (1975) ring-current model.

Citation: R.K. Burton, R.L. McPherron & C.T. Russell, "An empirical
relationship between interplanetary conditions and Dst," J. Geophys. Res.
80, 4204-4214 (1975). DOI: 10.1029/JA080i031p04204.

Formula (Eq. 4, PDF p. 4206; final prescription coefficients, PDF p. 4208):
    dDst_0/dt = F(E_y) - a_hr * Dst_0

    Dst_0 = Dst - b * sqrt(P_dyn) + c   (pressure-corrected Dst)

    F(E_y) = 0                       if Ey <  ey_thresh
    F(E_y) = d_hr * (Ey - ey_thresh)  if Ey >= ey_thresh

LAW_CONSTANTS (a_hr, c, d_hr, ey_thresh from Burton 1975 PDF p. 4207-4208
final-prescription table; b from Markidis 2025 Table 3 BMR row, PDF p. 9 —
modern OMNI-refit nPa convention; see Caveats):
    a_hr      = 0.1296   hr^-1       ring-current decay constant
                                     (3.6e-5 s^-1 × 3600 s/hr; Burton 1975 PDF p. 4208)
    b         = 0.20     nT/sqrt(nPa) magnetopause pressure coefficient
                                     (Markidis 2025 Table 3 BMR row, PDF p. 9 —
                                      modern OMNI refit; Burton 1975's original
                                      b=0.20 was in nT/(eV/cm^3)^(1/2) units,
                                      which converts to ~0.50 nT/sqrt(nPa), NOT
                                      to 0.20. The two numerical values 0.20 are
                                      a coincidence across different unit systems;
                                      the 0.20 nT/sqrt(nPa) used here is the
                                      modern community convention. See Caveats.)
    c         = 20.0     nT           pressure-correction offset (Burton 1975 PDF p. 4208)
    d_hr      = -5.4     nT/(hr*mV/m) injection rate coefficient
                                     (-1.5e-3 s^-1 × 3600; Burton 1975 PDF p. 4208)
    ey_thresh = 0.5      mV/m         injection threshold (Burton 1975 PDF p. 4207)

OTHER_CONSTANTS: none (all scalars from the published parameter table).

Type: Type I. All constants are globally fitted across the seven storm
sequences in Burton et al. 1975 (PDF p. 4208). No per-storm parameters.

Column mapping:
    paper E_y -> Ey (mV/m in released CSV, same convention)
    paper P   -> P_dyn (nPa; b restated to match nPa units per Markidis 2025
                 Table 3 BMR row, PDF p. 9, which uses b=0.2 nT/sqrt(nPa))
    paper Dst -> Dst (nT)

Caveats:
    Burton 1975 originally fit b=0.20 in units of nT/(eV/cm^3)^(1/2) with
    P = n V^2 * 10^-2 eV/cm^3 (PDF p. 4208 final-prescription block; the
    earlier text on PDF p. 4207 also gives "from dDst/d(P)^(1/2) is b = 0.2
    with a standard deviation of 0.1").
    Direct unit conversion: 1 eV/cm^3 = 0.1602 nPa, so the original b*sqrt(P_eV)
    equals b_nPa*sqrt(P_nPa) when b_nPa = 0.20 / sqrt(0.1602) ~ 0.50 nT/sqrt(nPa).
    NOT 0.20. Burton 1975's b in nPa convention would be ~0.50.
    However, the modern OMNI-refit literature (O'Brien 2000; Markidis 2025
    Table 3 BMR row, PDF p. 9: "dDst/dt = -0.13 D_st - 0.2 * sqrt(P_dyn) + ...")
    uses b = 0.20 nT/sqrt(nPa) directly — the same numerical value but with
    P_dyn in nPa rather than eV/cm^3. Because the pressure-correction term
    contributes only a few nT to Dst and the refit is performed against OMNI2
    in nPa, the 0.20-nPa convention is the accepted community usage.
    We follow that consensus; b=0.20 here is Markidis 2025's value, not a
    unit restatement of Burton 1975 (which would be ~0.50 in nPa).
"""

import numpy as np

USED_INPUTS = ["Dst", "Ey", "P_dyn"]
PAPER_REF = "summary_formula_burton_1975.md"
EQUATION_LOC = "Burton 1975 Eq. 4 + final-prescription coefficients, PDF p. 4206-4208"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a_hr":      0.1296,   # hr^-1          — PDF p. 4208 (3.6e-5 s^-1 × 3600)
    "b":         0.20,     # nT/sqrt(nPa)   — Markidis 2025 Table 3 BMR row, PDF p. 9 (see Caveats)
    "c":        20.0,      # nT             — PDF p. 4208
    "d_hr":     -5.4,      # nT/(hr*mV/m)  — PDF p. 4208 (-1.5e-3 s^-1 × 3600)
    "ey_thresh": 0.5,      # mV/m           — PDF p. 4207
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}

LOCAL_FITTABLE = {}


def predict(X: np.ndarray, a_hr: float, b: float, c: float,
            d_hr: float, ey_thresh: float) -> np.ndarray:
    """Predict dDst_0/dt from (Dst, Ey, P_dyn) using the BMR model.

    X.shape = (n, 3); columns in USED_INPUTS order: Dst, Ey, P_dyn.
    LAW_CONSTANTS arrive as named params via predict(X, **LAW_CONSTANTS).
    """
    Dst   = np.asarray(X[:, 0], dtype=float)
    Ey    = np.asarray(X[:, 1], dtype=float)
    P_dyn = np.asarray(X[:, 2], dtype=float)

    Dst_0 = Dst - b * np.sqrt(np.maximum(P_dyn, 0.0)) + c
    F     = np.where(Ey >= ey_thresh, d_hr * (Ey - ey_thresh), 0.0)
    return F - a_hr * Dst_0
