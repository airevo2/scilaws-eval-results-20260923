"""O'Brien & McPherron (2000) AK2 ring-current model.

Citation: T.P. O'Brien & R.L. McPherron, "Forecasting the Ring Current
Index Dst in Real Time," J. Atmos. Sol.-Terr. Phys. 62, 1295-1299 (2000).
DOI: 10.1016/S1364-6826(00)00072-9.

Formula (Eq. 1, PDF p. 1 + Table 1 AK2 row, PDF p. 2):
    dDst*/dt = Q(t) - Dst*/tau

    Dst* = Dst - b*sqrt(P_dyn) + c           (Eq. 3, PDF p. 2)

    Q  = 0                          if VBs <= vbs_thresh
    Q  = q_coef * (VBs - vbs_thresh)  if VBs >  vbs_thresh
    tau = tau_amp * exp(tau_num / (tau_off + VBs))

    VBs = max(Ey, 0)   [Eq. 2, PDF p. 1 — southward-only injection]

LAW_CONSTANTS (Table 1, AK2 row, PDF p. 2):
    q_coef     = -4.4  nT/(hr*mV/m)   injection coupling coefficient
    vbs_thresh = 0.5   mV/m            electric-field injection threshold
    tau_amp    = 2.4   hr              decay-time amplitude
    tau_num    = 9.74  (mV/m units)   decay-time exponential numerator
    tau_off    = 4.69  (mV/m units)   decay-time exponential offset
    b          = 7.26  nT/sqrt(nPa)   magnetopause current pressure coefficient
    c          = 11.0  nT              pressure-correction offset

OTHER_CONSTANTS: none (all scalars come from the paper's fitted coefficient set).

Type: Type I. All seven AK2 constants are globally fitted across OMNI
1964-1996 (O'Brien 2000 §3, PDF p. 2). No per-storm parameters.

Column mapping:
    paper VBs  -> max(Ey, 0)   [Ey = -V_sw*Bz_GSM*1e-3 in OMNI2; same sign convention]
    paper P    -> P_dyn (nPa)
    paper Dst  -> Dst (nT)

Caveats:
    The target column dDst_dt is the time derivative of *raw* Dst, while
    the AK2 ODE predicts dDst*/dt (pressure-corrected). The formula is
    applied to raw Dst inputs with the pressure correction computed
    internally (standard operational practice — Markidis 2025, §3, PDF p. 7
    uses the same convention).
"""

import numpy as np

USED_INPUTS = ["Dst", "Ey", "P_dyn"]
PAPER_REF = "summary_formula_obrien_2000.md"
EQUATION_LOC = "O'Brien & McPherron 2000, Eq. 1 + Table 1 AK2 row, PDF p. 1-2"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "q_coef":     -4.4,   # nT/(hr*mV/m)  — Table 1 AK2, PDF p. 2
    "vbs_thresh":  0.5,   # mV/m          — Table 1 AK2, PDF p. 2
    "tau_amp":     2.4,   # hr            — Table 1 AK2, PDF p. 2
    "tau_num":     9.74,  # mV/m units    — Table 1 AK2, PDF p. 2
    "tau_off":     4.69,  # mV/m units    — Table 1 AK2, PDF p. 2
    "b":           7.26,  # nT/sqrt(nPa)  — Table 1 AK2, PDF p. 2
    "c":          11.0,   # nT            — Table 1 AK2, PDF p. 2
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}

LOCAL_FITTABLE = {}


def predict(X: np.ndarray, q_coef: float, vbs_thresh: float, tau_amp: float,
            tau_num: float, tau_off: float, b: float, c: float) -> np.ndarray:
    """Predict dDst*/dt from (Dst, Ey, P_dyn) using the AK2 model.

    X.shape = (n, 3); columns in USED_INPUTS order: Dst, Ey, P_dyn.
    LAW_CONSTANTS arrive as named params via predict(X, **LAW_CONSTANTS).
    """
    Dst   = np.asarray(X[:, 0], dtype=float)
    Ey    = np.asarray(X[:, 1], dtype=float)
    P_dyn = np.asarray(X[:, 2], dtype=float)

    VBs     = np.maximum(Ey, 0.0)
    Q       = np.where(VBs > vbs_thresh, q_coef * (VBs - vbs_thresh), 0.0)
    tau     = tau_amp * np.exp(tau_num / (tau_off + VBs))
    Dst_star = Dst - b * np.sqrt(np.maximum(P_dyn, 0.0)) + c
    return Q - Dst_star / tau
