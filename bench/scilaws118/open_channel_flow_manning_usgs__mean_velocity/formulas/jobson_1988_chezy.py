"""Chezy's equation — mean flow velocity in open channels.

Jobson, H. E., and Froehlich, D. C. (1988). *Basic Hydraulic Principles of
Open-Channel Flow.* U.S. Geological Survey Open-File Report 88-707, Reston,
Virginia. URL: https://pubs.usgs.gov/of/1988/0707/report.pdf

Lesson 7 (PDF pp. 59-62) presents the Chezy resistance equation as the
predecessor to Manning's formula. Eq. 7-6 (PDF p. 60, labeled "Chezy
equation") is:

    V = C * sqrt(R * S_f)

where V is the cross-section mean velocity, R is the hydraulic radius (m),
S_f is the friction slope (dimensionless), and C is the Chezy resistance
coefficient. This is the simplified form where C absorbs any sqrt(g) factor
(the dimensional form sometimes written as V = ~C * sqrt(g*R*S_f) uses a
dimensionless ~C; here we use the common hydraulic engineering form with
dimensional C). Structural exponent on R is 1/2 (vs Manning's 2/3), making
this a structurally distinct baseline.

The relationship between Chezy C and Manning n is given by eq. 7-7
(PDF p. 61): C = (1.49/n) * R^(1/6) in US customary, confirming that C
and n are both per-cluster fit parameters that encode the same roughness
physics but via a different functional form.

IFMHA dataset (Erfani et al. 2024) supplies measurements in US customary
units; PROVENANCE.md documents the SI conversion applied by prep_data.py:
  - chan_area (ft^2)   -> A_m2 (m^2)  via x 0.09290304
  - chan_width (ft)    -> W_m  (m)     via x 0.3048
  - R_m = A_m2 / W_m  (hydraulic depth, approximation of R for wide channels)
  - SLOPE (dimensionless ft/ft) is unchanged

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The Chezy equation's structural exponent 1/2 on R and 1/2 on S_f
are universal factors of the Chezy-Manning hydraulic derivation, kept as
literals. C is a per-cluster fittable (not a paper-fixed constant).

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The exponent 1/2 is kept as a literal in the formula body.

LOCAL_FITTABLE — per-cluster, fitted by fit() via closed-form OLS
-----------------------------------------------------------------------
- C : Chezy resistance coefficient (m^(1/2)/s). Varies by channel geometry,
      roughness, and flow regime; fit per USGS gauge station (COMID).
      Typical range: 15-100 m^(1/2)/s for natural streams (Manning n
      0.01-0.10 and hydraulic radius 0.1-10 m via eq. 7-7, PDF p. 61).

init = None: fit() uses closed-form OLS (linearised as V = C * h, where
h = sqrt(R*S_f), so C = sum(h*V) / sum(h^2) — single-parameter zero-
intercept linear regression).
"""

import numpy as np

USED_INPUTS = ["R_m", "SLOPE"]
PAPER_REF = "summary_formula_jobson_1988.md"
EQUATION_LOC = (
    "Jobson & Froehlich (1988) eq. 7-6, PDF p. 60 — V = C*sqrt(R*S_f); "
    "Chezy resistance coefficient C is per-channel fittable. "
    "Structural exponent 1/2 on R distinguishes this from Manning (2/3)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "C": {"init": None},
}


def _hydraulic_term(R_m, SLOPE):
    """h = sqrt(R * S_f) — the Chezy conveyance term."""
    R_safe = np.clip(R_m, 0.0, None)
    S_safe = np.clip(SLOPE, 0.0, None)
    return np.sqrt(R_safe * S_safe)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form zero-intercept OLS for Chezy C.

    Linearise: V = C * h  where h = sqrt(R * S_f).
    Closed-form: C = sum(h_i * V_i) / sum(h_i^2)

    This is the exact least-squares solution under the constraint of a
    zero-intercept linear model — fast, deterministic, no multi-start needed.
    """
    R_m   = np.asarray(X_fit[:, 0], dtype=float)
    SLOPE = np.asarray(X_fit[:, 1], dtype=float)
    y     = np.asarray(y_fit, dtype=float)

    h = _hydraulic_term(R_m, SLOPE)

    valid = (y > 0) & (h > 0)
    if valid.sum() >= 1:
        h_v = h[valid]
        y_v = y[valid]
        C_fit = float(np.dot(h_v, y_v) / np.dot(h_v, h_v))
    else:
        C_fit = 30.0   # mid-range default

    # Clip to physically plausible range.
    # Chezy C for natural channels: ~10-100 m^(1/2)/s (Jobson 1988 eq. 7-7
    # rearranged: C = R^(1/6)/n in SI with n in [0.008, 0.5]).
    C_fit = float(np.clip(C_fit, 5.0, 300.0))
    return {"C": C_fit}


def predict(X: np.ndarray, C: float) -> np.ndarray:
    """Chezy equation: V = C * sqrt(R * S_f).

    X: (n_rows, 2) — columns [R_m, SLOPE].
    Returns V_mps (m/s), clipped to >= 0.
    """
    R_m   = np.asarray(X[:, 0], dtype=float)
    SLOPE = np.asarray(X[:, 1], dtype=float)
    h = _hydraulic_term(R_m, SLOPE)
    return np.clip(C * h, 0.0, None)
