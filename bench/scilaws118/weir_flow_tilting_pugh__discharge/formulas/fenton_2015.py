"""Fenton (2015) Kindsvater-Carter / Rehbock C_D formula for weir discharge.

Fenton, J. D. (2015). Calculating flow over rectangular sharp-edged weirs.
*Alternative Hydraulics Paper 6*. Open access:
http://johndfenton.com/Papers/Calculating-flow-over-rectangular-sharp-edged-weirs.pdf
(No DOI; technical report.)

---

**Core formula — Kindsvater-Carter dimensional form (Fenton Eq. 14):**

    Q = (2/3) * C_D * sqrt(2*g) * b * H^(3/2)

where H = h + k_H is the effective head (h is the measured head above the
weir crest, k_H = 0.001 m is the fluid-property head correction from Eq. 17).

**C_D polynomial approximation (Fenton Eq. 24) fitted to K&C Figure 10:**

    C_D = 0.589 - 0.008*(H/P) + (b/B)^2 * (0.013 + 0.083*(H/P))

For a full-span weir (b = B, i.e., b/B = 1):

    C_D = (0.589 + 0.013) + (-0.008 + 0.083)*(H/P)
        = 0.602 + 0.075*(H/P)

consistent with Rehbock's C_D = 0.605 + 0.08*(H/P) (Fenton §6.3 after Eq. 24).

---

**Application to the tilting-weir dataset (Pugh et al. 2024, JHE):**

The Pugh 2024 tilting-weir dataset has inclination angles theta = 25.7° to 90°.
Fenton (2015) derives C_D for a vertical sharp-crested weir only (no theta
dependence).  To apply the formula as a baseline to tilted configurations, a
LOCAL_FITTABLE additive correction `delta_C` is introduced:

    C_D_eff = C_D_Fenton + delta_C

where delta_C absorbs the systematic tilt-induced shift in the effective
discharge coefficient.  For a vertical weir (theta = 90°), the best-fit
delta_C should approach zero.

The weir crest width b is assumed equal to the channel width (full-span weir),
so b/B = 1 throughout.  This is consistent with the experimental setup
(Pugh 2024 Table 1: narrow rectangular flumes with full-width weir plates).

---

LAW_CONSTANTS — paper-published, frozen (Fenton 2015 Eq. 24)
---------------------------------------
The four coefficients of the C_D polynomial, which Fenton (2015) obtained as a
single "least-squares mathematical approximation to the lines on figure 10 of
Kindsvater and Carter" (PDF p. 11, after Eq. 24). They are the paper's fitted
cross-configuration claim and are held FIXED across all weir configurations
(passed to both fit() and predict(), never re-fit per cluster) → LAW per the
Type-II invariant-AND-fitted rule.
- C0     = 0.589   : C_D intercept at H/P=0, b/B=0 (Eq. 24).
- C_HP   = -0.008  : C_D slope in H/P at b/B=0 (Eq. 24).
- C_bB2  = 0.013   : (b/B)^2 intercept correction (Eq. 24).
- C_bBHP = 0.083   : (b/B)^2 slope correction in H/P (Eq. 24).

OTHER_CONSTANTS — universal / structural / given factors
------------------------------------------------
- g = 9.81 m s⁻²: Fenton (2015) uses 9.806 ≈ 9.81 for local gravity (Eq. 21,
  PDF p. 10); 9.81 m s⁻² is consistent with Pugh 2025's stated value (§2).
- factor_2_3_sqrt2g = (2/3)*sqrt(2g): derived from g (kind-b OTHER).
- k_H = 0.001 m : head correction for fluid-property effects. NOT a fitted
  coefficient — Fenton §6.2/Eq. 17 reports that K&C "found, for all their
  tests, that a constant value of kH = 0.001 m was 'adequate to compensate for
  fluid-property effects related to the head'." An assumed/adopted GIVEN the
  formula merely consumes → OTHER (invariant-but-given; cf. tumor_growth V0).
- Factor 2/3 and sqrt(2): from Poleni/K&C derivation (Fenton Eq. 14);
  integration constant, not a tuned coefficient (structural, inline).

LOCAL_FITTABLE — per-cluster, fitted by fit() via OLS
------------------------------------------------------
- delta_C : additive correction to C_D [dimensionless] for theta effects.
  init = None: fit() derives delta_C from the residual of the Fenton prediction.

init = None: fit() builds a data-derived start (closed-form OLS on the residual).
"""

import numpy as np

USED_INPUTS = ["h_m", "p_m", "b_m"]
PAPER_REF = "summary_formula_fenton_2015.md"
EQUATION_LOC = (
    "Fenton 2015 Eq. 14: Q = (2/3)*C_D*sqrt(2g)*b*H^(3/2), PDF p. 9; "
    "Eq. 24: C_D = 0.589 - 0.008*(H/P) + (b/B)^2*(0.013 + 0.083*(H/P)), PDF p. 11; "
    "Eq. 17: k_H = 0.001 m, PDF p. 8; "
    "Rehbock (1929) full-width limit C_D = 0.605 + 0.08*(H/P) confirmed in §6.3."
)

# ── LAW_CONSTANTS ─────────────────────────────────────────────────────────────
# The four coefficients of the C_D polynomial from Fenton (2015) Eq. 24, PDF
# p. 11 — a single least-squares fit to K&C Figure 10, held fixed across all
# weir configurations (invariant-AND-fitted → LAW).
LAW_CONSTANTS = {
    "C0":     0.589,    # C_D intercept at H/P=0 and b/B=0 (Eq. 24)
    "C_HP":  -0.008,    # C_D slope in H/P at b/B=0 (Eq. 24)
    "C_bB2":  0.013,    # (b/B)^2 intercept correction (Eq. 24)
    "C_bBHP": 0.083,    # (b/B)^2 slope correction in H/P (Eq. 24)
}

# ── OTHER_CONSTANTS ───────────────────────────────────────────────────────────
# g: Fenton Eq. 21 uses g = 9.806 m s⁻² (Atlanta latitude); Pugh 2025 §2 uses
# 9.81 m s⁻².  We adopt 9.81 m s⁻² for consistency with the primary source.
# k_H: Fenton §6.2 / Eq. 17 — K&C found a *constant* value 0.001 m "adequate to
# compensate for fluid-property effects related to the head"; an assumed/adopted
# given the formula consumes (NOT fitted), so OTHER (invariant-but-given).
OTHER_CONSTANTS = {"g": 9.81, "k_H": 0.001}          # g [m s⁻²]; k_H [m] (Eq. 17 given)
_G = OTHER_CONSTANTS["g"]                             # alias of the boxed gravity constant
_FACTOR = 2.0 / 3.0 * np.sqrt(2.0 * _G)              # (2/3)*sqrt(2g) [m^0.5 s^-1]
OTHER_CONSTANTS["factor_2_3_sqrt2g"] = _FACTOR        # derived, consumed by predict()

# ── LOCAL_FITTABLE ────────────────────────────────────────────────────────────
LOCAL_FITTABLE = {
    "delta_C": {"init": None},  # additive C_D correction for theta effects
}


def _C_D_fenton(H, P, bB=1.0):
    """C_D from Fenton (2015) Eq. 24 with b/B = bB (default 1.0, full-width).

    H  : effective head [m] = h + k_H
    P  : weir height [m]
    bB : b/B ratio (dimensionless); default 1.0 (full-span weir)
    """
    return (LAW_CONSTANTS["C0"]
            + LAW_CONSTANTS["C_HP"] * (H / P)
            + bB ** 2 * (LAW_CONSTANTS["C_bB2"]
                         + LAW_CONSTANTS["C_bBHP"] * (H / P)))


def _Q(h, p, b, delta_C):
    """Q = (2/3) * (C_D_Fenton + delta_C) * sqrt(2g) * b * H^(3/2)  [m³ s⁻¹].

    Assumes full-span weir (b/B = 1).  h and p are measured values [m].
    """
    H = h + OTHER_CONSTANTS["k_H"]   # effective head (Eq. 17 given)
    C_D_eff = _C_D_fenton(H, p) + delta_C
    return (2.0 / 3.0) * C_D_eff * np.sqrt(2.0 * _G) * b * H ** 1.5


def fit(X_fit: np.ndarray, y_fit: np.ndarray,
        C0: float, C_HP: float, C_bB2: float, C_bBHP: float) -> dict:
    """Closed-form OLS estimate of delta_C from residual of the Fenton prediction.

    The model Q = (2/3)*(C_D_Fenton + delta_C)*sqrt(2g)*b*H^(3/2) is linear
    in delta_C.  Defining the Fenton-predicted discharge as Q_F (with delta_C=0)
    and the scale factor S = (2/3)*sqrt(2g)*b*H^(3/2):

        Q_obs = Q_F + delta_C * S

    OLS on {Q_obs - Q_F} / S gives delta_C exactly.  Fallback: delta_C = 0
    (pure Fenton Eq. 24 with no tilt correction).

    The four C_D LAW constants are passed by the harness as keyword args
    (**LAW_CONSTANTS); k_H is a given read from OTHER_CONSTANTS.
    """
    h     = np.asarray(X_fit[:, 0], dtype=float)
    p     = np.asarray(X_fit[:, 1], dtype=float)
    b     = np.asarray(X_fit[:, 2], dtype=float)
    Q_obs = np.asarray(y_fit, dtype=float)

    H = h + OTHER_CONSTANTS["k_H"]
    S = (2.0 / 3.0) * np.sqrt(2.0 * _G) * b * H ** 1.5
    # Fenton prediction with delta_C = 0, using passed law constants
    CD_f = C0 + C_HP * (H / p) + 1.0 ** 2 * (C_bB2 + C_bBHP * (H / p))  # b/B=1
    Q_F  = CD_f * S

    good = (S > 0) & np.isfinite(S) & np.isfinite(Q_obs)
    if good.sum() < 1:
        return {"delta_C": 0.0}

    # OLS: delta_C = mean((Q_obs - Q_F) / S)
    residuals = (Q_obs[good] - Q_F[good]) / S[good]
    delta_C = float(np.mean(residuals))
    if not np.isfinite(delta_C):
        delta_C = 0.0
    return {"delta_C": delta_C}


def predict(X: np.ndarray,
            C0: float, C_HP: float, C_bB2: float, C_bBHP: float,
            delta_C: float) -> np.ndarray:
    """Q = (2/3)*(C_D_Fenton(H/P) + delta_C)*sqrt(2g)*b*H^(3/2)  [m³ s⁻¹].

    X: (n, 3) — columns [h_m, p_m, b_m].
    C_D_Fenton computed with b/B = 1 (full-span weir assumption).
    The four C_D LAW constants and the local delta_C are passed by the harness
    as keyword args (**LAW_CONSTANTS, **local); k_H is read from OTHER_CONSTANTS.
    """
    h = np.asarray(X[:, 0], dtype=float)
    p = np.asarray(X[:, 1], dtype=float)
    b = np.asarray(X[:, 2], dtype=float)
    H = h + OTHER_CONSTANTS["k_H"]
    C_D_eff = C0 + C_HP * (H / p) + 1.0 ** 2 * (C_bB2 + C_bBHP * (H / p)) + delta_C
    return (2.0 / 3.0) * C_D_eff * np.sqrt(2.0 * _G) * b * H ** 1.5
