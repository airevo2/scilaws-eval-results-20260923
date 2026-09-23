"""Heckman, Lochner & Todd (2005) — Mincer earnings function.

Heckman, J. J., Lochner, L. J., & Todd, P. E. (2005). Earnings functions,
rates of return and treatment effects: The Mincer equation and beyond.
NBER Working Paper No. 11544.  PDF: reference/heckman_2003.pdf.
Formula: Equation (1), PDF page 8, Section 2 "The Theoretical Foundations
of Mincer's Earnings Regression."

Functional form (Eq. 1, PDF p. 8):

    ln[Y(s, x)] = alpha + rho_s * s + beta_0 * x + beta_1 * x^2 + epsilon

where:
  Y(s, x)   = wage or earnings at schooling level s and experience x
  s          = years of schooling
  x          = potential work experience (years)
  alpha      = intercept (log-wage at zero schooling and zero experience)
  rho_s      = rate-of-return to schooling ("Mincer coefficient")
  beta_0     = linear experience coefficient (wage growth in early career)
  beta_1     = quadratic experience coefficient (expected < 0: declining returns)
  epsilon    = mean-zero residual; E(epsilon | s, x) = 0

LAW_CONSTANTS — the Mincer equation's defining coefficients
-----------------------------------------------------------------------
ln[Y] = alpha + rho_s*s + beta_0*x + beta_1*x^2 is the Mincer earnings
function. alpha (intercept), rho_s (schooling return), beta_0 and beta_1 (the
concave experience profile) are its defining coefficients — the formula's
characteristic parameters and the SR discovery target. Per the benchmark's
LAW role they are fit on the representative train split ("fit on train, check
correctness"); they reproduce exactly from OLS on train.csv (53,333 workers,
ACS PUMS 2019; verified 2026-05-30). They are NEVER exposed in priors.

  alpha   = 8.888690  (intercept; log-wage at s=0, x=0)
  rho_s   = 0.111306  (schooling return; ~11% per year of schooling)
  beta_0  = 0.028646  (linear-experience coefficient)
  beta_1  = -0.000379 (quadratic; peak at exp* = beta_0 / (-2*beta_1) ≈ 38 yrs)

Historical-filing note (corrected 2026-05-30, field-classification re-audit):
earlier versions placed these four in OTHER_CONSTANTS citing "data_spec §0.2.4
— values fit from data are not LAW". That rule is superseded: everything in
LAW was fit by someone; the axis is "defining coefficient (LAW) vs
universal/derived/structural given (OTHER)", not "fitted vs not". These four
are defining coefficients -> LAW. Comparable published cohort estimates
(Heckman et al. 2005 Table 2, PDF p. 156: 1990 whites rho_s=0.129, beta_0=0.130,
beta_1=-0.0023; 1960 rho_s=0.115) are retained only as priors *distractors*.

OTHER_CONSTANTS — {} (empty)
-----------------------------------------------------------------------
The form consumes no universal/derived/structural given constant; the x^2
exponent is structural and stays inline in predict(). OTHER_CONSTANTS = {}.

Type designation: TYPE I — the Mincer equation is a global cross-section
model; all coefficients are globally fitted once across all workers;
no per-cluster refit. LOCAL_FITTABLE = {}.

Column mapping (paper notation → released CSV):
  ln[Y(s, x)] → log_wage    (col 0, natural log of adjusted annual wage)
  s            → SCHL_years (col 1, years of schooling 0-22)
  x            → exp        (col 2, potential experience 0-58)
  x^2          → computed internally (NOT a CSV column — central SR challenge)

The quadratic-in-experience term is the central SR discovery target. On the
representative test split this quadratic baseline (test RMSE 0.6674, R^2 0.189)
beats both the train-mean naive predictor (0.7413) and a linear-in-experience
model (0.6695) — so recovering the x^2 structure is correctly rewarded.
"""

import numpy as np

USED_INPUTS = ["SCHL_years", "exp"]
PAPER_REF   = "summary_formula_heckman_2003.md"
EQUATION_LOC = "Heckman, Lochner & Todd (2005) NBER WP 11544, Eq. (1), PDF p. 8"

# === LAW_CONSTANTS — the Mincer equation's defining coefficients (fit on train) ===
# Reproduce EXACTLY from OLS on train.csv (verified 2026-05-30). The SR discovery
# target; never exposed in priors. Reclassified OTHER->LAW 2026-05-30 — the old
# "fit-from-data => not LAW" rule (data_spec §0.2.4) is superseded; see docstring.
LAW_CONSTANTS = {
    "alpha":  8.888690024317656,        # intercept; log-wage at s=0, x=0
    "rho_s":  0.11130553854535347,      # schooling return (~11%/yr)
    "beta_0": 0.02864629358148585,      # linear-experience coefficient
    "beta_1": -0.00037876090477542945,  # quadratic-experience coefficient (< 0: concave)
}

# === OTHER_CONSTANTS — none (no given constant; x^2 exponent is structural, inline) ===
OTHER_CONSTANTS = {}

LOCAL_FITTABLE = {}    # Type I — no per-cluster refit


def predict(X: np.ndarray, alpha: float, rho_s: float,
            beta_0: float, beta_1: float) -> np.ndarray:
    """Predict log_wage from the Mincer quadratic-in-experience earnings function.

    ln Y = alpha + rho_s * s + beta_0 * x + beta_1 * x^2

    X: (n, 2) — columns [SCHL_years, exp] in USED_INPUTS order.
    Coefficients are the LAW_CONSTANTS (Mincer defining coefficients); the harness
    passes them as predict(X, **LAW_CONSTANTS).
    Note: x^2 is computed internally — intentionally absent from the released CSV
    (its discovery is the central SR challenge).
    """
    X = np.asarray(X, dtype=np.float64)
    s = X[:, 0]    # SCHL_years
    x = X[:, 1]    # exp (potential experience)
    return alpha + rho_s * s + beta_0 * x + beta_1 * x ** 2
