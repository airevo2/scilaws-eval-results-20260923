"""Murphy & Welch (1990) — quartic-in-experience Mincer earnings function.

Murphy, K. M. and Welch, F. (1990). Empirical age-earnings profiles.
Journal of Labor Economics 8(2), 202-229.

Murphy & Welch (1990) argued that quartic polynomial terms in experience are
"empirically important for fitting the earnings curve" (Heckman, Lochner & Todd
2005, NBER WP 11544, text at PDF p. 23, heckman_2003.txt line 976-977).

The quartic-in-experience form (Murphy & Welch 1990, confirmed via Heckman et al.
2005 synthesis) extends the canonical Mincer equation:

    ln[Y(s, x)] = alpha + rho_s * s
                        + beta_0 * x
                        + beta_1 * x^2
                        + beta_2 * x^3
                        + beta_3 * x^4
                        + epsilon

The higher-order x^3 and x^4 terms capture the flattening of the earnings
profile across the career more flexibly than the quadratic alone.

LAW_CONSTANTS — the quartic Mincer's defining coefficients
-----------------------------------------------------------------------
The quartic-in-experience form ln[Y] = alpha + rho_s*s + beta_0*x + beta_1*x^2
+ beta_2*x^3 + beta_3*x^4 is the Murphy-Welch (1990) scientific claim; the six
coefficients are its defining parameters and the SR discovery target. Per the
benchmark's LAW role they are fit on the representative train split ("fit on
train, check correctness"); they reproduce exactly from OLS on train.csv (53,333
workers, ACS PUMS 2019; verified 2026-05-30). NEVER exposed in priors.

The quartic FORM is documented in the reference PDF: "Murphy and Welch (1990)
note that allowing for quartic terms in experience is empirically important for
fitting the earnings curve" (Heckman et al. 2005, NBER WP 11544, PDF p. 23 /
heckman_2003.txt line 976; Heckman et al. themselves also try a quartic, line 2118).
Murphy & Welch fit 1963-1987 CPS white males; the coefficients here are
re-estimated on the 2019 ACS train split (standard practice — LAW is fit on train).

Historical-filing note (corrected 2026-05-30, field-classification re-audit):
earlier versions placed these six in OTHER_CONSTANTS citing "data_spec §0.2.4 —
values fit from data are not LAW". That rule is superseded; the axis is "defining
coefficient (LAW) vs universal/derived/structural given (OTHER)", not "fitted vs
not". These six are defining coefficients -> LAW.

OTHER_CONSTANTS — {} (empty)
-----------------------------------------------------------------------
The form consumes no given constant; the x^2/x^3/x^4 exponents are structural
and stay inline in predict(). OTHER_CONSTANTS = {}.

On the representative test split the quartic achieves test RMSE = 0.6657
(R^2 0.193), the best of the baselines — edging the quadratic (0.6674) and
linear (0.6695) and clearly beating the train-mean naive predictor (0.7413).

Type designation: TYPE I — globally fitted; no per-cluster refit.

Column mapping (paper notation → released CSV):
  ln[Y(s, x)] → log_wage    (col 0, natural log of adjusted annual wage)
  s            → SCHL_years (col 1, years of schooling 0-22)
  x            → exp        (col 2, potential experience 0-58)
  x^2, x^3, x^4 → computed internally (NOT CSV columns — SR discovery challenge)

The central SR challenge for this baseline: discover that x^3 and x^4 terms
improve fit, on top of the already-hard challenge of finding x^2.
"""

import numpy as np

USED_INPUTS  = ["SCHL_years", "exp"]
PAPER_REF    = "reference/heckman_2003.pdf"  # Murphy & Welch (1990) cited at PDF p. 23
EQUATION_LOC = (
    "Murphy & Welch (1990), J. Labor Economics 8(2), 202-229; "
    "quartic form confirmed as 'empirically important' in "
    "Heckman, Lochner & Todd (2005) NBER WP 11544, PDF p. 23 / heckman_2003.txt line 976."
)

# === LAW_CONSTANTS — the quartic Mincer's defining coefficients (fit on train) ===
# Reproduce EXACTLY from OLS on train.csv (verified 2026-05-30). SR discovery target;
# never in priors. Reclassified OTHER->LAW 2026-05-30 (data_spec §0.2.4 superseded).
LAW_CONSTANTS = {
    "alpha":  8.788245264933096,       # intercept (log-wage at s=0, x=0)
    "rho_s":  0.11632026195707029,     # schooling return
    "beta_0": 0.015255538362460403,    # linear-experience coefficient
    "beta_1": 0.0019678574519002494,   # quadratic-experience coefficient
    "beta_2": -9.932045314264438e-05,  # cubic-experience coefficient
    "beta_3": 1.2221913482957929e-06,  # quartic-experience coefficient
}

# === OTHER_CONSTANTS — none (no given constant; x^2/x^3/x^4 exponents inline) ===
OTHER_CONSTANTS = {}

LOCAL_FITTABLE = {}    # Type I — no per-cluster refit


def predict(X: np.ndarray, alpha: float, rho_s: float, beta_0: float,
            beta_1: float, beta_2: float, beta_3: float) -> np.ndarray:
    """Predict log_wage from the Murphy-Welch (1990) quartic-in-experience function.

    ln Y = alpha + rho_s * s + beta_0 * x + beta_1 * x^2 + beta_2 * x^3 + beta_3 * x^4

    X: (n, 2) — columns [SCHL_years, exp] in USED_INPUTS order.
    Coefficients are the LAW_CONSTANTS; the harness passes predict(X, **LAW_CONSTANTS).
    Note: x^2, x^3, x^4 are computed internally — intentionally absent from the
    released CSV (discovering polynomial structure is the SR challenge).

    Source: Murphy & Welch (1990) J. Labor Economics 8(2), 202-229;
            quartic form cited in Heckman et al. (2005) NBER WP 11544, p. 23.
    """
    X = np.asarray(X, dtype=np.float64)
    s = X[:, 0]    # SCHL_years
    x = X[:, 1]    # exp (potential experience)
    return (alpha
            + rho_s * s
            + beta_0 * x
            + beta_1 * x ** 2
            + beta_2 * x ** 3
            + beta_3 * x ** 4)
