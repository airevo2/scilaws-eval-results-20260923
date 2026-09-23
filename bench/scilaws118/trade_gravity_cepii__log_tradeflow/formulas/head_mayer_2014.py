"""Log-linearised naive gravity equation with free GDP elasticities.

Head & Mayer (2014) "Gravity Equations: Workhorse, Toolkit, and Cookbook",
CEPII Working Paper 2013-27 (Handbook of International Economics Ch. 3).
Primary equations:

  Definition 3 / Eq. (4), PDF p. 12 — Naive gravity with free GDP elasticities:

    X_ni = G * Y_i^a * Y_n^b * phi_ni

  In log-linear form:

    ln X_ni = beta_0 + a * ln Y_i + b * ln Y_n
              + beta_dist * ln Dist_ni
              + beta_contig * Contig_ni
              + beta_comlang * ComLang_ni
              + beta_col * Colony_ni
              + beta_fta * RTA_ni

  where phi_ni is the bilateral trade accessibility, parameterised as a log-linear
  function of distance and binary dummies (Eq. 22, PDF p. 25).

LAW_CONSTANTS — All Gravity column medians from Table 4, PDF p. 34
------------------------------------------------------------------
Table 4 pools 2508 estimates across 159 papers. "All Gravity" median column:

  a           = 0.97   (Origin GDP; Table 4, row "Origin GDP", median)
  b           = 0.85   (Destination GDP; Table 4, row "Destination GDP", median)
  beta_dist   = -0.89  (Distance; Table 4, row "Distance", median)
  beta_contig = 0.49   (Contiguity; Table 4, row "Contiguity", median)
  beta_comlang= 0.49   (Common language; Table 4, row "Common language", median)
  beta_col    = 0.91   (Colonial link; Table 4, row "Colonial link", median)
  beta_fta    = 0.47   (RTA/FTA; Table 4, row "RTA/FTA", median)

  beta_0 (intercept) is not in the meta-analysis table; set to 0.0 as a default
  (the unit-change absorbs measurement units; the intercept captures avg MR).

OTHER_CONSTANTS — beta_0 only: a single data-fit intercept (the gravity constant
k = ln G plus the log-unit shift from GDP in current USD and tradeflow in thousands
USD). Table 4 does not report an intercept (it is unit-dependent). It is the OLS
mean-of-residuals on the train split with all 7 slopes fixed at the Table 4 medians.
Value: -16.7055.

Type designation: Type I — each row is an independent (origin, destination, year)
observation. All parameters are globally applicable across dyads; no per-cluster
refit. LOCAL_FITTABLE = {} confirms this.

Column mapping (paper notation → released CSV):
  ln Y_i     → log_gdp_o    (origin-country log GDP)
  ln Y_n     → log_gdp_d    (destination-country log GDP)
  ln Dist_ni → log_dist     (log bilateral distance)
  Contig_ni  → contig       (contiguity binary)
  ComLang_ni → comlang_off  (official common language binary)
  Colony_ni  → col_dep_ever (colonial link binary)
  RTA_ni     → fta_wto      (WTO-notified RTA binary)

Caveats:
  - No country fixed effects (mu_i, mu_n) are in the released CSV. The intercept
    beta_0 proxies for average multilateral resistance — a known "gold medal
    mistake" (Baldwin & Taglioni 2007, cited in Head & Mayer p. 13). The
    frozen All Gravity meta-analysis medians apply when FEs are absent.
  - `year` is excluded from USED_INPUTS; the gravity form is cross-sectional.
  - `log_tradeflow` is in log(thousands USD); the unit difference from log(USD)
    is a constant shift absorbed into beta_0.
"""

import numpy as np

USED_INPUTS = [
    "log_gdp_o",
    "log_gdp_d",
    "log_dist",
    "contig",
    "comlang_off",
    "col_dep_ever",
    "fta_wto",
]

PAPER_REF = "summary_formula_head_2014.md"

EQUATION_LOC = (
    "Eq. (4), PDF p. 12 (naive gravity, Definition 3, free GDP elasticities); "
    "Eq. (22), PDF p. 25 (log-linearised structural gravity); "
    "Table 4, PDF p. 34 (All Gravity meta-analysis medians)"
)

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a":            0.97,    # Origin GDP elasticity; Table 4, p. 34, All Gravity median
    "b":            0.85,    # Destination GDP elasticity; Table 4, p. 34, All Gravity median
    "beta_dist":   -0.89,    # Distance; Table 4, p. 34, All Gravity median
    "beta_contig":  0.49,    # Contiguity; Table 4, p. 34, All Gravity median
    "beta_comlang": 0.49,    # Common language; Table 4, p. 34, All Gravity median
    "beta_col":     0.91,    # Colonial link; Table 4, p. 34, All Gravity median
    "beta_fta":     0.47,    # RTA/FTA; Table 4, p. 34, All Gravity median
}

# === OTHER_CONSTANTS — data-fit intercept (not a paper value) ===
OTHER_CONSTANTS = {
    # beta_0: the intercept absorbs the constant k = ln G (gravity constant),
    # the log-scale shift due to GDP measured in current USD (not normalised),
    # and the log-unit shift from tradeflow in thousands USD.
    # Table 4 does NOT report the intercept — it varies with the GDP unit chosen
    # by each of the 159 pooled papers. Value here is OLS-estimated on train data
    # (year <= 2016) with all other slope parameters fixed at the All Gravity medians.
    # OLS mean-of-residuals: beta_0 = mean(y_train - a*log_gdp_o - b*log_gdp_d - ...).
    # Computed value: -16.7055 (see prep_data.py for full diagnostics).
    "beta_0": -16.7055,  # data-fit intercept; not from Table 4 (unit-dependent)
}

LOCAL_FITTABLE = {}    # Type I — single global parameter set, no per-cluster refit.


def predict(
    X: np.ndarray,
    a: float,
    b: float,
    beta_dist: float,
    beta_contig: float,
    beta_comlang: float,
    beta_col: float,
    beta_fta: float,
) -> np.ndarray:
    """Predict log_tradeflow from naive gravity (Head & Mayer 2014 Eq. 4).

    LAW_CONSTANTS (slope) params are passed in from the harness; beta_0 is
    read from OTHER_CONSTANTS in the module namespace.

    Args:
        X: array of shape (n, 7), columns in USED_INPUTS order:
           [log_gdp_o, log_gdp_d, log_dist, contig, comlang_off, col_dep_ever, fta_wto]
        a, b: origin/destination GDP elasticities (LAW; Table 4 All Gravity medians).
        beta_dist, beta_contig, beta_comlang, beta_col, beta_fta: bilateral cost
            coefficients (LAW; Table 4 All Gravity medians).

    Returns:
        np.ndarray of shape (n,) — predicted log_tradeflow values.
    """
    beta_0 = OTHER_CONSTANTS["beta_0"]   # data-fit intercept from module namespace
    X = np.asarray(X, dtype=float)
    log_gdp_o    = X[:, 0]
    log_gdp_d    = X[:, 1]
    log_dist     = X[:, 2]
    contig       = X[:, 3]
    comlang_off  = X[:, 4]
    col_dep_ever = X[:, 5]
    fta_wto      = X[:, 6]

    return (
        beta_0
        + a            * log_gdp_o
        + b            * log_gdp_d
        + beta_dist    * log_dist
        + beta_contig  * contig
        + beta_comlang * comlang_off
        + beta_col     * col_dep_ever
        + beta_fta     * fta_wto
    )
