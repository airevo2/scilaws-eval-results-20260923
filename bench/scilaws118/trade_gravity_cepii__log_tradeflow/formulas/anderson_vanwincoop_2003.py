"""Structural gravity equation with theory-constrained unit GDP elasticities.

Anderson & van Wincoop (2003) "Gravity with Gravitas: A Solution to the Border
Puzzle", American Economic Review 93(1), 170-192.
On-disk PDF is NBER Working Paper 8079 (pre-publication version).

Core theoretical result — Eq. (17), PDF p. 12 (NBER WP 8079):

    ln x_ij = k + ln y_i + ln y_j + (1-sigma)*rho * ln d_ij
              + (1-sigma) * ln b_ij
              - (1-sigma) * ln P_i
              - (1-sigma) * ln P_j

A structurally imposed consequence (stated explicitly in the text following
Eq. 17, PDF p. 12): "the theoretical gravity equation imposes unitary income
elasticities." The income elasticities on y_i and y_j are both exactly 1 by
CES demand + market-clearing — they are not free parameters.

The composite distance/barrier coefficients (1-sigma)*rho and (1-sigma)*ln b_ij
are free; the multilateral resistance terms P_i and P_j are endogenous and
unobserved in the released CSV — they are proxied here by the intercept beta_0.

LAW_CONSTANTS — bilateral cost coefficients (Head & Mayer 2014 Table 4 medians)
-------------------------------------------------------------------------------
  The income elasticities of exactly 1 are an exact structural result of Eq. (17)
  (CES demand + market clearing), not a table coefficient, so they appear as inline
  numerals in the formula. The bilateral cost coefficients ARE frozen LAW_CONSTANTS:
  Anderson & van Wincoop (2003) Eq. (17) writes them only as composites (1-sigma)*rho
  and (1-sigma)*ln b_ij (no numerical values), so their frozen values are taken from
  Head & Mayer (2014) Table 4, p. 34, All Gravity median column — verified against
  head_mayer_2014.txt p.34:

  beta_dist:    -0.89  (Distance;        Head & Mayer 2014 Table 4, p. 34, All Gravity median)
  beta_contig:   0.49  (Contiguity;      Head & Mayer 2014 Table 4, p. 34, All Gravity median)
  beta_comlang:  0.49  (Common language; Head & Mayer 2014 Table 4, p. 34, All Gravity median)
  beta_col:      0.91  (Colonial link;   Head & Mayer 2014 Table 4, p. 34, All Gravity median)
  beta_fta:      0.47  (RTA/FTA;         Head & Mayer 2014 Table 4, p. 34, All Gravity median)

Note on PDF encoding: the NBER WP 8079 PDF has garbled character encoding in the
text layer; the equation and structural result above are confirmed by reading the
PDF page directly (pages 12, 14 of the NBER WP 8079). The AER published version
may have slightly different page numbering.

OTHER_CONSTANTS — beta_0 only: a single data-fit intercept that absorbs the gravity
constant k, the (unobserved, unmodelled) multilateral-resistance terms P_i, P_j, and
the log-unit shift (GDP in current USD vs tradeflow in thousands USD). It is the OLS
mean-of-residuals on the train split with all slopes fixed at the values above; NOT a
paper value (unit-dependent). Value: -19.8428.

Note: this is a REDUCED form of Anderson & van Wincoop — the multilateral-resistance
terms P_i, P_j (the paper's core contribution) are absorbed into the intercept rather
than modelled, because the released columns carry no country fixed effects. The
distinguishing structural claim retained here is the unit income elasticity (a=b=1),
which is exactly what separates this baseline from head_mayer_2014 (free a, b).

Type designation: Type I — each (origin, destination, year) row is independent.
The gravity formula applies a single global parameter set to all dyads.
LOCAL_FITTABLE = {} confirms Type I.

Column mapping (paper notation → released CSV):
  ln y_i     → log_gdp_o    (origin log GDP; coeff = 1, theory-fixed)
  ln y_j     → log_gdp_d    (destination log GDP; coeff = 1, theory-fixed)
  ln d_ij    → log_dist     (log bilateral distance; coeff = beta_dist)
  b_ij terms → contig, comlang_off, col_dep_ever, fta_wto (binary barriers)
  P_i, P_j   → absorbed into intercept beta_0

Caveats:
  - Multilateral resistance (P_i, P_j) is proxied by the intercept; without
    country fixed effects this introduces the "gold medal mistake" bias.
  - The elasticity of substitution sigma and trade-cost parameter rho are jointly
    unidentified (only their composite (1-sigma)*rho enters). We treat beta_dist
    = (1-sigma)*rho as a single free parameter.
  - The `year` column is not used; the formula is cross-sectional.
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

PAPER_REF = "summary_formula_anderson_2003.md"

EQUATION_LOC = (
    "Eq. (17), PDF p. 12 (NBER WP 8079) — structural gravity with unit income "
    "elasticities; text p. 12 states 'theoretical gravity equation imposes "
    "unitary income elasticities'. Init coefficients from Head & Mayer (2014) "
    "Table 4, p. 34, All Gravity medians."
)

# Unit income elasticities (coeff = 1 on both log_gdp_o and log_gdp_d) are
# structural — they appear as inline numerals, not as LAW_CONSTANTS.
# The bilateral cost parameters are LAW_CONSTANTS; their init values come from
# Head & Mayer (2014) Table 4, p. 34, All Gravity medians.
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "beta_dist":    -0.89,   # Distance; H&M 2014 Table 4, p. 34, All Gravity median
    "beta_contig":   0.49,   # Contiguity; H&M 2014 Table 4, p. 34, All Gravity median
    "beta_comlang":  0.49,   # Common language; H&M 2014 Table 4, p. 34, All Gravity median
    "beta_col":      0.91,   # Colonial link; H&M 2014 Table 4, p. 34, All Gravity median
    "beta_fta":      0.47,   # RTA/FTA; H&M 2014 Table 4, p. 34, All Gravity median
}

# === OTHER_CONSTANTS — data-fit intercept (not a paper value) ===
OTHER_CONSTANTS = {
    # beta_0: the intercept absorbs k (gravity constant from Eq. 17), the
    # multilateral resistance proxy, and the log-scale shift from GDP measured
    # in current USD (not normalised). With unit income elasticities (a=b=1),
    # log_gdp_o + log_gdp_d averages ~38 for this dataset, but log_tradeflow
    # averages ~7.55, so the required shift is ~ -30 before adding the distance
    # and dummy terms. Table 4 does NOT report the intercept — it is unit-dependent
    # and not a universal gravity "law" constant.
    # Value OLS-estimated on train data (year<=2016) with slopes fixed at above medians:
    # beta_0 = mean(y_train - (log_gdp_o + log_gdp_d + beta_dist*log_dist + ...))
    # Computed value: -19.8428 (mean-of-residuals OLS on training set).
    "beta_0": -19.8428,  # data-fit intercept; not from any paper table (unit-dependent)
}

LOCAL_FITTABLE = {}    # Type I — single global parameter set.


def predict(
    X: np.ndarray,
    beta_dist: float,
    beta_contig: float,
    beta_comlang: float,
    beta_col: float,
    beta_fta: float,
) -> np.ndarray:
    """Predict log_tradeflow under structural gravity (Anderson & van Wincoop 2003 Eq. 17).

    Income elasticities are structurally fixed at 1 (theory-imposed); log_gdp_o
    and log_gdp_d enter with coefficient +1. The bilateral cost parameters are
    LAW_CONSTANTS (initialised from Head & Mayer 2014 Table 4 medians). The
    intercept beta_0 is read from OTHER_CONSTANTS (data-fit; not from Table 4).

    Args:
        X: array of shape (n, 7), columns in USED_INPUTS order:
           [log_gdp_o, log_gdp_d, log_dist, contig, comlang_off, col_dep_ever, fta_wto]
        beta_dist, beta_contig, beta_comlang, beta_col, beta_fta: bilateral cost
            parameters (LAW; initialised from Head & Mayer 2014 Table 4 medians).

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

    # Unit income elasticities: structural result of Eq. (17) (AV 2003 p. 12).
    # Both log_gdp_o and log_gdp_d enter with coefficient exactly 1.
    return (
        beta_0
        + 1.0 * log_gdp_o      # unit income elasticity (theory-imposed)
        + 1.0 * log_gdp_d      # unit income elasticity (theory-imposed)
        + beta_dist    * log_dist
        + beta_contig  * contig
        + beta_comlang * comlang_off
        + beta_col     * col_dep_ever
        + beta_fta     * fta_wto
    )
