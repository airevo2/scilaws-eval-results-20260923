"""Cobb-Douglas urban scaling — log_GMP via TFP decomposition (Lobo 2013).

Lobo, J., Bettencourt, L. M. A., Strumsky, D., and West, G. B. (2013).
*Urban scaling and the production function for cities.*
PLOS ONE 8(3): e58407. DOI:10.1371/journal.pone.0058407.

This module implements the Lobo 2013 Cobb-Douglas urban production function
framework as a baseline predictor for log_GMP. The theoretical derivation
(Eqs. 7–16, PDF pp. 3–4) shows that the constancy of the labor income share
alpha across cities forces the production function into Cobb-Douglas form:

    Y_i = A_i * L_i^(1-alpha) * K_i^alpha          [Eq. 16, PDF p. 4]

where alpha is the capital income share (≈ 0.30 for US MSAs, confirmed
population-size-independent across 1969–2009 data, Fig. 3 and surrounding
text, PDF p. 4; so 1-alpha ≈ 0.70 is the labor share).

The urban scaling relations (Eq. 3, PDF p. 2) applied to this framework give:

    ln Y_i = ln Y_0 + beta * ln N_i + xi_i           [Eq. 3, PDF p. 2]

where xi_i is the SAMI (Scale-Adjusted Metropolitan Indicator) for city i,
representing the size-independent deviation of city i from the scaling
expectation. This is the operationalized regression form for this dataset.

The TFP size-dependence is (Eq. 25, PDF p. 5):

    beta_A = (1-alpha)*(beta_W - beta_L) + alpha*(beta_R - beta_K)

which, under the approximation beta_R ≈ beta_K and with beta_W ≈ 1.146
(Eq. 4, PDF p. 2–3), beta_L ≈ 1, and (1-alpha) ≈ 0.70, gives:

    beta_A ≈ (1-alpha)*(beta_W - beta_L) ≈ 0.70*(1.146 - 1.0) ≈ 0.102 ≈ 0.11

The benchmark dataset contains only log_GMP and log_pop (not separate wages,
employment, or capital); therefore, the operationalizable formula is the same
log-linear scaling regression as Bettencourt (2007), but with alpha = 0.30 as
a paper-published LAW_CONSTANT constraining the theoretical interpretation,
and beta fitted to GMP (which plays the role of Y_i encompassing both labor
and capital income).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
- alpha : 0.30 — capital income share of total output; empirically confirmed
          population-size-independent (correlation ≈ 0.05) across all US MSAs
          and Micropolitan Areas 1969–2009 (PDF p. 4 txt; implied 1-alpha ≈ 0.70
          as stated in abstract and Eq. 26 approximation). Lobo et al. (2013)
          use 1-alpha ≈ 0.70 (labor share) and alpha ≈ 0.30 (capital share)
          throughout their empirical derivation.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The log-linear form follows directly from logarithmizing the power law
Y = Y_0 * N^beta; no additional structural factors appear.

LOCAL_FITTABLE — per-cluster, fit by OLS
-----------------------------------------
- log_Y0 : log of the scaling normalization constant (= OLS intercept); absorbs
           the Cobb-Douglas baseline productivity A_0 and factor-price terms.
           init = None (OLS; closed form).
- beta    : scaling exponent for GMP ~ N^beta; theoretically constrained to
           be a weighted combination of factor-specific exponents. For wages
           alone: beta_W ≈ 1.146 (Eq. 4, PDF p. 2–3, 943 US urban areas
           2009–2011, R² = 0.97). For GMP (wages + capital income), beta is
           fit to data; expected ≈ 1.10–1.20.
           init = None (OLS; closed form).

Note: xi_i (the SAMI, i.e. the size-adjusted productivity residual) is
implicitly the prediction error: xi_i = log_GMP_i - predict(X_i, log_Y0, beta).
The TFP size-dependence beta_A ≈ 0.11 (Eq. 25 / abstract) is a diagnostic
derived quantity, not a free parameter in the regression.
"""

import numpy as np

USED_INPUTS = ["log_pop"]
PAPER_REF = "summary_formula_lobo_2013.md"
EQUATION_LOC = (
    "Lobo et al. 2013, Eq. (3), PDF p. 2 — ln Y_i = ln Y_0 + beta * ln N_i + xi_i; "
    "Eq. (4), PDF p. 2–3 — empirical beta_W ≈ 1.146 for total wages (R² = 0.97); "
    "alpha = 0.30 from PDF p. 4 Cobb-Douglas derivation (Eqs. 7–16); "
    "beta_A ≈ 0.11 from Eq. (25) / abstract."
)

# alpha = capital income share; 1-alpha = labor income share ≈ 0.70.
# Empirically confirmed population-size-independent across US MSAs 1969–2009.
# Lobo et al. 2013, PDF p. 4 txt; stated as 1-alpha ≈ 0.70 in Eq. (26) context.
LAW_CONSTANTS = {
    "alpha": 0.30,
}

OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "log_Y0": {"init": None},
    "beta":   {"init": None},
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray, alpha: float = 0.30) -> dict:
    """OLS fit of the Lobo 2013 Cobb-Douglas urban scaling regression.

    X_fit: (n, 1) — column [log_pop], i.e. ln(population).
    y_fit: (n,)   — log_GMP = ln(GMP_i) for each MSA.
    alpha: LAW_CONSTANT — capital income share (≈ 0.30 for US MSAs).
           Passed by the harness as a keyword argument; not used in the
           regression fitting since wages/employment are unavailable, but
           accepted per the v2 contract: fit() kwargs = LAW_CONSTANTS.

    Returns: {"log_Y0": float, "beta": float}.

    OLS is the exact method used by Lobo et al. 2013 for the scaling
    regression (Eq. 3, PDF p. 2 — ln Y_i = ln Y_0 + beta * ln N_i + xi_i).
    The residuals xi_i are the Scale-Adjusted Metropolitan Indicators (SAMIs),
    Eq. (5), PDF p. 3 — they are the size-adjusted GMP deviations per city.

    Note: the LAW_CONSTANT alpha = 0.30 constrains the theoretical framework
    (Cobb-Douglas derivation, Eqs. 7–16) but does not enter the regression
    fitting for log_GMP directly, since wages and employment are not separately
    available in the benchmark dataset. The prediction form is identical to
    bettencourt_2007; the difference lies in the theoretical grounding.
    """
    log_pop = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # Design matrix for OLS: [intercept, log_pop]
    A = np.column_stack([np.ones_like(log_pop), log_pop])
    # Least-squares solution: [log_Y0, beta]
    result, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    log_Y0, beta = float(result[0]), float(result[1])
    return {"log_Y0": log_Y0, "beta": beta}


def predict(X: np.ndarray, log_Y0: float, beta: float, alpha: float = 0.30) -> np.ndarray:
    """Predicted log_GMP from the Cobb-Douglas urban scaling law.

    X: (n, 1) — column [log_pop], i.e. ln(metropolitan population).
    log_Y0: LOCAL — OLS intercept (log of scaling normalization constant).
    beta:   LOCAL — OLS scaling exponent.
    alpha:  LAW   — capital income share (≈ 0.30); accepted per v2 contract
                    (predict kwargs = LAW_CONSTANTS ∪ LOCAL_FITTABLE).
                    Not used in the prediction formula directly; theoretical
                    grounding only.

    Returns: (n,) array of predicted log_GMP = log_Y0 + beta * log_pop.

    The prediction residuals xi_i = log_GMP_i - predicted_i are the
    Scale-Adjusted Metropolitan Indicators (SAMIs), which capture
    city-specific productivity deviations independent of population size
    (Lobo et al. 2013, Eq. 5, PDF p. 3). The Cobb-Douglas framework
    (with alpha = 0.30 as LAW_CONSTANT) provides the theoretical grounding
    for interpreting these residuals as size-adjusted TFP.
    """
    log_pop = np.asarray(X[:, 0], dtype=float)
    return log_Y0 + beta * log_pop
