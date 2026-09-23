"""Preston-curve logistic fit of life expectancy in log-income (fixed ceiling 80).

Preston (1975), "The changing relation between mortality and level of
economic development," explicitly endorses a logistic functional form
for the cross-sectional relation between national life expectancy at
birth (e0) and national income per head (Y, constant 1963 USD). The
text fits the linearised transformation

    loge(80 / e0 - 1) = alpha + beta * loge(Y),

which is equivalent to the closed form

    e0 = 80 / (1 + exp(alpha + beta * loge(Y))).

The constant 80 in the linearisation is the FIXED theoretical maximum
adopted by Preston (PDF p. 5 of the on-disk reprint, paragraph beginning
"the proportion of variance in loge(80/e0 - 1) explained by income is
0.800 in the 1930s and 0.847 in the 1960s"). The practical upper
asymptote of the fitted curve is below 80: Preston reports it as
~66.8 years for the 1930s and ~71.5 years for the 1960s on PDF p. 3
(the paragraph beginning "The upper asymptotes differ by 4-7 years,
suggesting that the maximum average length of life to be attained by
gains in income alone was some 66.8 years in the 1930s and 71.5 years
in the 1960s"). The 80 in the form is the structural ceiling; the
fitted asymptote follows from alpha and beta.

Preston DID publish this logistic form WITH fitted (alpha, beta) per
decade in the unabridged 1975 Population Studies original. On PDF p. 6 of
the on-disk full reprint (reference/preston_1975_full.txt line 273) the
two equations are printed verbatim:

    1930s:  e0 = 80 / (1 + exp{-1.6251 + 2.0768 * (0.9317) * Y'})
    1960s:  e0 = 80 / (1 + exp{-2.1354 + 2.1697 * (0.7672) * Y'})

so (alpha, beta) ARE the paper's published *defining coefficients* of the
Preston logistic curve (the discovery target), not merely incidental
numbers. The abridged 2007 IJE reprint (reference/preston_1975.pdf) drops
the coefficient table (abridgement footnote PDF p. 1, "Abridged with the
author's consent"), but the unabridged original prints them.

Preston's published 1960s (alpha, beta) are expressed against a different
income transform Y' (a scaled/centred log-income, factor 0.7672), so they
are NOT numerically identical to our gdppc->log(Y) parameterisation. We
therefore recover the SAME defining coefficients on our own data: the
values baked into LAW_CONSTANTS below are the scipy.optimize.least_squares
re-fit of e0 = ceiling / (1 + exp(alpha + beta*loge(Y))) on the CC0-1.0
1960s training split. Being the formula's defining coefficients (whether
read from Preston's table or re-fit on our data), they are LAW_CONSTANTS,
not OTHER -- "fit on data" is not the LAW/OTHER axis (the axis is
defining-coefficient vs given/universal/structural).

Coefficient provenance (LAW; CC0-1.0 training-data re-fit of the published form):
  alpha = 2.9920451796666057
  beta  = -0.7487360435089442
  Test MSE  = 28.564 (test R^2 = 0.7698; strongest reference baseline -- Preston's
              preferred form with the structural ceiling 80).
  Source: CC0-1.0 data/train.csv (38 rows, 1960s cross-section, representative split).
  Re-fit 2026-05-29 after re-scoping to the 1960s single cross-section (see
  prep_data.py CHANGE LOG); prior pooled-decade values were alpha=1.9964, beta=-0.4578.
  Reproduces from data/train.csv (least_squares, lm) -- verified 2026-05-30.

OTHER_CONSTANTS = {ceiling: 80}: the fixed theoretical maximum Preston
adopts in his logistic linearisation loge(80/e0 - 1) = alpha + beta*loge(Y).
It is a fixed structural number written into the formula's definition (the
numerator "80" of both decade equations, full reprint PDF p. 6; also the
"80" in the goodness-of-fit transform on abridged PDF p. 5), NOT a
defining coefficient. The practical fitted asymptote (~66.8 / ~71.5) is
below 80 and follows from alpha and beta. -> OTHER (structural given).
"""

import numpy as np

USED_INPUTS = ["gdppc"]
PAPER_REF = "summary_formula_dataset_preston_1975.md"
EQUATION_LOC = (
    "Preston 1975, full Population Studies original, PDF p. 6 "
    "(reference/preston_1975_full.txt line 273): "
    "e0 = 80 / (1 + exp(alpha + beta * loge(Y))); the published 1960s "
    "logistic has alpha=-2.1354, beta=2.1697 against income transform Y'. "
    "Structural ceiling 80 fixed by Preston (also abridged reprint PDF p. 5); "
    "practical asymptote 66.8 (1930s) / 71.5 (1960s) on PDF p. 3"
)
# === LAW_CONSTANTS — the Preston-logistic defining coefficients (discovery target) ===
LAW_CONSTANTS = {
    "alpha": 2.9920451796666057,   # intercept of linearised logistic in log(Y)
    "beta": -0.7487360435089442,   # slope wrt log(Y) (negative => concave saturating)
}
# === OTHER_CONSTANTS — fixed structural number in Preston's linearisation ===
OTHER_CONSTANTS = {
    "ceiling": 80.0,   # theoretical maximum in loge(80/e0 - 1); full reprint PDF p. 6
}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Logistic Preston curve with fixed ceiling 80.

    Parameters
    ----------
    X : np.ndarray, shape (n, 1)
        Column ordered per USED_INPUTS, i.e. column 0 is gdppc (USD_1963).
    alpha, beta : float
        Preston-logistic defining coefficients (LAW_CONSTANTS), passed by the
        harness as predict(X, **LAW_CONSTANTS).

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted e0 in years. Bounded in (0, ceiling) by construction.
    """
    ceiling = OTHER_CONSTANTS["ceiling"]
    Y = np.asarray(X[:, 0], dtype=float)
    log_Y = np.log(Y)
    return ceiling / (1.0 + np.exp(alpha + beta * log_Y))
