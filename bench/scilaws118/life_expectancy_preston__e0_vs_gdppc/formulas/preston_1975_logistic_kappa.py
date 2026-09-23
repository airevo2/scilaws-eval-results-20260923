"""Preston-curve logistic with FITTABLE upper asymptote kappa.

Variant of preston_1975_logistic.py: Preston (1975) reports practical
upper asymptotes that are NOT the fixed 80-year ceiling used in his
linearisation; they are kappa ~ 66.8 years (1930s) and kappa ~ 71.5 years
(1960s) — i.e. the ceiling of e0 actually achievable through income gains
alone within a given decade's technology environment (PDF p. 3 of the
on-disk abridged 2007 IJE reprint, paragraph beginning "The upper
asymptotes differ by 4-7 years, suggesting that the maximum average
length of life to be attained by gains in income alone was some 66.8
years in the 1930s and 71.5 years in the 1960s").

The summary at reference/summary_formula_dataset_preston_1975.md gives the
form explicitly:

    e0 = kappa / (1 + exp(alpha + beta * loge(Y)))

with kappa "approximately 66.8 (1930s) or approximately 71.5 (1960s)". This
task uses the 1960s cross-section, so the fitted kappa is expected to land
near Preston's reported 1960s asymptote of ~71.5 years.

This is structurally distinct from preston_1975_logistic.py:
- preston_1975_logistic.py FIXES the linearisation ceiling at 80.
- This module fits kappa as a third parameter, letting the asymptote be
  determined by the data rather than by Preston's linearisation constant.

All three parameters (alpha, beta, kappa) are the *defining coefficients*
of this logistic-curve variant -- they fully determine the Preston curve's
shape (alpha, beta) and its upper asymptote (kappa). They are the SR
discovery target for this baseline. Preston himself fits this same logistic
form per decade (with the asymptote being the practical ceiling, ~66.8
years in the 1930s / ~71.5 years in the 1960s, full reprint PDF p. 6 line
273 / abridged PDF p. 3), so an asymptote parameter is a genuine published
characteristic of the curve, not an incidental given.

This is a Type I task (no group_id, no per-group fit, no fit()): kappa is
fit ONCE globally on the training split and baked as a module constant,
exactly like alpha and beta -- it is NOT a LOCAL_FITTABLE (per-group,
fit-at-runtime) parameter. Being a defining coefficient fit on our data, it
is LAW (the LAW/OTHER axis is defining-coefficient vs given, not fit vs
not-fit).

The fitted kappa = 72.67 lands close to Preston's reported 1960s practical
asymptote of 71.5 years (PDF p. 3) -- an independent recovery of the paper's
asymptote from the CC0 data, confirming the logistic form is well-identified
on a single clean cross-section. (On the prior pooled 1900s+1930s training
data kappa fitted to a degenerate 118.5 because three free parameters were
weakly identified across the cross-decade level shift; re-scoping to the
1960s single cross-section resolved this.)

Coefficient provenance (LAW; CC0-1.0 training-data fit):
  alpha = 4.890146292982432
  beta  = -1.2232208143238406
  kappa = 72.67173952368388   (~= Preston's reported 71.5 yr, PDF p. 3)
  Test MSE  = 28.778 (test R^2 = 0.7680).
  Source: CC0-1.0 data/train.csv (38 rows, 1960s cross-section, representative split).
  Re-fit 2026-05-29 after re-scoping to the 1960s single cross-section (see
  prep_data.py CHANGE LOG); prior pooled-decade values were alpha=2.0251,
  beta=-0.3078, kappa=118.53.
  Reproduces from data/train.csv (least_squares, lm) -- verified 2026-05-30.

OTHER_CONSTANTS = {} -- this variant has no fixed structural ceiling (kappa
replaces the 80 of preston_1975_logistic.py as a fitted defining coefficient);
the only inline literal is the structural "1" of the logistic 1/(1+exp(.)).
"""

import numpy as np

USED_INPUTS = ["gdppc"]
PAPER_REF = "summary_formula_dataset_preston_1975.md"
EQUATION_LOC = (
    "Preston 1975, abridged 2007 IJE reprint, PDF p. 3 (practical "
    "upper asymptotes 66.8 / 71.5 years per decade); closed form "
    "e0 = kappa / (1 + exp(alpha + beta * loge(Y))) with kappa as "
    "the per-decade asymptote. This module treats kappa as a third "
    "defining coefficient estimated from data rather than fixing the "
    "asymptote at the linearisation ceiling 80."
)
# === LAW_CONSTANTS — the three defining coefficients of this logistic variant ===
LAW_CONSTANTS = {
    "alpha": 4.890146292982432,    # intercept of linearised logistic in log(Y)
    "beta": -1.2232208143238406,   # slope wrt log(Y)
    "kappa": 72.67173952368388,    # fitted upper asymptote (yr); ~ Preston 1960s 71.5
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, alpha: float, beta: float, kappa: float) -> np.ndarray:
    """Logistic Preston curve with fittable upper asymptote.

    Parameters
    ----------
    X : np.ndarray, shape (n, 1)
        Column ordered per USED_INPUTS, i.e. column 0 is gdppc (USD_1963).
    alpha, beta, kappa : float
        Defining coefficients (LAW_CONSTANTS), passed by the harness as
        predict(X, **LAW_CONSTANTS). kappa is the fitted upper asymptote.

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted e0 in years. Bounded in (0, kappa) by construction.
    """
    Y = np.asarray(X[:, 0], dtype=float)
    log_Y = np.log(Y)
    return kappa / (1.0 + np.exp(alpha + beta * log_Y))
