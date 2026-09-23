"""Preston-curve simple log-linear baseline.

Preston (1975) reports the simple log-linear correlation between national
life expectancy at birth (e0) and the natural logarithm of national income
per head (Y, constant 1963 USD) as

    e0 = a + b * loge(Y),

with cross-sectional Pearson correlations of r = 0.885 (1930s) and r =
0.880 (1960s). These correlation values are explicit on PDF p. 2 of the
on-disk abridged 2007 IJE reprint, paragraph beginning "The simple
correlation between life expectancy and the logarithm of income per head
is 0.885 in the 1930s and 0.880 in the 1960s". The same paragraph
notes that the log-linear form "consistently overestimated life
expectancy at lower levels of income," motivating Preston's preferred
logistic form (see preston_1975_logistic.py). The log-linear is retained
here as a textbook simpler comparator.

The numerical values of a and b are NOT printed in the abridged reprint
(and the unabridged original prints the logistic, not log-linear,
coefficients). a and b are nonetheless the *defining coefficients*
(intercept and log-income slope) of this log-linear Preston-curve form --
the SR discovery target for this baseline. They are fit on our CC0-1.0
1960s training split. "Fit on data" is not the LAW/OTHER axis; defining
coefficients are LAW whether published or re-fit on our data.

Coefficient provenance (LAW; CC0-1.0 training-data fit):
  a = 10.820753208677221
  b = 8.63390922772425
  Test MSE  = 34.955 (test R^2 = 0.7182; recovers Preston's reported within-decade
              fit r = 0.880 for the 1960s, summary §3 / PDF p. 2).
  Source: CC0-1.0 data/train.csv (38 rows, 1960s cross-section, representative split).
  Re-fit 2026-05-29 after re-scoping to the 1960s single cross-section (see
  prep_data.py CHANGE LOG); prior pooled-decade values were a=2.3334, b=8.6273.
  Reproduces from data/train.csv (numpy lstsq) -- verified 2026-05-30.

OTHER_CONSTANTS = {} -- this form e0 = a + b*loge(Y) has no fixed structural
number; both numbers are defining coefficients.

Validity caveat: this form lacks an upper asymptote, so for Y -> infinity
it predicts e0 -> infinity (unphysical). It is included as a baseline
for the simplest reasonable closed form, not as a defensible long-run
extrapolator.
"""

import numpy as np

USED_INPUTS = ["gdppc"]
PAPER_REF = "summary_formula_dataset_preston_1975.md"
EQUATION_LOC = (
    "Preston 1975, abridged 2007 IJE reprint, PDF p. 2: 'The simple "
    "correlation between life expectancy and the logarithm of income "
    "per head is 0.885 in the 1930s and 0.880 in the 1960s'. The "
    "fitted form is e0 = a + b * loge(Y); the per-decade a, b values "
    "are not printed in the abridged reprint."
)
# === LAW_CONSTANTS — the log-linear Preston-curve defining coefficients ===
LAW_CONSTANTS = {
    "a": 10.820753208677221,   # intercept (yr)
    "b": 8.63390922772425,     # slope wrt loge(Y) (yr per log-unit income)
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, a: float, b: float) -> np.ndarray:
    """Log-linear Preston-curve baseline.

    Parameters
    ----------
    X : np.ndarray, shape (n, 1)
        Column ordered per USED_INPUTS, i.e. column 0 is gdppc (USD_1963).
    a, b : float
        Defining coefficients (LAW_CONSTANTS), passed by the harness as
        predict(X, **LAW_CONSTANTS).

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted e0 in years. Unbounded above by construction.
    """
    Y = np.asarray(X[:, 0], dtype=float)
    return a + b * np.log(Y)
