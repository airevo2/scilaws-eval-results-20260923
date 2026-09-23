"""Aki (1965) — Gutenberg-Richter frequency-magnitude relation (Type I).

Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a − bM
and its confidence limits. Bulletin of the Earthquake Research Institute,
University of Tokyo, 43, 237–239.
URL: https://repository.dl.itc.u-tokyo.ac.jp/records/32706

Formula (PDF p. 1, unlabelled display equation):
    log10 N = a − b * M

where N is the cumulative number of earthquakes with magnitude ≥ M, M is the
moment magnitude (dimensionless), a is the seismic-activity intercept
(per-dataset fit), and b is the G-R slope (per-dataset fit).

LAW_CONSTANTS — the formula's defining coefficients (a, b)
----------------------------------------------------------
    a = 8.1352   (seismic-activity intercept)
    b = 1.0191   (Gutenberg-Richter slope)

These two coefficients ARE the G-R law's defining parameters — the SR
discovery target. Aki (1965) states the functional form log N = a − bM and
gives the MLE *procedure* for b, but publishes no numerical value for any
specific catalog (PDF p. 1: "the b value … is determined by the least squares
method"). For this benchmark the values are obtained by OLS on the USGS NEIC
1980-2024 training split (M < 7.5): a = 8.1352, b = 1.0191 (reproducible from
data/train.csv). Per the four-field definition, a coefficient that is fit
(by the paper OR by our train split) and is the formula's characteristic
parameter is a LAW_CONSTANT — never a "given". b = 1.0191 is consistent with
Frohlich & Davis (1993) global range [0.72, 1.34].

OTHER_CONSTANTS
---------------
None. The formula is dimensionally clean: log10 is the implicit base-10
operator; its base is a structural choice, not a free constant.

Column mapping:
    M_threshold (col 1) → M  (moment-magnitude threshold)
    log10_N     (col 0) → log10 N  (cumulative annual frequency, base-10)

Caveats:
- The formula is valid only within the completeness window of the catalog
  (above Mc, below saturation regime). The benchmarked data (M ∈ [5.6, 9.1])
  was filtered to the completeness window by gr_aggregate.py (Mc ≈ 5.6).
- a and b are the LAW_CONSTANTS (the discovery target); the harness passes
  them as kwargs via predict(X, **LAW_CONSTANTS). For this single catalog
  they are obtained by OLS on the training split.

Type designation: Type I — each row is an independent cumulative-rate
observation for a fixed global catalog; no cluster structure; a and b are
the formula's defining coefficients. LOCAL_FITTABLE = {}.
"""

import numpy as np

USED_INPUTS = ["M_threshold"]
PAPER_REF = "summary_formula_aki_1965.md"
EQUATION_LOC = "Aki 1965, unlabelled display eq., PDF p. 1"

# === LAW_CONSTANTS — the G-R law's defining coefficients (discovery target) ===
LAW_CONSTANTS = {
    "a": 8.1352,    # seismic-activity intercept; OLS on USGS NEIC train (M < 7.5)
    "b": 1.0191,    # Gutenberg-Richter slope; OLS on USGS NEIC train (M < 7.5)
}
OTHER_CONSTANTS = {}        # no external constants required
LOCAL_FITTABLE = {}         # Type I — no per-cluster parameters


def predict(X: np.ndarray, a: float, b: float) -> np.ndarray:
    """log10 N = a - b * M.

    X: (n, 1) — column M_threshold (moment-magnitude threshold).
    a: log10 of seismic activity level (LAW_CONSTANTS).
    b: Gutenberg-Richter slope (LAW_CONSTANTS).
    Both arrive via predict(X, **LAW_CONSTANTS).
    Returns log10 of the cumulative annual earthquake rate above M.
    """
    M = X[:, 0]
    return a - b * M
