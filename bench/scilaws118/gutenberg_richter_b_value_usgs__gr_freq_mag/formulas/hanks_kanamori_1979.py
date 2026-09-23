"""Hanks & Kanamori (1979) — moment-magnitude scale + Kagan (2010) theoretical b.

Hanks, T. C. and Kanamori, H. (1979). A moment magnitude scale. Journal of
Geophysical Research: Solid Earth, 84(B5), 2348-2350. DOI: 10.1029/JB084iB05p02348.

Kagan, Y. Y. (2010). Earthquake size distribution: Power law with exponent β ≡ 1/2?
Tectonophysics, 490(1-2), 103-114. DOI: 10.1016/j.tecto.2010.04.034.

Hanks & Kanamori (1979) define the moment magnitude scale (PDF p. 1, journal
p. 2348, displayed Eq. immediately below abstract):

    M_w = (2/3) log10(M_0) - 10.7

b_GR vs β distinction
---------------------
In seismology there are two related but distinct slope parameters:
  - β (beta): the Pareto (power-law) exponent of the seismic-moment distribution.
              F(M) = (Mt/M)^β  (Kagan 2010 Eq. 4, PDF p. 9).
  - b_GR:     the Gutenberg-Richter slope in the frequency-magnitude relation
              log10 N(M_w) = a - b_GR * M_w.
They are related by b_GR = (3/2) β, i.e. b_GR = 1.5β
(Kagan 2010 PDF p. 9, text following Eq. 4: "b = (3/2) β").

Kagan (2010) theoretical prediction
------------------------------------
Kagan (2010) argues that the true universal β is close to 0.5, based on
theoretical branching-process and percolation arguments:
  - Kagan 2010 PDF p. 3 (text line 66): "larger than 1/2 (or 0.75 for the
    b-value)" — directly equates β = 1/2 with b_GR = 0.75.
  - Kagan 2010 PDF p. 3 (lines 26-28, Abstract): "the recently obtained
    β-value of 0.63 could be reduced to about 0.52-0.56: near the universal
    constant value (1/2) predicted by theoretical arguments."
  - Kagan 2010 PDF p. 27 (line 990): "We take a slope β to be 0.5."

Applying b_GR = 1.5 × β = 1.5 × 0.5 = 0.75 gives the Kagan (2010) theoretical
prediction for the G-R slope. This is a LAW_CONSTANT: a theoretical prediction
from the physics of earthquake-size self-similarity, not a free empirical fit.

LAW_CONSTANTS — the formula's defining coefficients (b, a)
---------------------------------------------------------
b = 0.75: Kagan (2010) theoretical G-R slope derived from β = 0.5 (universal
  Pareto exponent for seismic-moment distribution) via b_GR = (3/2) β = 0.75.
  PDF-traceable: Kagan 2010 PDF p. 9 (line 259 in kagan_2010.txt): "b = (3/2)β";
  β = 0.5 stated at PDF p. 3 (kagan_2010.txt line 66): "larger than 1/2 (or
  0.75 for the b-value)"; and PDF p. 27 (kagan_2010.txt line 990): "We take a
  slope β to be 0.5."

  This is a SCIENTIFIC CLAIM (the universality of β = 0.5 from branching
  process theory), not an empirical per-dataset fit — the discovery target.

a = 6.3859: the log-seismicity intercept. With the slope frozen at b = 0.75,
  a is the remaining defining coefficient of log10 N = a − b·M_w. It is fit
  ONCE on the training rows (M_w < 7.5) under the constraint b = 0.75
  (mean residual = 0): a = mean(log10_N_train + 0.75·M_threshold_train) =
  6.3859 (reproducible from data/train.csv). A coefficient that is fit and is
  the formula's characteristic parameter is LAW (the same axis as b), not a
  "given" — so it lives in LAW_CONSTANTS, not as a hidden predict default.

OTHER_CONSTANTS
---------------
None. The formula is dimensionally clean once b is fixed; log10 is the
structural base-10 operator (literal numeral 10 in the body).

Type designation: Type I — single globally-aggregated frequency-magnitude
table, no cluster structure. LOCAL_FITTABLE = {}.

Column mapping:
    M_threshold (col 1) → M_w  (moment-magnitude threshold; the USGS data is
                                already filtered to the Mw family per
                                data_raw/README.md §5)
    log10_N     (col 0) → log10 N(M_w)  (cumulative annual rate)

Caveats
-------
- The b = 0.75 prediction is tested on the USGS NEIC 1980-2024 catalog. On
  the test split (M_w >= 7.5; the large-event tail) this gives RMSE ≈ 0.759
  and R² ≈ -0.40 — worse than the free-b baselines (aki_1965, kagan_2010
  R² ≈ 0.82) but better than the previously incorrect b = 2/3 (R² ≈ -1.03).
  The negative R² is a physically meaningful result: the theoretical b = 0.75
  is a worse predictor than the sample mean alone on this tail, indicating
  the empirical b in the large-event tail is steeper than 0.75. This is
  acceptable per §9.6 (weakest rung); the module serves as a theoretical-
  prediction contrast rung in the benchmark.
- `a` is a per-dataset offset fitted ONCE on the training rows
  (M_w < 7.5) under the constraint b = 0.75 (mean residual = 0). It is a
  defining coefficient and is declared in LAW_CONSTANTS; the harness invokes
  `predict(X, **LAW_CONSTANTS)` so both b = 0.75 and a = 6.3859 arrive as
  kwargs. Per Type I conventions there is no per-row `fit()`.
- Cited PDF on disk: `reference/hanks_kanamori_1979.pdf` (3 pages, ~200 KB,
  US Federal authored content; OA via Resolution Copper EIS mirror).
- Supporting evidence for β = 0.5 / b_GR = 0.75 on disk:
    reference/kagan_2010.pdf p. 9 (b = (3/2)β), p. 3 (β = 0.5 → b = 0.75),
    p. 27 (β = 0.5 assumed for theoretical curves).
"""
import numpy as np

USED_INPUTS = ["M_threshold"]
PAPER_REF = "summary_formula_hanks_kanamori_1979.md"
EQUATION_LOC = "Hanks & Kanamori 1979, displayed Eq. below abstract, PDF p. 1 (journal p. 2348); b = 0.75 Kagan (2010) theoretical prediction via β = 0.5, b_GR = 1.5β (Kagan 2010 PDF p. 9)"

# === LAW_CONSTANTS — the formula's defining coefficients (b frozen, a fitted) ===
LAW_CONSTANTS = {
    "b": 0.75,         # Kagan (2010) theoretical G-R slope: b_GR = 1.5 * β = 1.5 * 0.5 = 0.75
                       # PDF trace: kagan_2010.txt p. 9 "b = (3/2)β"; p. 3 "β = 0.5 → b = 0.75"
    "a": 6.3859000000000004,  # log-seismicity intercept; fit on USGS NEIC train (M_w < 7.5)
                              # under b = 0.75: a = mean(log10_N + 0.75*M_threshold)
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, b: float, a: float) -> np.ndarray:
    """log10 N(M_w) = a - b * M_w.

    X: (n, 1) — column M_threshold (moment-magnitude threshold).
    b: G-R slope in moment magnitude (LAW_CONSTANTS; frozen at 0.75 per
       Kagan (2010) theoretical prediction: b_GR = 1.5 * β = 1.5 * 0.5 = 0.75;
       PDF trace: kagan_2010.txt p. 9 "b = (3/2)β", p. 3 "β = 0.5 → b = 0.75").
    a: log-seismicity intercept (LAW_CONSTANTS; fitted once on the training
       rows under the constraint b = 0.75: a = mean(log10_N + 0.75*M)).
    Both arrive via predict(X, **LAW_CONSTANTS).
    Returns log10 of the cumulative annual earthquake rate above M_w.
    """
    M = X[:, 0]
    return a - b * M
