"""Kagan (2010) — Gutenberg-Richter relation with threshold parameterization.

Kagan, Y. Y. (2010). Statistical distributions of earthquake numbers: consequence
of branching process. arXiv:0908.1207v2 [physics.geo-ph], 18 Mar 2010.
Published in Geophysical Journal International 180(3):1313–1328, 2010.

Formula (Equation 2, PDF p. 8):
    log10 N(m) = a_t − b * (m − m_t)    for m_t ≤ m

where a_t = log10(N(m_t)) is the log-seismicity at the threshold magnitude m_t,
b is the G-R slope, and m_t is the completeness threshold.

The paper (PDF p. 8, lines 227-233) states the form
"log10 N(m) = a_t − b (m − m_t) … where N(m) is the number of earthquakes with
magnitude ≥ m, and **a_t and b are parameters**" — i.e. a_t and b are the
formula's defining coefficients (the discovery target), while m_t is the
*given* completeness threshold the form is centred on.

LAW_CONSTANTS — the formula's defining coefficients (a_t, b)
-----------------------------------------------------------
    a_t = 2.4281   (log-seismicity at the threshold m_t)
    b   = 1.0191   (Gutenberg-Richter slope)

Per the paper these are "the parameters" of Eq. 2 (PDF p. 8 line 233). For this
benchmark they are obtained by OLS on the USGS NEIC 1980-2024 training split
(M < 7.5), threshold-centred form: a_t = 2.4281, b = 1.0191 (reproducible from
data/train.csv). A coefficient that is fit and is the formula's characteristic
parameter is LAW — never a "given".

OTHER_CONSTANTS — the given completeness threshold (pivot)
----------------------------------------------------------
m_t = 5.6: the catalog completeness threshold the form is centred on — a fixed
  structural number the formula consumes, NOT a defining coefficient (same role
  as the log-P "pivot" in a Cepheid PLZ relation). Kagan (2010) PDF p. 8 uses
  m_t = 5.8 for the 1977-2008 CMT catalog and m_t = 5.6 for the 1982-2008
  catalog (extended threshold). The benchmark data was aggregated from USGS
  NEIC events with Mc ≈ 5.6 (gr_aggregate.py), matching the lower CMT
  completeness threshold stated on PDF p. 8. The paper names only a_t and b as
  the formula's "parameters" (kagan_2010.txt L233: "at and b are parameters");
  m_t is the catalog completeness threshold (kagan_2010.txt L216-217: "for the
  1982-2008 catalog it is mt = 5.6"), a given the form is centred on — not a
  discovery target. Declared in OTHER_CONSTANTS; predict reads it from the dict.

Type designation: Type I — single globally-aggregated catalog, no cluster
structure. LOCAL_FITTABLE = {}.

Column mapping:
    M_threshold (col 1) → m  (moment-magnitude threshold)
    log10_N     (col 0) → log10 N(m)

Caveats:
- The threshold-centred form a_t − b*(m − m_t) is algebraically equivalent to
  log10 N = a − b*M with a = a_t + b*m_t. The distinction matters only for
  interpretation of a_t (activity at threshold vs. intercept at M = 0).
- The paper argues the true β (b = 1.5β) is likely 0.5 (critical branching),
  implying b ≈ 0.75, whereas the empirically measured b ≈ 0.95–1.0 for global
  shallow earthquakes. Here b is fit on the training split (b = 1.0191) and is
  a LAW coefficient; the given completeness threshold m_t = 5.6 is the only
  OTHER_CONSTANTS entry.
- Kagan (2010) also presents the tapered G-R (TGR) extension (Eqs. 5–6,
  PDF p. 9), but the plain G-R (Eq. 2) is benchmarked here because the data
  is restricted to M ≤ 9.1 where the taper correction is small.
"""

import numpy as np

USED_INPUTS = ["M_threshold"]
PAPER_REF = "summary_formula_kagan_2010.md"
EQUATION_LOC = "Kagan 2010, Eq. 2, PDF p. 8"

# === LAW_CONSTANTS — Eq. 2's defining coefficients (PDF p. 8: "a_t and b are parameters") ===
LAW_CONSTANTS = {
    "a_t": 2.4281,   # log-seismicity at m_t; OLS on USGS NEIC train (M < 7.5)
    "b":   1.0191,   # Gutenberg-Richter slope; OLS on USGS NEIC train (M < 7.5)
}
# === OTHER_CONSTANTS — given completeness threshold (pivot the form is centred on) ===
OTHER_CONSTANTS = {
    "m_t": 5.6,   # catalog completeness threshold; Kagan 2010 PDF p. 8
                  # (m_t = 5.6 for the 1982-2008 extended CMT threshold)
}
LOCAL_FITTABLE = {}   # Type I


def predict(X: np.ndarray, a_t: float, b: float) -> np.ndarray:
    """log10 N(m) = a_t - b * (m - m_t).

    X: (n, 1) — column M_threshold.
    a_t: log-seismicity at m_t (LAW_CONSTANTS).
    b: G-R slope (LAW_CONSTANTS).
    Both arrive via predict(X, **LAW_CONSTANTS); the given threshold m_t is
    read from OTHER_CONSTANTS.
    """
    m_t = OTHER_CONSTANTS["m_t"]
    m = X[:, 0]
    return a_t - b * (m - m_t)
