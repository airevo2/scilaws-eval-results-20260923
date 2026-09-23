"""Niklas & Enquist (2001) 3/4-power allometric scaling — a_lf.

Niklas, K. J. & Enquist, B. J. (2001). Invariant scaling relationships for
interspecific plant biomass production rates and body size. Proceedings of
the National Academy of Sciences USA 98(5): 2922-2927.
DOI: 10.1073/pnas.041590298. PDF at reference/niklas_enquist_2001.pdf.

Core formula (Results, PDF p. 2; Fig. 3A):

    log10(M_p) = 0.762 * log10(M_n) + 0.192

i.e.

    M_p = 10^0.192 * M_n^0.762 ~ 1.556 * M_n^0.762

where M_p = photosynthetic (foliage) biomass, M_n = non-photosynthetic
(stem + root) biomass.

Published RMA regression: alpha_RMA = 0.762 +/- 0.023, n=292, r2=0.747,
F=856.2, P<0.0001 (PDF p. 2, Results; Fig. 3A).

Type II — per-species normalization coefficient k is LOCAL_FITTABLE.

Column mapping (paper -> CSV):
  M_n (non-photosynthetic biomass, kg) -> a_ssbh (sapwood area at BH, m2)
    [sapwood area is a hydraulic proxy for non-photosynthetic structural
    tissue per WBE / Enquist framework]
  M_p (photosynthetic / foliage biomass, kg) -> a_lf (leaf area, m2)
    [leaf area is a proxy for photosynthetic tissue; differs from leaf mass
    by species-specific SLA]

LAW_CONSTANTS:
  alpha_RMA = 0.762: RMA scaling exponent, PDF p. 2 Results, Fig. 3A;
    95% CI +/- 0.023, n=292, r2=0.747. Theoretically predicted as exactly
    3/4 by WBE fractal-network theory (West et al. 1999, Nature 400:664-667,
    ref. 16 in Niklas-Enquist 2001).

OTHER_CONSTANTS: (none)

LOCAL_FITTABLE:
  k: per-species normalization coefficient (positive, units m2 / m^(2*0.762)
     ~ m^(0.476)). Differences in k across species reflect species-level
     differences in SLA, wood density, bark fraction, and hydraulic
     architecture. The paper's published intercept (10^0.192 ~ 1.556) applies
     to interspecific pooling of biomass data in kg; it does not carry over
     numerically to the area-on-area mapping in BAAD.
     init = None: closed-form OLS fit in log10 space with fixed alpha_RMA.

Caveat: The BAAD columns are leaf area (m2) rather than leaf mass (kg) and
sapwood area (m2) rather than non-photosynthetic mass (kg). The paper's
intercept (0.192 in log10 space) is unit-dependent and not transferable;
only the exponent (0.762) is unit-invariant and directly applicable.
"""

import numpy as np

USED_INPUTS = ["a_ssbh"]
PAPER_REF = "summary_formula_niklas_2001.md"
EQUATION_LOC = (
    "Niklas & Enquist (2001) PDF p. 2 Results, Fig. 3A: "
    "log10(M_p) = 0.762 * log10(M_n) + 0.192; "
    "alpha_RMA = 0.762 +/- 0.023, n=292, r2=0.747."
)

LAW_CONSTANTS = {
    "alpha_RMA": 0.762,   # RMA scaling exponent, PDF p. 2 Results / Fig. 3A
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "k": {"init": None},  # per-species normalization coefficient; closed-form fit
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray,
        alpha_RMA: float) -> dict:
    """Fit per-species k via OLS in log10-space with alpha_RMA fixed.

    log10(a_lf) = log10(k) + alpha_RMA * log10(a_ssbh)
    => log10(k) = mean(log10(a_lf) - alpha_RMA * log10(a_ssbh))

    X_fit[:, 0] = a_ssbh. Only strictly positive rows are used.
    """
    a_ssbh = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    mask = (a_ssbh > 0) & (y > 0) & np.isfinite(a_ssbh) & np.isfinite(y)
    if mask.sum() < 1:
        return {"k": 1.0}

    log_x = np.log10(a_ssbh[mask])
    log_y = np.log10(y[mask])
    log_k = float(np.mean(log_y - alpha_RMA * log_x))
    k = 10.0 ** log_k
    if not np.isfinite(k) or k <= 0:
        k = 1.0
    return {"k": k}


def predict(X: np.ndarray, k: float,
            alpha_RMA: float) -> np.ndarray:
    """a_lf = k * a_ssbh^alpha_RMA.

    X: (n, 1) — column [a_ssbh].
    k: per-species normalization coefficient (LOCAL_FITTABLE).
    alpha_RMA: 3/4-power scaling exponent (LAW_CONSTANT, 0.762).
    """
    a_ssbh = np.asarray(X[:, 0], dtype=float)
    return k * np.power(np.clip(a_ssbh, 1e-30, None), alpha_RMA)
