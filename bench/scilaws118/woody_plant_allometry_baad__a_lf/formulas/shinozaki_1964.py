"""Shinozaki et al. (1964) Pipe Model Theory — a_lf.

Shinozaki, K., Yoda, K., Hozumi, K. & Kira, T. (1964). A quantitative
analysis of plant form -- the pipe model theory. I. Basic analyses.
Japanese Journal of Ecology 14(3): 97-105.

As reviewed and formalized in:
Lehnebach, R., Beyer, R., Letort, V. & Heuret, P. (2018). The pipe model
theory half a century on: a review. Annals of Botany 121(5): 773-795.
DOI: 10.1093/aob/mcx194. PDF at reference/lehnebach_2018.pdf.

Core formula (Lehnebach 2018, PDF p. 3-4; Shinozaki 1964 property 1):

    A_L = k_p * A_S

where A_L = leaf area (or mass) supported above a cross-section, A_S = sapwood
(conductive) cross-sectional area at the same cross-section, and k_p = the
"specific pipe length" (Shinozaki's original L notation) or leaf-area-to-sapwood
ratio [m2/m2], dimensionless.

The PMT predicts that leaf area is proportional (exponent = 1.0) to sapwood
cross-sectional area. The constant k_p varies across species, ontogenetic
stage, and environment (Lehnebach 2018 pp. 10, 16-17), so k_p is LOCAL_FITTABLE
per species cluster.

Type II — per-species k_p is LOCAL_FITTABLE.

Column mapping (paper -> CSV):
  A_S (conductive sapwood area) -> a_ssbh (sapwood cross-sectional area at
    breast height, m2). Note: BAAD uses sapwood area at BH, which overestimates
    effective conductive area relative to crown-base SA (Lehnebach 2018, p. 7).
  A_L (leaf area) -> a_lf (whole-plant leaf area, m2).

LAW_CONSTANTS: (none)
  The PMT as a functional form (linear proportionality) is the scientific claim;
  no published numerical constant is universally applicable across species or
  the BAAD dataset. The exponent is structurally fixed at 1.0 as an inline
  numeral (operator-definition constant, not a LAW_CONSTANT).

OTHER_CONSTANTS: (none)
  The exponent 1.0 in a_lf = k_p * a_ssbh^1.0 is a structural inline literal.

LOCAL_FITTABLE:
  k_p: per-species leaf-area-to-sapwood-area ratio [m2 leaf / m2 sapwood],
    physically dimensionless when both areas share the same unit (m2).
    Fit by closed-form OLS (single-predictor linear regression through origin
    in linear space).
    init = None: closed-form fit (k_p = mean(a_lf / a_ssbh) per cluster).

Caveat: The PMT predicts exponent = 1 (proportionality), while WBE / Niklas-
Enquist theory predicts exponent = 3/4 (power law). The PMT exponent is
theoretically applicable within a species across ontogeny (intraspecific),
while the 3/4-power scaling is interspecific. In BAAD, both baselines are
per-species fits, so the distinction matters primarily for the LAW_CONSTANTS
versus LOCAL_FITTABLE assignment.
"""

import numpy as np

USED_INPUTS = ["a_ssbh"]
PAPER_REF = "summary_supporting_lehnebach_2018.md"
EQUATION_LOC = (
    "Shinozaki et al. (1964) Pipe Model Theory, as reviewed in "
    "Lehnebach et al. (2018) Annals of Botany 121(5):773-795, "
    "PDF pp. 3-4, property 1: A_L = k_p * A_S "
    "(leaf area proportional to sapwood cross-sectional area)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "k_p": {"init": None},  # per-species A_L/A_S ratio; closed-form fit
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Fit per-species k_p via OLS through origin in linear space.

    A_L = k_p * A_S  =>  k_p = sum(A_S * A_L) / sum(A_S^2)
    (least-squares regression through the origin).

    X_fit[:, 0] = a_ssbh. Only strictly positive rows are used.
    """
    a_ssbh = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    mask = (a_ssbh > 0) & (y > 0) & np.isfinite(a_ssbh) & np.isfinite(y)
    if mask.sum() < 1:
        return {"k_p": 1.0}

    xs = a_ssbh[mask]
    ys = y[mask]
    # OLS through origin: k_p = sum(xs * ys) / sum(xs^2)
    k_p = float(np.sum(xs * ys) / np.sum(xs * xs))
    if not np.isfinite(k_p) or k_p <= 0:
        k_p = 1.0
    return {"k_p": k_p}


def predict(X: np.ndarray, k_p: float) -> np.ndarray:
    """a_lf = k_p * a_ssbh.

    X: (n, 1) — column [a_ssbh].
    k_p: per-species leaf-area-to-sapwood-area ratio (LOCAL_FITTABLE).
    """
    a_ssbh = np.asarray(X[:, 0], dtype=float)
    return k_p * np.clip(a_ssbh, 0.0, None)
