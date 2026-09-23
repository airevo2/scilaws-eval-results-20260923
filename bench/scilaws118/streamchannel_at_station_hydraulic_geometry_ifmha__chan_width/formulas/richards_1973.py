"""Richards (1973) log-quadratic at-a-station hydraulic geometry — chan_width.

Richards, K. S. (1973). Hydraulic geometry and channel roughness; a non-linear
system. *American Journal of Science*, 273(10), 877-896.
DOI: 10.2475/ajs.273.10.877.

Richards (1973) observed that the simple Leopold & Maddock (1953) power law
w = a*Q^b often fails to capture the curvature visible on at-a-station
hydraulic-geometry plots: because depth and velocity are functions of channel
roughness and roughness varies non-uniformly with stage, the log-log
width/depth/velocity-discharge relations are frequently better described by a
second-degree polynomial (a parabola in log space) than by a straight line.
This is the canonical "non-linear" / log-quadratic extension of at-a-station
hydraulic geometry:

    log(w) = p0 + p1*log(Q) + p2*(log(Q))^2

Equivalent multiplicative form:
    w = exp(p0) * Q^(p1 + p2*log(Q))

The effective instantaneous width exponent at discharge Q is
    b_eff(Q) = p1 + 2*p2*log(Q),
so p2 = 0 recovers the constant-exponent Leopold-Maddock power law exactly
(this model NESTS Leopold-Maddock). For channels with irregular cross-sections
or a sand/gravel regime boundary, p2 != 0 and the quadratic improves the fit.

Source-attribution note (anti-fabrication)
-------------------------------------------
The Richards (1973) primary article (Am. J. Sci.) is paywalled, so no PDF is
shipped (`reference_pdf: null`); the functional form is verified from two
open-access secondary sources that explicitly attribute the log-quadratic
at-a-station model to Richards (1973) — see
`summary_formula_richards_1973.md` and `reference/singh_2003_hydraulic_geometry_theories.pdf`:
  * Singh, V. P. (2003), Int. J. Sediment Research 18(3):196-218, p.198:
    "Richards (1973, 1976) has reasoned that ... the power function model ...
    will not reflect the true hydraulic nature ... He then proposed a model
    for describing the nonuniform variation ...".
  * Said (2005), J. Applied Sciences 5(9):1606-1612: "Richards [1973] pointed
    out that for many rivers, the scatter of points on hydraulic geometry
    diagrams is often better fit by polynomial relationships (fit to the
    logarithms of the variables) rather than by linear relationships."
  * Knighton (1979) found Richards' log-quadratic model significantly better
    than the Leopold-Maddock power function for some stations.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The structural form log(w) = p0 + p1*log(Q) + p2*(log(Q))^2 is the
scientific claim. All three coefficients are per-station empirical parameters
fitted from current-meter records; no universal numerical constant appears.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
None. The `2` exponent on log(Q) is a structural power of the polynomial form
(inline), not a declared constant.

LOCAL_FITTABLE — per-cluster (site_no), fitted via OLS on log-transformed data
------------------------------------------------------------------------------
- p0 : log-space intercept (log(a) in the constant-slope case when p2=0)
- p1 : linear log-slope (analogous to Leopold-Maddock b when p2~=0)
- p2 : curvature coefficient (zero for perfectly log-linear channels; non-zero
       for channels with curvature in log-log space)
"""

import numpy as np

USED_INPUTS = ["chan_discharge"]
PAPER_REF = "summary_formula_richards_1973.md"
EQUATION_LOC = (
    "Richards (1973) Am. J. Sci. 273(10):877-896 — log-quadratic (non-linear) "
    "at-a-station hydraulic geometry log(w)=p0+p1*log(Q)+p2*(log(Q))^2; primary "
    "paywalled, form verified via Singh (2003) IJSR 18(3):196-218 p.198 and "
    "Said (2005) J. Appl. Sci. 5(9):1606-1612 (see summary_formula_richards_1973.md)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "p0": {"init": None},
    "p1": {"init": None},
    "p2": {"init": None},
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Fit log(w) = p0 + p1*log(Q) + p2*(log(Q))^2 via OLS.

    Parameters
    ----------
    X_fit : (n, 1) array — column [chan_discharge] in cfs.
    y_fit : (n,) array — chan_width in ft.
    """
    Q = np.asarray(X_fit[:, 0], dtype=float)
    w = np.asarray(y_fit, dtype=float)

    mask = (Q > 0.0) & (w > 0.0)
    if mask.sum() >= 3:
        logQ = np.log(Q[mask])
        logw = np.log(w[mask])
        A = np.column_stack([np.ones(mask.sum()), logQ, logQ ** 2])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, logw, rcond=None)
            p0, p1, p2 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
            if all(np.isfinite([p0, p1, p2])):
                return {"p0": p0, "p1": p1, "p2": p2}
        except Exception:  # noqa: BLE001
            pass

    # Fallback: use Leopold-Maddock linear fit (p2=0)
    if mask.sum() >= 2:
        logQ = np.log(Q[mask])
        logw = np.log(w[mask])
        A = np.column_stack([np.ones(mask.sum()), logQ])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, logw, rcond=None)
            p0_lin, p1_lin = float(coeffs[0]), float(coeffs[1])
            if np.isfinite(p0_lin) and np.isfinite(p1_lin):
                return {"p0": p0_lin, "p1": p1_lin, "p2": 0.0}
        except Exception:  # noqa: BLE001
            pass

    # Ultimate fallback: Leopold & Maddock average b=0.26
    b_fb = 0.26
    Q_ref = float(np.nanmedian(Q[Q > 0])) if (Q > 0).any() else 1.0
    w_ref = float(np.nanmedian(w[w > 0])) if (w > 0).any() else 1.0
    a_fb = w_ref / (Q_ref ** b_fb) if Q_ref > 0 else 1.0
    return {"p0": float(np.log(a_fb)), "p1": b_fb, "p2": 0.0}


def predict(X: np.ndarray, p0: float, p1: float, p2: float) -> np.ndarray:
    """Log-log quadratic width: w = exp(p0 + p1*log(Q) + p2*(log(Q))^2).

    Parameters
    ----------
    X  : (n, 1) array — column [chan_discharge] in cfs.
    p0 : log-space intercept.
    p1 : linear log-slope.
    p2 : curvature (quadratic log-log coefficient).

    Returns
    -------
    (n,) array of predicted chan_width in ft.
    """
    Q = np.asarray(X[:, 0], dtype=float)
    # Guard against Q <= 0 (should not occur in cleaned data)
    Q_safe = np.where(Q > 0, Q, np.nan)
    logQ = np.log(Q_safe)
    log_w_pred = p0 + p1 * logQ + p2 * logQ ** 2
    return np.exp(log_w_pred)
