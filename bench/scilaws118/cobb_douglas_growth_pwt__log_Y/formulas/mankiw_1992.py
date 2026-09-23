"""Mankiw-Romer-Weil (1992) augmented Cobb-Douglas — log income per worker.

Mankiw, N. G., Romer, D., and Weil, D. N. (1992). A contribution to the
empirics of economic growth. *Quarterly Journal of Economics*,
107(2):407-437. DOI: 10.2307/2118477.

The augmented production function (eq. 8, PDF p. 10 / journal p. 416):

    Y(t) = K(t)^alpha * H(t)^beta * (A(t)*L(t))^(1-alpha-beta)

where H is the stock of human capital, and alpha + beta < 1 (decreasing
returns to all reproducible capital, required for a steady state to exist).

Applied to PWT 11.0, taking logs and rearranging for target log(Y/L):

    log(Y/L) = log_A + alpha*log(K) + beta*log(H) + (1-alpha-beta)*log(L)
             = log_A + alpha*log(cn) + beta*log(hc*emp) + (1-alpha-beta)*log(emp)

where H = hc * emp (human capital index times persons engaged).

Expanding further:
    log(Y/L) = log_A + alpha*log(K) + beta*log(hc) + log(emp) - log(emp)
             -- note: the (1-alpha-beta)*log(emp) - log(emp) = -(alpha+beta)*log(emp)

After simplification:
    log(Y/L) = log_A + alpha*log(cn) + beta*log(hc*emp) + (1-alpha-beta)*log(emp)
    Since the target is log(cgdpo/emp) = log(cgdpo) - log(emp), and:
        log(cgdpo) = log_A*_production_level + alpha*log(K) + beta*log(H) + (1-alpha-beta)*log(A*L)
    We write it as the reduced-form:
        log(Y/L) = log_A + alpha*(log(cn) - log(emp)) + beta*log(hc)
    (derivation: factor out log(emp); the (1-alpha-beta)*log(L) and the -log(L) from
     Y/L = Y - log(L) combine to -(alpha+beta)*log(L); then log(H) = log(hc*emp) =
     log(hc)+log(emp), so beta*log(hc*emp) - (alpha+beta)*log(emp) = beta*log(hc) - alpha*log(emp),
     which with alpha*(log(K)-log(emp)) term gives: log_A + alpha*log(K/L) + beta*log(hc).)

Final compact form used in fit() and predict():
    log(Y/L) = log_A + alpha*log(cn/emp) + beta*log(hc)

This is the structural production-function form with PWT 11.0 columns.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. Mankiw, Romer, and Weil (1992) assume g + delta = 0.05 uniformly
across countries (stated at PDF p. 7 footnote 5 / text line 340:
"We assume that g + delta is 0.05"), but this assumption governs the
steady-state dilution-rate input ln(n+g+delta) in the reduced-form
specification — not the structural production-function form used here.
For the structural form, alpha and beta are LOCAL_FITTABLE (paper
reports alpha ~= 0.31, beta ~= 0.28 for the intermediate sample,
Table II, PDF p. 14 / journal p. 420, but these are cross-sectional OLS
estimates for the 1960-1985 Summers-Heston data, not universal
constants applicable to PWT 11.0 country clusters). g + delta = 0.05
is deliberately excluded from this structural-form formula; it enters
only the reduced steady-state form which is not the benchmark target.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster, computed by fit() via OLS in log-space
---------------------------------------------------------------------
- log_A  : log total factor productivity (log dimensionless). Fitted as
           the intercept in log-log OLS. Per-country in PWT benchmark.
- alpha  : physical capital elasticity of output (dimensionless). Paper
           prior ~1/3; empirical ~0.31 (Mankiw 1992 Table II intermediate
           sample). Fitted per cluster as the unconstrained OLS coefficient.
- beta   : human capital elasticity of output (dimensionless). Paper prior
           1/3-1/2; empirical ~0.28-0.30 (Mankiw 1992 Table II). Fitted per
           cluster as the unconstrained OLS coefficient (0<alpha+beta<1 is
           an economic interpretation, not a fit constraint).
init = None on all three: fit() builds a data-derived OLS start.
"""

import numpy as np

USED_INPUTS = ["cn", "emp", "hc"]
PAPER_REF = "summary_formula_mankiw_1992.md"
EQUATION_LOC = (
    "Mankiw, Romer, and Weil (1992) eq. (8), PDF p. 10 / journal p. 416: "
    "Y = K^alpha * H^beta * (A*L)^(1-alpha-beta); "
    "log-linear form for target log(Y/L): "
    "log(Y/L) = log_A + alpha*log(K/L) + beta*log(hc)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "log_A": {"init": None},
    "alpha": {"init": None},
    "beta":  {"init": None},
}


def _log_y(log_K, log_L, log_hc, log_A, alpha, beta):
    """log(Y/L) = log_A + alpha*log(K/L) + beta*log(hc).

    Derivation from the augmented production function Y = K^alpha * H^beta * (A*L)^(1-alpha-beta)
    with H = hc * L:
        log(Y) = log_A + alpha*log(K) + beta*(log(hc)+log(L)) + (1-alpha-beta)*log(L)
        log(Y/L) = log(Y) - log(L)
                 = log_A + alpha*log(K) + beta*log(hc) + (beta + 1 - alpha - beta)*log(L) - log(L)
                 = log_A + alpha*log(K) + beta*log(hc) - alpha*log(L)
                 = log_A + alpha*(log(K) - log(L)) + beta*log(hc)
    """
    return log_A + alpha * (log_K - log_L) + beta * log_hc


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """OLS in log-space for the MRW augmented Cobb-Douglas form.

    Regresses log_Y = log_A + alpha*log(K/L) + beta*log(hc) via ordinary
    least squares with two regressors — exactly as Mankiw, Romer & Weil
    (1992) estimate their log-linear specification by OLS.

    No constraints are imposed on alpha or beta. The economic conditions
    0<alpha, 0<beta, alpha+beta<1 (steady-state existence, MRW p. 416) are
    INTERPRETATIONS of the fitted elasticities, not fitting constraints.
    Clamping/rescaling the OLS coefficients — or pinning beta to a prior
    when log(hc) has low within-cluster variance — without re-deriving the
    intercept corrupts predict() on held-out rows; a per-cluster slope
    outside the economic range is a valid LS estimate that log_A absorbs.

    The prep_data.py interleaved within-cluster split spans the full year
    range in both fit and test windows, preserving cross-temporal variation
    in log(K/L) and log(hc).
    """
    cn  = np.asarray(X_fit[:, 0], dtype=float)
    emp = np.asarray(X_fit[:, 1], dtype=float)
    hc  = np.asarray(X_fit[:, 2], dtype=float)
    y   = np.asarray(y_fit, dtype=float)

    valid = (cn > 0) & (emp > 0) & (hc > 0) & np.isfinite(y)
    if valid.sum() < 3:
        return {"log_A": 0.0, "alpha": 1.0 / 3.0, "beta": 1.0 / 3.0}

    log_KL = np.log(cn[valid]) - np.log(emp[valid])   # log(K/L)
    log_hc = np.log(hc[valid])

    A_mat = np.column_stack([np.ones(valid.sum()), log_KL, log_hc])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, y[valid], rcond=None)
        log_A_hat = float(coeffs[0])
        alpha_hat = float(coeffs[1])
        beta_hat  = float(coeffs[2])
        if not np.isfinite(log_A_hat):
            log_A_hat = 0.0
        return {"log_A": log_A_hat, "alpha": alpha_hat, "beta": beta_hat}
    except Exception:                               # noqa: BLE001
        return {"log_A": 0.0, "alpha": 1.0 / 3.0, "beta": 1.0 / 3.0}


def predict(
    X: np.ndarray,
    log_A: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Augmented CRS Cobb-Douglas: log(Y/L) = log_A + alpha*log(K/L) + beta*log(hc).

    X: (n, 3) — columns [cn, emp, hc].
    Returns log(cgdpo/emp) predictions.
    """
    cn  = np.asarray(X[:, 0], dtype=float)
    emp = np.asarray(X[:, 1], dtype=float)
    hc  = np.asarray(X[:, 2], dtype=float)

    safe_K  = np.where(cn  > 0, cn,  1e-30)
    safe_L  = np.where(emp > 0, emp, 1e-30)
    safe_hc = np.where(hc  > 0, hc,  1e-30)

    return _log_y(
        np.log(safe_K),
        np.log(safe_L),
        np.log(safe_hc),
        log_A,
        alpha,
        beta,
    )
