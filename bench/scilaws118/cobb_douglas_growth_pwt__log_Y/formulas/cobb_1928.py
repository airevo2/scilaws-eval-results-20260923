"""Classic Cobb-Douglas production function — log income per worker.

Cobb, C. W. and Douglas, P. H. (1928). A theory of production.
*American Economic Review*, 18(1 Supplement):139-165.
JSTOR stable URL: http://www.jstor.org/stable/1811556

The general parametric form (PDF p. 15 / journal p. 152, §7):

    P = b * L^k * C^(1-k)

subject to constant returns to scale (CRS): exponents on L and C sum to 1.

Applied to the PWT 11.0 benchmark, the formula becomes (taking logs):

    log_Y = log(cgdpo/emp) = log_A + alpha*log(cn) + (1-alpha)*log(emp)

where log_A = ln(A) is the (per-cluster) log total factor productivity,
alpha is the capital elasticity, and (1-alpha) is the labor elasticity.
CRS is imposed structurally by using a single elasticity parameter alpha
with the labor elasticity fixed as its complement 1-alpha.

Symbol mapping: 1928 (P, C, L, k, b) -> PWT benchmark (cgdpo/emp, cn, emp, alpha, A).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. Cobb and Douglas (1928) report k = 3/4 as the best-fit value for
US manufacturing 1899-1922 (PDF p. 14 / journal p. 151, §6), but they
explicitly warn that k "may have to be changed" when applied to
different indices or time periods (PDF p. 15 / journal p. 152). For the
PWT cross-country benchmark, k (here alpha) is a LOCAL_FITTABLE
per-country parameter — no universal numerical value is published for
cross-country data.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The CRS constraint (alpha + (1-alpha) = 1) is an algebraic
identity, not a tunable constant.

LOCAL_FITTABLE — per-cluster, computed by fit() via OLS in log-space
---------------------------------------------------------------------
- log_A : log total factor productivity (log dimensionless). Fitted as
          the intercept in log-log OLS. Per-country in PWT benchmark.
- alpha  : capital elasticity of output (dimensionless). Original 1928 US
           manufacturing value k = 1/4 for capital (Cobb and Douglas 1928,
           PDF p. 14 / journal p. 151, §6); modern cross-country prior ~1/3.
           Fitted per country cluster as the unconstrained OLS slope (the
           economic 0<alpha<1 range is an interpretation, not a fit clamp).
init = None on both: fit() builds a data-derived OLS start.
"""

import numpy as np

USED_INPUTS = ["cn", "emp"]
PAPER_REF = "summary_formula_cobb_1928.md"
EQUATION_LOC = (
    "Cobb and Douglas (1928) §7, PDF p. 15 / journal p. 152: "
    "P = b * L^k * C^(1-k) (general parametric CRS form); "
    "log-linear version: log(Y/L) = log_A + alpha*log(K) + (1-alpha)*log(L)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "log_A": {"init": None},
    "alpha": {"init": None},
}


def _log_y(log_K, log_L, log_A, alpha):
    """log(Y/L) = log_A + alpha*log(K) + (1-alpha)*log(L).

    Since the target is log(cgdpo/emp) = log(Y) - log(L), we expand:
    log(Y/L) = log_A + alpha*log(K) + (1-alpha)*log(L) - log(L)
              = log_A + alpha*log(K) - alpha*log(L)
              = log_A + alpha*(log(K) - log(L))
    This is equivalent and numerically stable.
    """
    return log_A + alpha * (log_K - log_L)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """OLS in log-space for the CRS Cobb-Douglas form.

    Regresses log_Y = log_A + alpha*(log_K - log_L) via ordinary
    least squares. The single regressor is log(K/L) = log(cn/emp).

    The prep_data.py interleaved within-cluster split ensures the fit
    window spans the full year range (every other year), preserving
    cross-temporal variation in log(K/L) and making OLS well-conditioned.
    """
    cn  = np.asarray(X_fit[:, 0], dtype=float)
    emp = np.asarray(X_fit[:, 1], dtype=float)
    y   = np.asarray(y_fit, dtype=float)

    # Guard against non-positive values before taking logs.
    valid = (cn > 0) & (emp > 0) & np.isfinite(y)
    if valid.sum() < 2:
        return {"log_A": 0.0, "alpha": 1.0 / 3.0}

    log_K = np.log(cn[valid])
    log_L = np.log(emp[valid])
    x_reg = log_K - log_L   # log(K/L) — single regressor under CRS

    # OLS: design matrix with intercept and single regressor.
    A_mat = np.column_stack([np.ones(valid.sum()), x_reg])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, y[valid], rcond=None)
        log_A_hat = float(coeffs[0])
        alpha_hat = float(coeffs[1])
        # No clamp: alpha is the unconstrained OLS slope, exactly as Cobb &
        # Douglas (1928) fit by least squares. The economic range 0<alpha<1 is
        # an INTERPRETATION of the fitted elasticity, not a fitting constraint.
        # A per-cluster slope >1 or <0 is a valid LS estimate (capital deepening
        # / collinearity) that the intercept log_A absorbs; clamping alpha
        # WITHOUT re-fitting log_A corrupts predict() on held-out rows.
        if not np.isfinite(log_A_hat):
            log_A_hat = 0.0
        return {"log_A": log_A_hat, "alpha": alpha_hat}
    except Exception:                               # noqa: BLE001
        return {"log_A": 0.0, "alpha": 1.0 / 3.0}


def predict(X: np.ndarray, log_A: float, alpha: float) -> np.ndarray:
    """Classic CRS Cobb-Douglas: log(Y/L) = log_A + alpha*(log(K) - log(L)).

    X: (n, 2) — columns [cn, emp].
    Returns log(cgdpo/emp) predictions.
    """
    cn  = np.asarray(X[:, 0], dtype=float)
    emp = np.asarray(X[:, 1], dtype=float)

    # Protect against zero / negative inputs.
    safe_K = np.where(cn  > 0, cn,  1e-30)
    safe_L = np.where(emp > 0, emp, 1e-30)

    return _log_y(np.log(safe_K), np.log(safe_L), log_A, alpha)
