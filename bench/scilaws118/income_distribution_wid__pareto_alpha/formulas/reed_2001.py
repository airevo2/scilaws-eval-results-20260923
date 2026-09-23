"""Reed (2001) double-Pareto upper-tail exponent.

Reed (2001), Economics Letters 74(1):15-19, Eq. (A.4) (PDF p. 4) gives
the upper-tail Pareto exponent of the cross-sectional double-Pareto
distribution as the positive root of the characteristic quadratic:

    alpha = (-(mu - sigma**2 / 2)
            + sqrt((mu - sigma**2 / 2)**2 + 2 * sigma**2 * lam))
            / sigma**2

In Reed's model, individual incomes follow geometric Brownian motion
`dX = mu * X dt + sigma * X dw` with observation time T ~ Exp(lam).

The implied Pareto upper-tail then satisfies the same log-log
self-similarity as in Atkinson 2011 / Gabaix 2016:

    log_top_share = (1 - 1/alpha) * log_p_above + C

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. Reed (2001) does not publish numerical values for mu, sigma, or
lam — the paper applies the formula qualitatively to US 1998 male income
and US settlement-size data without reporting fitted coefficients
(PDF pp. 2-3, §3 final paragraph).

OTHER_CONSTANTS — universal physics / unit factors
--------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster
----------------------------
- mu    : log-income drift (GBM, country-specific). Init at 0.
- sigma : log-income diffusion (country-specific). Init at 0.1.
- lam   : observation-time exponential rate (country-specific). Init 0.05.
- C     : intercept absorbing tail-truncation scale.

Positivity constraints: sigma > 0, lam > 0.

Identifiability caveat
----------------------
4 unknowns (mu, sigma, lam, C) vs effectively (slope, intercept) — under-
determined by 2 d.o.f. for the (mu, sigma, lam) triple (one slope is
recoverable; one free degree per excess parameter). fit() picks the local
minimum closest to (mu=0, sigma=0.1, lam=0.05).
"""

import numpy as np
from scipy.optimize import minimize

USED_INPUTS = ["log_p_above"]
PAPER_REF = "summary_formula_reed_2001.md"
EQUATION_LOC = (
    "Reed 2001 Eq. (A.4), PDF p. 4 "
    "(double-Pareto upper-tail characteristic quadratic)"
)

LAW_CONSTANTS = {}                       # no paper-published values
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {                       # scalar init = single-start (Nelder-Mead)
    "mu":    {"init": 0.0},
    "sigma": {"init": 0.1},
    "lam":   {"init": 0.05},
    "C":     {"init": 0.0},
}


def _alpha_from(mu: float, sigma: float, lam: float) -> float:
    s2 = sigma * sigma
    m = mu - 0.5 * s2
    return (-m + np.sqrt(m * m + 2.0 * s2 * lam)) / s2


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Fit (mu, sigma, lam, C) by minimising squared log_top_share residual."""
    log_p = np.asarray(X_fit[:, 0], dtype=float)
    y_obs = np.asarray(y_fit, dtype=float)

    def loss(params):
        mu, sigma, lam, C = params
        if sigma <= 1e-6 or lam <= 1e-6:
            return 1e10
        alpha = _alpha_from(mu, sigma, lam)
        if alpha <= 1.0:                           # Pareto requires α > 1
            return 1e10
        slope = 1.0 - 1.0 / alpha
        y_pred = slope * log_p + C
        return float(np.sum((y_pred - y_obs) ** 2))

    x0 = np.array([
        LOCAL_FITTABLE["mu"]["init"],
        LOCAL_FITTABLE["sigma"]["init"],
        LOCAL_FITTABLE["lam"]["init"],
        LOCAL_FITTABLE["C"]["init"],
    ])
    res = minimize(
        loss, x0=x0, method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 4000},
    )
    return {
        "mu":    float(res.x[0]),
        "sigma": float(res.x[1]),
        "lam":   float(res.x[2]),
        "C":     float(res.x[3]),
    }


def predict(X: np.ndarray, mu: float, sigma: float, lam: float, C: float) -> np.ndarray:
    """Apply Pareto self-similarity with alpha derived from (mu, sigma, lam)."""
    log_p = np.asarray(X[:, 0], dtype=float)
    alpha = _alpha_from(mu, sigma, lam)
    slope = 1.0 - 1.0 / alpha
    return slope * log_p + C
