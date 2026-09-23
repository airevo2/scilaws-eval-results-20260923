"""Gabaix-Lasry-Lions-Moll (2016) Friction-1 stationary tail exponent.

Gabaix, Lasry, Lions & Moll (2016), Econometrica 84(6):2071-2111,
Eq. (5) (PDF p. 11), gives the stationary Pareto upper-tail exponent of
the income process dx_it = mu dt + sigma dW_it with Poisson death and
reinjection at rate delta (Friction 1):

    alpha = (-mu + sqrt(mu**2 + 2 * sigma**2 * delta)) / sigma**2

The implied Pareto upper-tail then satisfies the same log-log
self-similarity as in Atkinson 2011:

    log_top_share = (1 - 1/alpha) * log_p_above + C

gabaix_2016 therefore differs from atkinson_2011 by **parameterising**
alpha from underlying income-process dynamics (mu, sigma) and the
demographic Poisson rate (delta), instead of fitting alpha directly.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
- delta = 1/30 yr^-1
  Gabaix et al. (2016) PDF p. 21 calibrate delta = 1/30 (= 30-year average
  work life). This is the paper's universal demographic constant.

OTHER_CONSTANTS — universal physics / unit factors
--------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster
----------------------------
- mu    : Ito drift of log income (country-specific). Init at 0.
- sigma : diffusion coefficient (country-specific). Init at 0.094
          (Gabaix 2016 PDF p. 21 US calibration).
- C     : intercept absorbing the country's tail-truncation scale (not
          part of Gabaix's dynamic model, but required to fit the
          observed log_top_share level — log-log slope is structural,
          intercept depends on currency/normalisation conventions).

fit() minimises sum((log_top_share_pred - log_top_share_obs)^2) over
(mu, sigma, C) via Nelder-Mead. (mu, sigma) jointly produce alpha; the
loss is a true regression residual, well-conditioned with 35 fit rows.
"""

import numpy as np
from scipy.optimize import minimize

USED_INPUTS = ["log_p_above"]
PAPER_REF = "summary_formula_gabaix_2016.md"
EQUATION_LOC = (
    "Gabaix-Lasry-Lions-Moll 2016 Eq. (5), PDF p. 11; "
    "delta = 1/30 calibration PDF p. 21"
)

LAW_CONSTANTS = {
    "delta": 1.0 / 30.0,                 # 30-year work life (PDF p. 21)
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "mu":    {"init": 0.0},              # scalar = single-start; neutral, no prior
    "sigma": {"init": 0.094},            # scalar = single-start; paper US calibration
    "C":     {"init": 0.0},              # scalar = single-start; intercept
}


def _alpha_from(mu: float, sigma: float, delta: float) -> float:
    """Friction-1 stationary alpha (Gabaix 2016 Eq. 5)."""
    s2 = sigma * sigma
    return (-mu + np.sqrt(mu * mu + 2.0 * s2 * delta)) / s2


def fit(X_fit: np.ndarray, y_fit: np.ndarray, delta: float) -> dict:
    """Fit (mu, sigma, C) by minimising squared log_top_share residual."""
    log_p = np.asarray(X_fit[:, 0], dtype=float)
    y_obs = np.asarray(y_fit, dtype=float)

    def loss(params):
        mu, sigma, C = params
        if sigma <= 1e-6:
            return 1e10
        alpha = _alpha_from(mu, sigma, delta)
        if alpha <= 1.0:                          # Pareto requires α > 1
            return 1e10
        slope = 1.0 - 1.0 / alpha
        y_pred = slope * log_p + C
        return float(np.sum((y_pred - y_obs) ** 2))

    x0 = np.array([
        LOCAL_FITTABLE["mu"]["init"],
        LOCAL_FITTABLE["sigma"]["init"],
        LOCAL_FITTABLE["C"]["init"],
    ])
    res = minimize(
        loss, x0=x0, method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 3000},
    )
    return {
        "mu":    float(res.x[0]),
        "sigma": float(res.x[1]),
        "C":     float(res.x[2]),
    }


def predict(X: np.ndarray, delta: float, mu: float, sigma: float, C: float) -> np.ndarray:
    """Apply Pareto self-similarity with alpha derived from (mu, sigma, delta)."""
    log_p = np.asarray(X[:, 0], dtype=float)
    alpha = _alpha_from(mu, sigma, delta)
    slope = 1.0 - 1.0 / alpha
    return slope * log_p + C
