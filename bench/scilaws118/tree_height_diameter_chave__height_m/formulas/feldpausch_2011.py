"""Feldpausch et al. (2011) pantropical log-log H:D allometric equation — height_m.

Feldpausch, T. R., Banin, L., Phillips, O. L., et al. (2011). Height-diameter
allometry of tropical forest trees. *Biogeosciences* 8, 1081–1106.
DOI:10.5194/bg-8-1081-2011. CC Attribution 3.0 License.

## Pantropical geography-only working equation (Appendix A, Table A2; PDF p. 20)

    ln(H) = β₀ + β₁·ln(D*)

where D* is diameter in centimetres, β₀ = 1.2229, β₁ = 0.5320.

Equivalently:

    H = exp(1.2229 + 0.5320·ln(D_cm))  = exp(1.2229) × D_cm^0.5320

## Pantropical environment-structure working equation (Appendix A, Table A1; PDF p. 20)

    ln(H) = 0.4893 + 0.5296·ln(D_cm) + 0.0098·A + 0.0034·PV − 0.0632·SD + 0.0204·TA

where per-cluster covariates are:
  A  — stand basal area (m² ha⁻¹)
  PV — precipitation coefficient of variation (fraction)
  SD — dry season length (months with < 0.1 m rainfall)
  TA — mean annual temperature (°C)

In this benchmark task only D (DBH, cm) is present in the raw CSV; A, PV, SD,
TA are not available. The benchmark therefore uses the geography-only form
(β₀ = 1.2229, β₁ = 0.5320) with a LOCAL_FITTABLE per-cluster offset δ to
absorb site-level H:D variation.

S2-P12 C10a (2026-05-28): the environment-structure coefficients (Table A1:
beta0_env, beta1_env, coeff_A, coeff_PV, coeff_SD, coeff_TA) have been moved
to OTHER_CONSTANTS because they are not referenced inside predict() — the
covariates A, PV, SD, TA are absent from the released CSV. Only beta0_geo and
beta1_geo (Table A2, the geography-only form) remain in LAW_CONSTANTS as they
are the only constants used by predict().

LAW_CONSTANTS — paper-published, frozen; used in predict() (Feldpausch 2011)
-----------------------------------------------------------------------------
Geography-only pantropical equation (Table A2, PDF p. 20):
- beta0_geo  : 1.2229  — Table A2, PDF p. 20; pantropical ln-intercept (D in cm)
- beta1_geo  : 0.5320  — Table A2, PDF p. 20; pantropical log-log D slope

OTHER_CONSTANTS — paper-published but NOT used in predict() (S2-P12 C10a)
--------------------------------------------------------------------------
Environment-structure pantropical equation (Table A1, PDF p. 20); covariates
A, PV, SD, TA absent from raw CSV:
- beta0_env  : 0.4893  — Table A1, PDF p. 20; ln-intercept
- beta1_env  : 0.5296  — Table A1, PDF p. 20; log-log D slope
- coeff_A    : 0.0098  — Table A1, PDF p. 20; stand basal area coefficient
- coeff_PV   : 0.0034  — Table A1, PDF p. 20; precip. CV coefficient
- coeff_SD   : -0.0632 — Table A1, PDF p. 20; dry season length coefficient
- coeff_TA   : 0.0204  — Table A1, PDF p. 20; mean annual temperature coefficient

LOCAL_FITTABLE — per-cluster, fitted by fit() via OLS in ln-space
-----------------------------------------------------------------
- delta : additive ln-space intercept offset (dimensionless).
          Captures per-site H:D deviation from the pantropical law-constant
          prediction (the role filled by random effect U₀p in Eqns 5–6 of the
          paper). init = 0.0.
"""

import numpy as np

USED_INPUTS = ["DBH_cm"]
PAPER_REF = "summary_formula_feldpausch_2011.md"
EQUATION_LOC = (
    "Feldpausch et al. (2011) Appendix A, Table A2, PDF p. 20: "
    "ln(H) = 1.2229 + 0.5320*ln(D_cm)  [pantropical, geography-only]. "
    "Environment-structure form: Table A1, PDF p. 20."
)

LAW_CONSTANTS = {
    # Geography-only pantropical (Table A2, PDF p. 20) — used in predict()
    "beta0_geo":  1.2229,   # Feldpausch 2011 Table A2, PDF p. 20
    "beta1_geo":  0.5320,   # Feldpausch 2011 Table A2, PDF p. 20
}
OTHER_CONSTANTS = {
    # Environment-structure pantropical coefficients (Table A1, PDF p. 20).
    # NOT used in predict() — covariates A, PV, SD, TA absent from raw CSV.
    # S2-P12 C10a: moved from LAW_CONSTANTS because these keys are never
    # referenced inside predict() or _ln_H().
    "beta0_env":  0.4893,   # Feldpausch 2011 Table A1, PDF p. 20
    "beta1_env":  0.5296,   # Feldpausch 2011 Table A1, PDF p. 20
    "coeff_A":    0.0098,   # Feldpausch 2011 Table A1, PDF p. 20 (stand basal area)
    "coeff_PV":   0.0034,   # Feldpausch 2011 Table A1, PDF p. 20 (precip. CV)
    "coeff_SD":  -0.0632,   # Feldpausch 2011 Table A1, PDF p. 20 (dry season length)
    "coeff_TA":   0.0204,   # Feldpausch 2011 Table A1, PDF p. 20 (mean annual temp.)
}
LOCAL_FITTABLE = {
    "delta": {"init": 0.0},  # per-cluster ln-space intercept offset
}


def _ln_H(D_cm, delta):
    """ln(H) from Feldpausch 2011 geography-only pantropical Eqn (Table A2)
    plus per-cluster offset delta."""
    return (
        LAW_CONSTANTS["beta0_geo"]
        + LAW_CONSTANTS["beta1_geo"] * np.log(D_cm)
        + delta
    )


def fit(X_fit: np.ndarray, y_fit: np.ndarray, **law_constants) -> dict:
    """OLS fit of the per-cluster ln-space offset delta.

    Law constants β₀_geo = 1.2229, β₁_geo = 0.5320 are frozen (received via
    **law_constants per the v2 harness contract; they are not re-fit).
    D_cm = X_fit[:, 0].

    Start: delta0 = 0.0 (published pantropical equation).
    """
    D_cm = np.asarray(X_fit[:, 0], dtype=float)
    H_obs = np.asarray(y_fit, dtype=float)

    lnH_obs = np.log(np.clip(H_obs, 1e-6, None))
    lnH_pred0 = _ln_H(D_cm, delta=0.0)
    delta_hat = float(np.mean(lnH_obs - lnH_pred0))

    return {"delta": delta_hat}


def predict(X: np.ndarray, **params) -> np.ndarray:
    """Total tree height H (m) from trunk diameter D (cm).

    X: (n, 1) — column [DBH_cm].
    params contains LAW_CONSTANTS (beta0_geo, beta1_geo) plus LOCAL (delta).
    Uses the pantropical geography-only working equation from Feldpausch 2011
    Table A2 (PDF p. 20), plus per-cluster intercept offset delta.
    """
    D_cm = np.asarray(X[:, 0], dtype=float)
    delta = float(params.get("delta", 0.0))
    return np.exp(_ln_H(D_cm, delta=delta))
