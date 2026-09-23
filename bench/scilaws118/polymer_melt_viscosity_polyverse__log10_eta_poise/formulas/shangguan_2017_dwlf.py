"""Developed WLF (DWLF) equation — log10(eta/poise).

Shangguan, Y. et al. (2017). New Insight into Time-Temperature Correlation
for Polymer Relaxations Ranging from Secondary Relaxation to Terminal Flow:
Application of a Universal and Developed WLF Equation. Polymers 9(11), 567.
DOI:10.3390/polym9110567. Open-access (MDPI, CC-BY-4.0).

DWLF equation (PDF p. 5, Eqs. 7–9):

    lg alpha_T = -C1' * (T - Ts) / (C2' + (T - Ts))
    C2' = Ts   (Eq. 8 — this constraint is the DWLF scientific claim)

Substituting C2' = Ts into the denominator:
    C2' + (T - Ts) = Ts + T - Ts = T

So:
    lg alpha_T = -C1' * (T_K - Ts_K) / T_K

Since alpha_T = eta(T) / eta(Ts):
    log10_eta(T) = log10_eta_ref - C1' * (T_K - Ts_K) / T_K

where:
    T_K          = temperature in K (input column Temperature_K)
    Ts_K         = reference temperature in K; here Ts = Tg (per-cluster
                   observed covariate, input column Tg_K)
    C1'          = per-cluster LOCAL_FITTABLE coefficient (dimensionless)
    log10_eta_ref = log10(eta) at T = Ts — per-cluster LOCAL_FITTABLE

Note: the DWLF formula structure (functional form with C2' = Ts_K)
is universal — it applies from secondary relaxations through glass
transition to terminal flow. The per-cluster C1' encodes polymer-specific
activation energy.

LAW_CONSTANTS — paper-published, frozen (the scientific claim of DWLF)
-----------------------------------------------------------------------
  None in the numeric sense. The structural constraint C2' = Ts is the
  invariant — it is hardcoded into the formula (denominator = T_K, not a
  free parameter). The physical claim is the functional FORM, not a number.

  Physical relationship: C1' = 0.434 * E_a / (R * Ts) (Eq. 7, PDF p. 5).
  In this implementation C1' is fitted directly from rheological data;
  E_a and R are not separated as distinct parameters.

OTHER_CONSTANTS
---------------
  (none; 273.15 offset applied in prep_data.py before Temperature_K is stored)

LOCAL_FITTABLE — per-cluster (per-polymer), fitted by fit()
-----------------------------------------------------------
  C1_prime      : DWLF activation-energy-like coefficient; dimensionless;
                  > 0. Physical range 3.5–43 across the polymers in the paper
                  (PDF pp. 6–15, Table 1). init = None (data-derived).

  log10_eta_ref : log10(eta_poise) at T = Ts (reference viscosity intercept).
                  Per-polymer. init = None (data-derived).
"""

import numpy as np
from scipy.optimize import least_squares

# NOTE: scipy.optimize is used for fit() only. predict() uses only numpy.

USED_INPUTS = ["Temperature_K", "Tg_K"]

PAPER_REF = "summary_formula_shangguan_2017.md"
EQUATION_LOC = (
    "Shangguan et al. (2017) Polymers 9(11):567, PDF p. 5, Eqs. 7–9 (DWLF); "
    "C2'=Ts constraint PDF p. 5 Eq. 8 (denominator simplifies to T_K); "
    "physical range C1'=3.5..43 from PDF pp. 6–15 Table 1."
)

LAW_CONSTANTS = {}   # DWLF has no frozen universal numeric constants
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "C1_prime":      {"init": None},
    "log10_eta_ref": {"init": None},
}


def _log10_eta_dwlf(
    T_K: np.ndarray,
    Ts_K: np.ndarray,
    C1_prime: float,
    log10_eta_ref: float,
) -> np.ndarray:
    """log10(eta/poise) via DWLF with C2' = Ts_K (denominator = T_K).

    T_K          : temperature, Kelvin  (n,)
    Ts_K         : reference temperature = Tg, Kelvin  (n,) or scalar
    C1_prime     : DWLF coefficient (dimensionless, > 0)
    log10_eta_ref: per-cluster intercept
    """
    # Denominator under DWLF C2'=Ts: C2' + (T - Ts) = T (Eq. 8 of paper).
    # Guard: T_K is always > 0 in physical conditions.
    safe_T = np.where(np.abs(T_K) > 1e-3, T_K, 1e-3)
    shift = -C1_prime * (T_K - Ts_K) / safe_T
    return log10_eta_ref + shift


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Fit C1_prime and log10_eta_ref on per-cluster data.

    X_fit : (n, 2)  columns [Temperature_K, Tg_K].
    y_fit : (n,)    log10(eta_poise) observations.

    Data-derived starting points:
      log10_eta_ref0 = median(y)          (value at T≈Ts where shift≈0)
      C1_prime0      = 10.0               (mid-range of 3.5–43 from paper)
    """
    T_K  = np.asarray(X_fit[:, 0], dtype=float)
    Ts_K = np.asarray(X_fit[:, 1], dtype=float)
    y    = np.asarray(y_fit, dtype=float)

    ref0 = float(np.median(y))
    if not np.isfinite(ref0):
        ref0 = 5.0
    C1_0 = 10.0

    def residuals(p):
        return _log10_eta_dwlf(T_K, Ts_K, p[0], p[1]) - y

    lo = [0.01, -20.0]
    hi = [60.0,  30.0]

    try:
        sol = least_squares(residuals, [C1_0, ref0], bounds=(lo, hi),
                            method="trf", max_nfev=4000)
        C1p, refv = sol.x
        if not np.all(np.isfinite([C1p, refv])):
            raise RuntimeError("non-finite")
        return {"C1_prime": float(C1p), "log10_eta_ref": float(refv)}
    except Exception:  # noqa: BLE001
        return {"C1_prime": float(C1_0), "log10_eta_ref": float(ref0)}


def predict(X: np.ndarray, C1_prime: float, log10_eta_ref: float) -> np.ndarray:
    """DWLF prediction of log10(eta_poise).

    The harness calls predict(X, **fitted_local) per cluster (LAW={}), i.e.
    with the per-cluster fitted C1_prime and log10_eta_ref scalars.

    X             : (n, 2)  columns [Temperature_K, Tg_K], ordered per USED_INPUTS.
    C1_prime      : per-cluster fitted DWLF coefficient.
    log10_eta_ref : per-cluster fitted intercept.

    Returns log10(eta/poise) as (n,) array.
    """
    T_K  = np.asarray(X[:, 0], dtype=float)
    Ts_K = np.asarray(X[:, 1], dtype=float)
    return _log10_eta_dwlf(T_K, Ts_K, C1_prime, log10_eta_ref)
