"""Williams-Landel-Ferry (WLF) equation with universal constants — log10(eta/poise).

Shangguan, Y. et al. (2017). New Insight into Time-Temperature Correlation
for Polymer Relaxations Ranging from Secondary Relaxation to Terminal Flow:
Application of a Universal and Developed WLF Equation. Polymers 9(11), 567.
DOI:10.3390/polym9110567. Open-access (MDPI, CC-BY-4.0).

NOTE on author / bib key: The bib entry is keyed ``yin2017`` and the PDF
filename is ``yin_2017_polymers_wlf.pdf``, but the first author listed on the
PDF title page is **Yonggang Shangguan**. This file follows the spec rule
(first author on the PDF title page) and is named ``shangguan_2017_wlf.py``.

WLF equation (PDF p. 4, Eq. 1):

    lg alpha_T = -C1 * (T - Ts) / (C2 + (T - Ts))

Universal constants when Ts = Tg (glass-transition temperature):
    C1 = 17.44  (dimensionless)
    C2 = 51.6 K

Since the WLF shift factor alpha_T = eta(T) / eta(Ts):

    log10_eta(T) = log10_eta_ref - C1 * (T_K - Tg_K) / (C2 + (T_K - Tg_K))

where:
    T_K          = temperature in K (input column Temperature_K)
    Tg_K         = glass-transition temperature in K — per-cluster observed
                   covariate (input column Tg_K in the data files)
    log10_eta_ref = log10(eta(Tg)) — per-cluster LOCAL_FITTABLE parameter

Note on Mw and Shear_Rate columns: this WLF variant treats the T-dependence
only. Mw and Shear_Rate are present in the dataset but are NOT used by this
formula — they reside in the dataset for SR methods that may wish to
incorporate them (e.g. the Berry-Fox Mw^3.4 dependence). ``USED_INPUTS``
below accurately reflects what this formula consumes.

LAW_CONSTANTS — paper-published, frozen, invariant across all polymers
---------------------------------------------------------------------
  C1 = 17.44 (dimensionless), C2 = 51.6 K. These are the Williams-Landel-Ferry
  (1955) EMPIRICALLY FITTED "universal" coefficients of the named WLF equation,
  reported as applicable to most amorphous polymers when Ts = Tg. They are the
  *defining coefficients of the law*: paper-fitted (WLF 1955), frozen, and held
  invariant across every polymer cluster here (the harness passes them to fit()
  and predict() via **LAW). This is the textbook LAW case — distinct from an
  observed/assumed given (which would be OTHER): C1/C2 were fitted by the source
  paper, not merely consumed as a standard reference value.
  Source: Shangguan et al. (2017) PDF p. 4, Eq. 1 / L192-193:
  "C1, C2 are empirical constants. C1 = 17.44 and C2 = 51.6 K are applicable to
  most amorphous polymers provided Tg is chosen as Ts"; original Williams,
  Landel, Ferry (1955) JACS 77:3701, Eq. 1.

OTHER_CONSTANTS — universal / structural givens
-----------------------------------------------
  (empty.) The 273.15 K Celsius->Kelvin offset is applied in prep_data.py
  before Temperature_K is stored; it is not consumed here.

LOCAL_FITTABLE — per-cluster (per-polymer), fitted by fit()
-----------------------------------------------------------
  log10_eta_ref : log10(eta_poise) at T = Tg_K; the absolute viscosity scale
                  for this polymer cluster.
                  init = None  (data-derived; see fit() below)
"""

import numpy as np
from scipy.optimize import least_squares

# NOTE: scipy.optimize is imported for the LOCAL_FITTABLE fit() only.
# predict() uses only numpy.

USED_INPUTS = ["Temperature_K", "Tg_K"]

PAPER_REF = "summary_formula_shangguan_2017.md"
EQUATION_LOC = (
    "Shangguan et al. (2017) Polymers 9(11):567, PDF p. 4, Eq. 1; "
    "universal constants C1=17.44 and C2=51.6 K stated PDF p. 4 text (L192-193)."
)

# C1, C2 are the WLF (1955) paper-fitted universal defining coefficients of the
# WLF equation — frozen and invariant across all polymers -> LAW_CONSTANTS.
LAW_CONSTANTS = {
    "C1": 17.44,  # dimensionless; WLF universal coefficient at Ts=Tg; PDF p. 4
    "C2": 51.6,   # K; WLF universal coefficient at Ts=Tg; PDF p. 4
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "log10_eta_ref": {"init": None},
}


def _log10_eta_wlf(T_K: np.ndarray, Tg_K: np.ndarray,
                   C1: float, C2: float, log10_eta_ref: float) -> np.ndarray:
    """Compute log10(eta/poise) via WLF with universal constants.

    T_K           : temperature, Kelvin  (n,)
    Tg_K          : glass-transition temperature, Kelvin  (n,) or scalar
    C1, C2        : WLF universal constants (LAW, passed by the harness)
    log10_eta_ref : per-cluster intercept
    """
    dT    = T_K - Tg_K                   # T - Ts
    denom = C2 + dT                      # C2 + (T - Tg); > 0 for T > Tg - C2
    # Guard: avoid division by zero (should not occur after FM-J4 pre-flight,
    # but defensive for out-of-range inputs from SR methods)
    safe_denom = np.where(np.abs(denom) > 1e-6, denom, np.sign(denom + 1e-30) * 1e-6)
    shift = -C1 * dT / safe_denom
    return log10_eta_ref + shift


def fit(X_fit: np.ndarray, y_fit: np.ndarray, C1: float, C2: float) -> dict:
    """Fit log10_eta_ref (sole LOCAL_FITTABLE) on per-cluster data.

    The harness passes the LAW constants C1, C2 as keyword arguments
    (fit(X_fit, y_fit, **LAW_CONSTANTS)); only the per-cluster intercept
    log10_eta_ref is fitted.

    X_fit : (n, 2)  columns [Temperature_K, Tg_K].
    y_fit : (n,)    log10(eta_poise) observations.

    Analytic optimum: log10_eta_ref = mean(y - shift).
    least_squares is used for consistency with the v2 fit contract.
    """
    T_K  = np.asarray(X_fit[:, 0], dtype=float)
    Tg_K = np.asarray(X_fit[:, 1], dtype=float)
    y    = np.asarray(y_fit, dtype=float)

    dT    = T_K - Tg_K
    denom = C2 + dT
    safe_denom = np.where(np.abs(denom) > 1e-6, denom, np.sign(denom + 1e-30) * 1e-6)
    shift = -C1 * dT / safe_denom

    # Analytic starting point
    ref0 = float(np.median(y - shift))
    if not np.isfinite(ref0):
        ref0 = 5.0  # fallback: typical log10(eta/poise) mid-range

    def residuals(p):
        return _log10_eta_wlf(T_K, Tg_K, C1, C2, p[0]) - y

    try:
        sol = least_squares(residuals, [ref0], bounds=([-20.0], [25.0]),
                            method="trf", max_nfev=2000)
        val = float(sol.x[0])
        if not np.isfinite(val):
            raise RuntimeError("non-finite")
        return {"log10_eta_ref": val}
    except Exception:  # noqa: BLE001
        return {"log10_eta_ref": ref0}


def predict(X: np.ndarray, C1: float, C2: float, log10_eta_ref: float) -> np.ndarray:
    """WLF prediction of log10(eta_poise).

    The harness calls predict(X, **LAW_CONSTANTS, **fitted_local), i.e. with
    C1, C2 (LAW) and the per-cluster fitted log10_eta_ref.

    X             : (n, 2)  columns [Temperature_K, Tg_K], ordered per USED_INPUTS.
    C1, C2        : WLF universal constants (LAW).
    log10_eta_ref : per-cluster fitted intercept.

    Returns log10(eta/poise) as (n,) array.
    """
    T_K  = np.asarray(X[:, 0], dtype=float)
    Tg_K = np.asarray(X[:, 1], dtype=float)
    return _log10_eta_wlf(T_K, Tg_K, C1, C2, log10_eta_ref)
