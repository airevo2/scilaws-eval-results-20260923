"""Chave et al. (2014) pantropical H:D allometric equation — height_m.

Chave, J., Réjou-Méchain, M., Búrquez, A., et al. (2014). Improved allometric
models to estimate the aboveground biomass of tropical trees. *Global Change
Biology* 20, 3177–3190. DOI:10.1111/gcb.12629.

Eqn 6a (PDF pp. 6–7) — pantropical log-quadratic H:D sub-model:

    ln(H) = 0.893 − E + 0.760·ln(D) − 0.0340·[ln(D)]²

Solving for H:

    H = exp(0.893 − E + 0.760·ln(D) − 0.0340·[ln(D)]²)

where D is trunk diameter at breast height (cm), H is total tree height (m),
and E is the bioclimatic stress index (Eqn 6b; dimensionless):

    E = 1e-3 × (0.178·TS − 0.938·CWD − 6.61·PS)

The four numeric coefficients in Eqn 6a (0.893, −1, 0.760, −0.0340) are
paper-published fixed values (LAW_CONSTANTS). The three Eqn 6b coefficients
(0.178, −0.938, −6.61, scale 1e-3) are also paper-published but are NOT used
in predict() because E cannot be computed from the raw CSV (TS, CWD, PS absent).
They are therefore stored in OTHER_CONSTANTS (S2-P12 C10a sweep, 2026-05-28).

Dataset note (PROVENANCE.md): The raw CSV contains only D (= DBH, cm) and H.
The bioclimatic raster values (TS, CWD, PS) are not present in the CSV and
would require a separate GIS join. Therefore E is not computable from the
available columns. Per PROVENANCE.md, E = 0 is used as a site-level
approximation for sites near the global median stress. fit() does NOT re-fit
the four Eqn-6a law constants; it fits a single per-cluster intercept
correction δ (additive in ln-space) to absorb any site-mean offset, treating
all four Eqn-6a coefficients as frozen. If site-level E data become available,
they should be passed as a per-cluster covariate rather than absorbed into δ.

LAW_CONSTANTS — paper-published, frozen (Chave 2014 Eqn 6a only)
-----------------------------------------------------------------
These four constants are directly referenced in _ln_H() / predict():
- intercept   : 0.893   — Eqn 6a, PDF p. 7; ln(H) baseline at D=1 cm, E=0
- coeff_E     : -1.0    — Eqn 6a, PDF p. 7; sensitivity of ln(H) to stress E
                          (C10c borderline: E=0.0 always in this implementation,
                          but coeff_E is structurally in Eqn 6a)
- coeff_lnD   : 0.760   — Eqn 6a, PDF p. 7; log-log scaling exponent of D
- coeff_lnD2  : -0.0340 — Eqn 6a, PDF p. 7; quadratic ln(D) curvature term

OTHER_CONSTANTS — documentation companions (S2-P12 C10a)
---------------------------------------------------------
Eqn 6b coefficients: paper-published but NOT referenced in predict() because
TS, CWD, PS are absent from the released CSV. Stored here for completeness:
- e6b_TS    : 0.178   — Eqn 6b, PDF p. 7; temperature seasonality weight
- e6b_CWD   : -0.938  — Eqn 6b, PDF p. 7; climatic water deficit weight
- e6b_PS    : -6.61   — Eqn 6b, PDF p. 7; precipitation seasonality weight
- e6b_scale : 1e-3    — Eqn 6b, PDF p. 7; overall scaling factor (literal 10^-3)

LOCAL_FITTABLE — per-cluster, fitted by fit() via ordinary least squares
------------------------------------------------------------------------
- delta : additive intercept correction in ln-space (dimensionless).
          Absorbs per-site mean H:D offset not explained by Eqn 6a when E is
          unavailable. At E = 0 and with the true site-E incorporated, delta
          should converge to 0; in practice it captures site-level deviations.
          init = 0.0.
"""

import numpy as np

USED_INPUTS = ["DBH_cm"]
PAPER_REF = "summary_formula_chave_2014.md"
EQUATION_LOC = (
    "Chave et al. (2014) Eqn 6a, PDF pp. 6–7: "
    "ln(H) = 0.893 - E + 0.760*ln(D) - 0.0340*[ln(D)]^2; "
    "Eqn 6b: E = 1e-3*(0.178*TS - 0.938*CWD - 6.61*PS)."
)

LAW_CONSTANTS = {
    "intercept":  0.893,    # Chave 2014 Eqn 6a, PDF p. 7
    "coeff_E":   -1.0,      # Chave 2014 Eqn 6a, PDF p. 7
    "coeff_lnD":  0.760,    # Chave 2014 Eqn 6a, PDF p. 7
    "coeff_lnD2": -0.0340,  # Chave 2014 Eqn 6a, PDF p. 7
}
OTHER_CONSTANTS = {
    # Eqn 6b coefficients (NOT used in predict(); TS/CWD/PS absent from CSV).
    # S2-P12 C10a: moved from LAW_CONSTANTS because none of these keys are
    # referenced inside predict() — they compute E, which is hardcoded to 0.
    "e6b_TS":    0.178,    # Chave 2014 Eqn 6b, PDF p. 7; temperature seasonality
    "e6b_CWD":  -0.938,   # Chave 2014 Eqn 6b, PDF p. 7; climatic water deficit
    "e6b_PS":   -6.61,    # Chave 2014 Eqn 6b, PDF p. 7; precip. seasonality
    "e6b_scale": 1e-3,    # Chave 2014 Eqn 6b, PDF p. 7; scale factor (10^-3)
}
LOCAL_FITTABLE = {
    "delta": {"init": 0.0},  # per-cluster ln-space intercept offset
}


def _ln_H(D_cm, delta, E=0.0):
    """ln(H) from Chave 2014 Eqn 6a plus per-cluster offset delta."""
    lnD = np.log(D_cm)
    return (
        LAW_CONSTANTS["intercept"]
        + LAW_CONSTANTS["coeff_E"] * E
        + LAW_CONSTANTS["coeff_lnD"] * lnD
        + LAW_CONSTANTS["coeff_lnD2"] * lnD ** 2
        + delta
    )


def fit(X_fit: np.ndarray, y_fit: np.ndarray, **law_constants) -> dict:
    """OLS fit of the per-cluster ln-space offset delta.

    The four Eqn-6a law constants are held frozen (received via **law_constants
    per the v2 harness contract; they are not re-fit). Only delta is optimised.
    D_cm = X_fit[:, 0]; E is not in the CSV so E=0 is used throughout.

    Start: delta0 = 0.0 (Eqn 6a as-published).
    """
    D_cm = np.asarray(X_fit[:, 0], dtype=float)
    H_obs = np.asarray(y_fit, dtype=float)

    # Work in ln-space for a convex OLS problem.
    lnH_obs = np.log(np.clip(H_obs, 1e-6, None))
    lnH_pred0 = _ln_H(D_cm, delta=0.0, E=0.0)
    delta_hat = float(np.mean(lnH_obs - lnH_pred0))

    return {"delta": delta_hat}


def predict(X: np.ndarray, **params) -> np.ndarray:
    """Total tree height H (m) from trunk diameter D (cm).

    X: (n, 1) — column [DBH_cm].
    params contains LAW_CONSTANTS (intercept, coeff_E, coeff_lnD, coeff_lnD2)
    plus LOCAL (delta). E is fixed to 0 (bioclimatic raster not in CSV; see
    module docstring). Uses module-level LAW_CONSTANTS for structural
    coefficients; delta is the per-cluster fitted offset.
    """
    D_cm = np.asarray(X[:, 0], dtype=float)
    delta = float(params.get("delta", 0.0))
    return np.exp(_ln_H(D_cm, delta=delta, E=0.0))
