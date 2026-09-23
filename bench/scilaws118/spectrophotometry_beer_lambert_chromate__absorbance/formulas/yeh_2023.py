"""Yeh et al. (2023) Generalized Beer-Lambert-Bouguer model (Eq. 3).

Yeh Y-C, Haasdonk B, Schmid-Staiger U, Stier M, Tovar GEM (2023).
"A novel model extended from the Bouguer-Lambert-Beer law can describe the
non-linear absorbance of potassium dichromate solutions and microalgae
suspensions."  Frontiers in Bioengineering and Biotechnology, 11:1116735.
DOI: 10.3389/fbioe.2023.1116735.  CC BY 4.0.

Formula (Eq. 3, PDF p. 13):
    A_λ = ε'_λ · c^α_λ · l^β_λ

where the subscript λ indicates that all three parameters are specific to
each measurement wavelength.  For potassium dichromate, β_λ ≈ 1 across the
full spectrum (the BLB deviation is chemical, not optical scattering — see
§3.1 of the paper, PDF pp. 17-19), while α_λ varies substantially and ε'_λ
varies by orders of magnitude across 200-450 nm.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None.  The functional *form* (power-law extension of BLB) is Yeh 2023's
scientific claim; no globally fixed numeric values are published for all
wavelengths.  The per-wavelength values are LOCAL_FITTABLE.

OTHER_CONSTANTS — universal physics / unit factors
--------------------------------------------------
(empty: the formula is dimensionless; c and l carry explicit SI-domain units
that cancel with ε'.)

LOCAL_FITTABLE — per-cluster (per-wavelength), fit by nonlinear least squares
------------------------------------------------------------------------------
- eps_prime : effective specific absorbance at wavelength λ [units depend on
              α, β; A is dimensionless so [ε'] = L^α g^{-α} cm^{-β}].
              Paper: Figure 2A shows ε' varying across wavelengths; fitted
              per-λ by NonlinearLeastSquares (Matlab R2022a, PDF p. 14).
              Positive by definition. init=0.1 (small positive start).
- alpha     : correction coefficient of concentration (real positive).
              Paper: Figure 2D shows α close to ~0.9-1.1 for K2Cr2O7.
              init=1.0 (BLB starting point).
- beta      : correction coefficient of path length (real positive).
              Paper: Figure 2D shows β ≈ 1.0 for K2Cr2O7 (chemical deviation
              is concentration-driven, not path-length-driven, §3.1 p. 17-18).
              init=1.0 (BLB starting point).

Type: TYPE II — per-cluster (per-wavelength) fit required.
The task is a grid of 501 wavelengths × 11 (c, l) subsamples; wavelength is
the cluster identifier.  All three formula parameters are wavelength-specific
and have no global paper-published values.

Column mapping (paper → CSV):
    c → concentration   [g/L]
    l → path_length     [cm]
    A → absorbance      (dimensionless; target, column 0)
    λ → wavelength      [nm] (cluster id, not a formula input to predict())

Caveats:
    Scipy Nelder-Mead is used for nonlinear fitting (3 params; gradient-free).
    Multi-start would be more robust but 3×multi-start per 501 wavelengths is
    slow; a single start at (1.0, 1.0, 0.1) is near the BLB regime.
    If fit() encounters negative predictions during optimization, clip to 1e-12.
"""

import numpy as np
from scipy.optimize import minimize

USED_INPUTS = ["concentration", "path_length"]
PAPER_REF   = "summary_formula_dataset_yeh_2023.md"
EQUATION_LOC = "Eq. 3, PDF p. 13 (wavelength-specific form); Eq. 1, PDF p. 5 (general form)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS   = {}   # no globally fixed numeric values; form IS the claim
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}   # dimensionless after ε' unit adjustment

LOCAL_FITTABLE = {
    "eps_prime": {"init": [0.01, 0.1, 1.0, 10.0]},  # multi-start: absorptivity varies orders of magnitude
    "alpha":     {"init": [0.5, 1.0, 1.5]},
    "beta":      {"init": [0.8, 1.0, 1.2]},
}


def predict(X: np.ndarray, eps_prime: float, alpha: float, beta: float) -> np.ndarray:
    """Generalized Beer-Lambert-Bouguer law at a fixed wavelength.

    A = eps_prime * c^alpha * l^beta

    X[:, 0] = concentration c [g/L]
    X[:, 1] = path_length l [cm]
    """
    c = np.asarray(X[:, 0], dtype=float)
    l = np.asarray(X[:, 1], dtype=float)
    # Protect against negative base (should not occur; c, l are positive)
    c_safe = np.where(c > 0, c, 1e-12)
    l_safe = np.where(l > 0, l, 1e-12)
    return eps_prime * np.power(c_safe, alpha) * np.power(l_safe, beta)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear least-squares fit of (eps_prime, alpha, beta) per wavelength.

    Mimics the Matlab R2022a NonlinearLeastSquares procedure (PDF p. 14, Eq. 3).
    Uses Nelder-Mead (gradient-free) with multi-start to handle the non-convex
    eps_prime × alpha × beta landscape.

    Returns dict with keys exactly matching LOCAL_FITTABLE.
    """
    y = np.asarray(y_fit, dtype=float)

    def residuals_sq(params):
        ep, al, be = params
        if ep <= 0:
            return 1e12
        y_pred = predict(X_fit, eps_prime=ep, alpha=al, beta=be)
        return float(np.sum((y - y_pred) ** 2))

    # Multi-start grid over eps_prime × alpha × beta
    best_result = None
    best_val = np.inf
    for ep0 in LOCAL_FITTABLE["eps_prime"]["init"]:
        for al0 in LOCAL_FITTABLE["alpha"]["init"]:
            for be0 in LOCAL_FITTABLE["beta"]["init"]:
                res = minimize(
                    residuals_sq,
                    x0=[ep0, al0, be0],
                    method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000},
                )
                if res.fun < best_val and res.x[0] > 0:
                    best_val = res.fun
                    best_result = res

    if best_result is None or best_result.x[0] <= 0:
        # Fallback: classical BLB fit (linear in log-log space)
        log_c = np.log(np.maximum(X_fit[:, 0], 1e-12))
        log_l = np.log(np.maximum(X_fit[:, 1], 1e-12))
        log_y = np.log(np.maximum(y, 1e-12))
        A = np.column_stack([np.ones_like(log_c), log_c, log_l])
        coef, *_ = np.linalg.lstsq(A, log_y, rcond=None)
        return {
            "eps_prime": float(np.exp(coef[0])),
            "alpha": float(coef[1]),
            "beta": float(coef[2]),
        }

    ep, al, be = best_result.x
    return {"eps_prime": float(ep), "alpha": float(al), "beta": float(be)}
