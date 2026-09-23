"""Classical Bouguer-Lambert-Beer law (alpha = beta = 1 special case).

Beer A (1852). "Bestimmung der Absorption des rothen Lichts in farbigen
Fluessigkeiten."  Annalen der Physik, 162(5):78-88.
Lambert JH (1760). Photometria.  Augsburg: Eberhard Klett.
Bouguer P (1729). Essai d'optique sur la gradation de la lumière.

As stated in Yeh et al. 2023, Eq. 1 / Eq. 2 (PDF pp. 5, 13):
    A_λ = ε_λ · c · l

The classical law sets alpha = beta = 1 in the generalized form.  For
potassium dichromate, the classical law deviates due to chemical
equilibria (dimerization of HCrO4⁻ to Cr2O7²⁻; see Yeh 2023 §3.1,
PDF pp. 17-19), particularly in the UV region where the two species
have very different molar absorption coefficients.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
alpha = 1.0  — exponent of concentration, fixed at 1 in classical BLB
               (Yeh 2023 Eq. 1/2, PDF pp. 5, 13; classical limit stated
               explicitly: "If α and β are equal to 1, the proposed
               equation is identical to the original BLB law.")
beta  = 1.0  — exponent of path length, fixed at 1 in classical BLB
               (same citation)

OTHER_CONSTANTS — universal physics / unit factors
--------------------------------------------------
(empty: the formula is dimensionless.)

LOCAL_FITTABLE — per-cluster (per-wavelength)
---------------------------------------------
- epsilon : apparent specific absorbance at wavelength λ [L g⁻¹ cm⁻¹].
            Fitted per-λ by linear regression in log or direct OLS in
            the linear domain (since alpha=beta=1, A = epsilon * c * l
            is linear in epsilon given c and l).
            init=None (closed-form OLS).

Type: TYPE II — per-cluster (per-wavelength) fit required.
Even with alpha=beta=1 frozen, epsilon varies strongly with wavelength
(it IS the absorption spectrum).  A global epsilon pooled across all 501
wavelengths would produce RMSE that explodes because ε changes by two
orders of magnitude.

Column mapping (paper → CSV):
    c → concentration   [g/L]
    l → path_length     [cm]
    A → absorbance      (dimensionless; target, column 0)

Caveats:
    With alpha=1, beta=1, the model is linear in epsilon: epsilon = A/(c·l).
    fit() uses closed-form OLS (minimize sum((A - epsilon*c*l)^2 over rows),
    solution: epsilon = sum(A*c*l) / sum((c*l)^2)).
"""

import numpy as np

USED_INPUTS = ["concentration", "path_length"]
PAPER_REF   = "summary_formula_dataset_yeh_2023.md"
EQUATION_LOC = "Yeh 2023 review, Eq. 2 PDF p. 13 (BLB law per wavelength); Eq. 1 PDF p. 5 (general form, classical limit alpha=beta=1). The historical Beer 1852 law per se is not equation-numbered in modern style."

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "alpha": 1.0,   # concentration exponent — classical BLB; Yeh 2023 Eq. 1 PDF p. 5
    "beta":  1.0,   # path-length exponent  — classical BLB; Yeh 2023 Eq. 1 PDF p. 5
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}

LOCAL_FITTABLE = {
    "epsilon": {"init": None},   # closed-form OLS; apparent specific absorbance
}


def predict(X: np.ndarray, alpha: float, beta: float, epsilon: float) -> np.ndarray:
    """Classical Beer-Lambert-Bouguer law A = epsilon * c^alpha * l^beta.

    With LAW_CONSTANTS alpha=beta=1 this reduces to A = epsilon * c * l.

    X[:, 0] = concentration c [g/L]
    X[:, 1] = path_length l [cm]
    """
    c = np.asarray(X[:, 0], dtype=float)
    l = np.asarray(X[:, 1], dtype=float)
    c_safe = np.where(c > 0, c, 1e-12)
    l_safe = np.where(l > 0, l, 1e-12)
    return epsilon * np.power(c_safe, alpha) * np.power(l_safe, beta)


def fit(X_fit: np.ndarray, y_fit: np.ndarray, alpha: float, beta: float) -> dict:
    """Closed-form OLS for epsilon given alpha=1, beta=1.

    Since A = epsilon * c * l, minimize sum((A - epsilon * c * l)^2):
        epsilon = sum(A * c * l) / sum((c * l)^2)

    Receives LAW_CONSTANTS alpha, beta as kwargs.
    """
    c = np.asarray(X_fit[:, 0], dtype=float)
    l = np.asarray(X_fit[:, 1], dtype=float)
    y = np.asarray(y_fit, dtype=float)
    cl = np.power(np.maximum(c, 1e-12), alpha) * np.power(np.maximum(l, 1e-12), beta)
    numerator   = float(np.dot(y, cl))
    denominator = float(np.dot(cl, cl))
    if denominator < 1e-30:
        epsilon = 1.0
    else:
        epsilon = numerator / denominator
    return {"epsilon": epsilon}
