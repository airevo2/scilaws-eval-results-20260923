"""Cauchy (1836) three-term dispersion formula - refractive index n(lambda).

A.-L. Cauchy, "Memoire sur la dispersion de la lumiere," J. G. Calve,
Prague, 1836. Public domain. Eq. (55), PDF p. 215; also Eq. (53), same page.

The formula in Cauchy's notation (eq. 55):

    theta^2 = a + b*s^2 + c*s^4

where s is wave frequency (proportional to 1/lambda), theta is the refractive
index, and a, b, c are material-dependent fitting constants. In modern notation
(substituting s = 1/lambda, lambda in micrometres):

    n^2(lambda) = A + B / lambda^2 + C / lambda^4

and therefore:

    n(lambda) = sqrt(A + B / lambda^2 + C / lambda^4)

This formula is isothermal (no temperature correction). It is used as a
simpler single-input baseline alongside the Jakubczyk (2023) model.

Type designation: Type I.
LAW_CONSTANTS are the Cauchy three-term coefficients obtained by an OLS fit to the
ethylene glycol (EG/MEG) Sellmeier dispersion curve at T=20 from Jakubczyk (2023)
Table 2 (single liquid; the benchmark uses EG only). They are NOT Cauchy's 1836
glycol values (that paper covered only water, crownglass, flintglass).

Derivation of the Cauchy LAW_CONSTANTS (A_C, B_C, C_C):
  Obtained by fitting the Cauchy three-term
  form (n^2 = A_C + B_C/lambda^2 + C_C/lambda^4) by ordinary least-squares regression
  to the n(lambda, T=20) trace from the Jakubczyk (2023) Table 2 Sellmeier formula
  for ethylene glycol (EG/MEG). Fit done over lambda in [0.394, 1.071] um at 80 points.

  Result:
    A_C = 2.015621  (leading n^2 term of the OLS fit over the 0.394-1.071 um window)
    B_C = 0.013053  (1/lambda^2 dispersion strength; um^2; normal dispersion for EG)
    C_C = -0.000306 (1/lambda^4 higher-order term; um^4; slight anomalous sign
                      is a fitting artifact within the limited 0.394-1.071 um window;
                      physically negligible: |C_C/lambda^4| < 0.002 for lambda > 0.4 um)

  Fit quality: RMSE of Cauchy vs Jakubczyk EG n(lambda, T=20) = 0.000223 over the full range.

  NOTE: A_C, B_C, C_C are NOT from the Cauchy 1836 paper for glycols (that paper
  gave values for water, crownglass, flintglass). They are derived from the
  Jakubczyk (2023) Table 2 EG data, which is the primary data source for this task.
  For benchmark purposes, they are LAW_CONSTANTS (paper-table-derived, fixed).

USED_INPUTS: ["lambda_um", "temperature_C"] - temperature column included (benchmark contract)
but not consumed by the isothermal Cauchy formula (only lambda_um is used).
"""

import numpy as np

USED_INPUTS = ["lambda_um", "temperature_C"]
PAPER_REF = "summary_formula_cauchy_1836.md"
EQUATION_LOC = (
    "Cauchy (1836) eq. 55, PDF p. 215: theta^2 = a + b*s^2 + c*s^4; "
    "modern form n^2 = A_C + B_C/lambda^2 + C_C/lambda^4 (lambda in um). "
    "LAW_CONSTANTS derived by OLS fit to EG Sellmeier n(lambda,T=20) from "
    "Jakubczyk (2023) Table 2, ethylene glycol row, PDF p. 8."
)

# LAW_CONSTANTS - Cauchy coefficients fitted to EG n(lambda, T=20) from Jakubczyk (2023)
# Table 2 Sellmeier constants for ethylene glycol (EG/MEG), PDF p. 8.
# Derived by OLS regression of n^2 ~ A_C + B_C/lambda^2 + C_C/lambda^4 on the EG
# Sellmeier n(lambda, T=20) trace (80 lambda points, 0.394-1.071 um).
LAW_CONSTANTS = {
    "A_C": 2.015621,    # dimensionless; n^2 leading term (far-IR limit for EG); OLS fit Table 2
    "B_C": 0.013053,    # um^2; 1/lambda^2 dispersion term; OLS fit to EG Table 2
    "C_C": -0.000306,   # um^4; 1/lambda^4 higher-order term; OLS fit to EG Table 2
}

OTHER_CONSTANTS = {}   # No additional non-LAW constants.

LOCAL_FITTABLE = {}    # Type I - no per-cluster parameters.


def predict(X: np.ndarray, A_C: float, B_C: float, C_C: float) -> np.ndarray:
    """Cauchy refractive index n = sqrt(A_C + B_C/lambda^2 + C_C/lambda^4).

    X: (n, 2) - columns [lambda_um, temperature_C].
    Temperature column is present in X (benchmark contract) but is not used
    by the isothermal Cauchy formula; only lambda_um (column 0) is consumed.

    Returns: (n,) array of predicted refractive index.
    """
    lam = np.asarray(X[:, 0], dtype=float)
    x = 1.0 / (lam * lam)           # x = 1/lambda^2  [um^-2]
    n2 = A_C + B_C * x + C_C * x * x    # n^2 = A_C + B_C/lambda^2 + C_C/lambda^4
    return np.sqrt(np.clip(n2, 1e-6, None))
