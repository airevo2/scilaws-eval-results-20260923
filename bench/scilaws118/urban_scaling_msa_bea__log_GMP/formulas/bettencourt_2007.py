"""Urban power-law scaling law — log_GMP ~ beta * log_pop + log_Y0.

Bettencourt, L. M. A., Lobo, J., Helbing, D., Kühnert, C., and West, G. B.
(2007). *Growth, innovation, scaling and the pace of life in cities.*
Proceedings of the National Academy of Sciences, 104(17):7301–7306.
DOI:10.1073/pnas.0610172104.

Equation (1) of the paper (PDF p. 9) states the urban power-law scaling law:

    Y(t) = Y_0 * N(t)^beta

Taking logarithms (the regression form, Materials and Methods, PDF p. 17):

    ln Y_i = ln Y_0 + beta * ln N_i + epsilon_i

where:
- Y_i  is the urban indicator for MSA i (benchmark target: GMP in USD)
- N_i  is the metropolitan population of MSA i
- Y_0  is a per-indicator normalization constant (absorbed into log_Y0 below)
- beta  is the scaling exponent (dimensionless; >1 for socioeconomic quantities)
- epsilon_i is the residual (SAMI — Scale-Adjusted Metropolitan Indicator)

The benchmark target is log_GMP = ln(GMP_i). The input is log_pop = ln(N_i).
Fit is by OLS on the log-linear form; the predicted target is log_GMP_hat.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The paper reports indicative ranges for beta (e.g., GDP/wages exponents
1.07–1.34 across country-years, Table 1, PDF p. 26; total wages USA 2002 gives
beta = 1.12 [1.09, 1.13]), but no universal frozen beta for BEA MSA GMP is
stated; the paper fits beta per indicator per country-year. No universal Y_0 is
provided either; it is always fit to data.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The log-linear regression form ln Y = ln Y_0 + beta * ln N is
algebraically identical to the power-law form; no additional structural
factors appear.

LOCAL_FITTABLE — per-cluster, fit by OLS
-----------------------------------------
- log_Y0 : log of the normalization constant (= OLS intercept); units of
           log(USD) for the GMP target.  init = None (OLS; closed form).
- beta    : scaling exponent (dimensionless; expected >1 for GMP/wages).
           init = None (OLS; closed form).

Fit procedure: standard OLS on the (log_pop, log_GMP) scatter, equivalent to
the Materials and Methods regression described on PDF p. 17 of Bettencourt 2007.
No iterative solver is required; numpy.linalg.lstsq gives the unique solution.
"""

import numpy as np

USED_INPUTS = ["log_pop"]
PAPER_REF = "summary_formula_bettencourt_2007.md"
EQUATION_LOC = (
    "Bettencourt et al. 2007, Eq. (1), PDF p. 9 — Y = Y_0 * N^beta; "
    "log-linear OLS form: ln Y = ln Y_0 + beta * ln N + epsilon "
    "(Materials and Methods, PDF p. 17)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "log_Y0": {"init": None},
    "beta":   {"init": None},
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """OLS fit of the log-linear urban scaling law.

    X_fit: (n, 1) — column [log_pop], i.e. ln(population).
    y_fit: (n,)   — log_GMP = ln(GMP_i) for each MSA.

    Returns: {"log_Y0": float, "beta": float}.
    The OLS solution is exact (no iteration), mirroring the regression
    procedure in Bettencourt et al. 2007 Materials and Methods (PDF p. 17).
    """
    log_pop = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # Design matrix for OLS: [intercept, log_pop]
    A = np.column_stack([np.ones_like(log_pop), log_pop])
    # Least-squares solution: [log_Y0, beta]
    result, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    log_Y0, beta = float(result[0]), float(result[1])
    return {"log_Y0": log_Y0, "beta": beta}


def predict(X: np.ndarray, log_Y0: float, beta: float) -> np.ndarray:
    """Predicted log_GMP from urban power-law scaling.

    X: (n, 1) — column [log_pop], i.e. ln(metropolitan population).

    Returns: (n,) array of predicted log_GMP = log_Y0 + beta * log_pop.
    """
    log_pop = np.asarray(X[:, 0], dtype=float)
    return log_Y0 + beta * log_pop
