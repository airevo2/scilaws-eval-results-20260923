"""Holling (1959) disc equation — Holling Type II functional response.

Citation
--------
Bosomtwe, A., Opit, G., Giles, K., Kard, B., & Goad, C. (2025).
Functional Responses of the Warehouse Pirate Bug *Xylocoris flavipes*
(Hemiptera: Anthocoridae) on a Diet of *Liposcelis decolor*
(Psocodea: Liposcelididae). *Insects*, 16(1), 101.
DOI: 10.3390/insects16010101

Equation
--------
The Holling disc equation (originally Holling 1959, Canadian Entomologist
91:385-398) as stated in Bosomtwe et al. 2025, Section 2.5, PDF page 5:

    Na = a * T * N / (1 + a * T_h * N)

With trial duration T = 1 day absorbed into the per-cluster fitted
parameters (standard practice when fitting the disc equation to fixed-
duration trials), the working form is:

    prey_eaten = a * prey_density / (1 + a * h * prey_density)

LAW_CONSTANTS
-------------
None. The disc equation's "claim" is its functional form (Michaelis-Menten
type saturation), not a specific numeric constant. The form itself is the
invariant law across all predator-prey systems in this benchmark.

Note: the structural constants of the form — the literal 1 in the
denominator and the linear exponent on prey_density in both numerator and
denominator — are inline numerals (not declared as LAW_CONSTANTS per
data_spec §5.2). They are what distinguishes Type II from Type III (which
has N^2).

OTHER_CONSTANTS
---------------
None. The formula is dimensionally clean (both prey_eaten and prey_density
are dimensionless counts, T=1 is absorbed).

LOCAL_FITTABLE
--------------
Per-cluster (per-experiment) parameters fitted by Nelder-Mead minimisation
of SSE, with positive-constraint transformations (log-space optimisation):

  a  : attack rate — rate at which the predator searches for and attacks
       prey per unit time (per-experiment specific).
       init = 0.5 (a common starting point across most functional-response
       datasets; broad enough for diverse predator-prey pairs).
  h  : handling time — time required to pursue, subdue, and consume one
       prey item, effectively limiting the maximum consumption rate.
       init = 0.1 (order-of-magnitude start; covers both very fast and
       slow handlers at the range of N in the FoRAGE dataset).

Type designation
----------------
TYPE II. Per-experiment (group_id) parameter recovery is the target.
Each cluster in the FoRAGE database is a distinct functional-response
experiment (unique predator-prey species pair, arena conditions,
temperature), making rows across different clusters non-interchangeable.
Both a and h must be refit for every cluster.

Column mapping
--------------
Paper notation → released CSV columns:
  Na               → prey_eaten   (column 0, SR target)
  N (original x)   → prey_density (column 1, input)
  a, T_h (= h)     → LOCAL_FITTABLE parameters (not CSV columns)
  T = 1 day        → absorbed into fitted a, h (not a CSV column)

Caveats
-------
- The Holling disc equation assumes a linear functional response at low N
  (Na ≈ a*N for small N) and saturation at high N (Na → 1/h as N → ∞).
  At N → ∞ the formula approaches the maximum consumption rate 1/h.
- Within-cluster split: test_fit has the low-N (easy) regime;
  test_test has the high-N (saturating) regime. The formula's saturation
  asymptote is directly tested in test_test.
- Nelder-Mead is run in log-space (log_a = ln a, log_h = ln h) to enforce
  positivity, then back-transformed. The fit() below implements this.
- The per-experiment median RMSE is typically ~2-5 prey items; the cross-
  cluster pooled RMSE reflects both within-cluster residuals and the
  challenging high-N extrapolation tested in test_test.
"""

import numpy as np
from scipy.optimize import minimize

USED_INPUTS = ["prey_density"]
PAPER_REF = "summary_formula_holling_1959_bosomtwe_2025.md"
EQUATION_LOC = (
    "Bosomtwe et al. 2025, Insects 16(1):101, Section 2.5, PDF page 5 "
    "(no equation number; described as 'the Holling disc equation'). "
    "Original Holling 1959: Canadian Entomologist 91:385-398."
)

LAW_CONSTANTS = {}     # the form is the law; no paper-published scalar
OTHER_CONSTANTS = {}   # dimensionally clean; T=1 absorbed into a, h

LOCAL_FITTABLE = {
    "a": {"init": 0.5},   # attack rate; positive scalar
    "h": {"init": 0.1},   # handling time; positive scalar
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Fit (a, h) per cluster by minimising SSE in log-parameter space.

    X_fit[:,0] is prey_density; y_fit is prey_eaten.
    Optimisation is in (log_a, log_h) to enforce a > 0, h > 0.
    Returns {"a": ..., "h": ...}.
    """
    N = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    def sse(log_params):
        a_ = np.exp(log_params[0])
        h_ = np.exp(log_params[1])
        denom = 1.0 + a_ * h_ * N
        pred = a_ * N / np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        return np.sum((pred - y) ** 2)

    # Initial values from LOCAL_FITTABLE
    x0 = np.array([np.log(LOCAL_FITTABLE["a"]["init"]),
                   np.log(LOCAL_FITTABLE["h"]["init"])])
    result = minimize(sse, x0, method="Nelder-Mead",
                      options={"xatol": 1e-8, "fatol": 1e-8,
                               "maxiter": 10000})
    a_fit = float(np.exp(result.x[0]))
    h_fit = float(np.exp(result.x[1]))
    return {"a": a_fit, "h": h_fit}


def predict(X: np.ndarray, a: float, h: float) -> np.ndarray:
    """Holling disc equation: prey_eaten = a*N / (1 + a*h*N).

    X[:,0] = prey_density (N). a and h are per-cluster fitted values.
    """
    N = np.asarray(X[:, 0], dtype=float)
    a = float(a)
    h = float(h)
    denom = 1.0 + a * h * N
    # Guard against numerical zero denominator (should not occur for a,h > 0, N >= 0)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return a * N / denom
