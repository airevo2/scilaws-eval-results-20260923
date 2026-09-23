"""Hathaway-Wilson-Reichmann (HWR) cycle-shape formula — per-cycle fit.

Hathaway (2015) "The Solar Cycle", Living Reviews in Solar Physics 12:4,
Eq. (6), PDF p. 40 (originally Hathaway, Wilson & Reichmann 1994,
Solar Physics 151, 177):

    F(t) = A * ((t - t0) / b)**3
           * [exp(((t - t0) / b)**2) - c]**(-1)

where t is decimal-year time, A is cycle amplitude (SSN), t0 is the cycle
start time (decimal year, approximately 4 months before cycle minimum per
the paper), b is the rise-time parameter, and c = 0.8 is the asymmetry
constant fixed across individual cycles (PDF p. 40, paragraph after Eq. 6).

Per Hathaway et al. (1994) and summarised in Hathaway (2015) §4.5 (PDF p. 40):
"good fits to most cycles could be obtained with a fixed value for the
parameter c and a parameter b that is allowed to vary with the amplitude."
The standard analysis fits (A, t0, b) per cycle while keeping c = 0.8.

Parameter classification
------------------------
- c = 0.8 (LAW_CONSTANTS): fixed asymmetry constant stated on PDF p. 40.
  Paper text: "The average cycle is well fit with A = 195, b = 56, c = 0.8,
  and t0 = −4 months (prior to minimum)."
- Structural exponents 3 and 2 are inline numerals (not constants).
- A, t0, b are LOCAL_FITTABLE: per-cycle, fit by the harness on test_fit rows.

Unit convention
---------------
The paper uses months for t, t0, b. The CSV ships t_year in decimal years.
Because the formula depends only on the dimensionless ratio (t - t0) / b,
it is unit-invariant as long as t, t0, and b share the same unit.
We work in decimal years throughout:
    b  = 56 months / 12 = 4.6667 yr  (init value, paper average cycle)
    t0 = slightly before cycle minimum (init = cycle minimum decimal year)
    A  = 195 SSN (paper average-cycle amplitude, ISN v1 scale)
    In ISN v2 scale (current data) A is ~1.5× larger; init 195 is a safe
    starting point; the fitter will converge to the v2-scale amplitude.

Column mapping
--------------
    t (months, paper) -> t_year (decimal years, CSV column 0 of USED_INPUTS)
    F(t) (SSN)        -> ssn (target, column 0 of data CSVs)

Type II — per-cycle fit
------------------------
Each solar cycle (cluster) has its own (A, t0, b). The harness calls
fit(X_fit, y_fit, c=0.8) on test_fit rows, then predict(X_test, A=...,
t0=..., b=..., c=0.8) on test_test rows.

Validity caveat
---------------
Note: F(t0) = 0 / (exp(0²) − 0.8) = 0 / (1 − 0.8) = 0 / 0.2 = 0; the
formula has a regular zero at the cycle-start time, not an indeterminate
form. For t slightly above t0 the formula gives small positive values; for
t < t0 it gives negative values (the formula's domain is t > t0).
All released CSV rows are post-minimum (t > t0 for any reasonable t0).

Fit notes
---------
Nelder-Mead minimises MSE on (y_fit, predict(X_fit)) over (A, t0, b).
Multi-start is used (init = list) to avoid local minima: [195, 290] for A;
[min(t)-0.5, min(t)-0.33, min(t)-0.1] for t0; [3.5, 4.67, 6.0] for b.
"""

import numpy as np
from scipy.optimize import minimize

USED_INPUTS = ["t_year"]
PAPER_REF = "summary_formula+dataset_hathaway_2015.md"
EQUATION_LOC = "Eq. 6, PDF p. 40, hathaway_2015.pdf"

# --- Law constant (paper-published, frozen) ---
# Hathaway (2015) PDF p. 40: "The average cycle is well fit with
# A = 195, b = 56, c = 0.8, and t0 = -4 months (prior to minimum)."
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {"c": 0.8}

# === OTHER_CONSTANTS — none (empty) ===
OTHER_CONSTANTS = {}   # structural exponents 3 and 2 are inline numerals

# Per-cycle fit parameters — TYPE II
LOCAL_FITTABLE = {
    "A":  {"init": [195.0, 290.0]},          # cycle amplitude (SSN)
    "t0": {"init": [None, None, None]},       # cycle start (yr); set in fit()
    "b":  {"init": [3.5, 4.6667, 6.0]},      # rise-time parameter (yr)
}


def _hwrf(t, A, t0, b, c):
    """Core HWR scalar formula; safe at t == t0 (returns 0.0)."""
    u = (t - t0) / b
    denom = np.exp(u * u) - c
    # At t = t0: u = 0, exp(0) - c = 1 - 0.8 = 0.2, numerator = 0 -> F = 0
    with np.errstate(over="ignore", invalid="ignore"):
        val = A * (u ** 3) / denom
    # Replace NaN/Inf that arise when t << t0 or b very small
    val = np.where(np.isfinite(val), val, 0.0)
    return val


def fit(X_fit: np.ndarray, y_fit: np.ndarray, c: float = 0.8) -> dict:
    """Fit (A, t0, b) to one cycle's fit-window rows.

    Uses multi-start Nelder-Mead to avoid local minima. The t0 init values
    are built from the data: just before the observed start of the cluster.
    Returns dict with keys A, t0, b (matching LOCAL_FITTABLE).
    """
    t = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    t_min = float(t.min())
    # Three t0 inits: 6 months, 4 months, 1 month before earliest observed point
    t0_inits = [t_min - 0.5, t_min - 0.333, t_min - 0.083]
    A_inits = [195.0, 290.0]
    b_inits = [3.5, 4.6667, 6.0]

    def objective(params):
        A_, t0_, b_ = params
        if b_ <= 0.0 or A_ <= 0.0:
            return 1e12
        pred = _hwrf(t, A_, t0_, b_, c)
        return float(np.mean((pred - y) ** 2))

    best_mse = np.inf
    best_params = (195.0, t_min - 0.333, 4.6667)

    for A0 in A_inits:
        for t0_0 in t0_inits:
            for b0 in b_inits:
                try:
                    res = minimize(
                        objective,
                        x0=[A0, t0_0, b0],
                        method="Nelder-Mead",
                        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 5000},
                    )
                    if res.fun < best_mse:
                        best_mse = res.fun
                        best_params = tuple(res.x)
                except Exception:
                    pass

    A_fit, t0_fit, b_fit = best_params
    # Guard against degenerate fits
    if b_fit <= 0.0:
        b_fit = 4.6667
    if A_fit <= 0.0:
        A_fit = float(np.max(y)) if len(y) > 0 else 195.0
    return {"A": float(A_fit), "t0": float(t0_fit), "b": float(b_fit)}


def predict(X: np.ndarray, A: float, t0: float, b: float, c: float = 0.8) -> np.ndarray:
    """HWR cycle-shape formula (Eq. 6, Hathaway 2015, PDF p. 40).

    X[:, 0] = t_year (decimal year). Returns predicted SSN array.
    params: A (amplitude), t0 (cycle start year), b (rise-time, years).
    c is the LAW constant (0.8 from paper).
    """
    t = np.asarray(X[:, 0], dtype=float)
    return _hwrf(t, A, t0, b, c)
