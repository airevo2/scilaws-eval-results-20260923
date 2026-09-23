"""Langmuir adsorption isotherm — q_CO2 (mmol/g) from pressure P (bar).

Langmuir, I. (1918). The adsorption of gases on plane surfaces of glass,
mica and platinum. *J. Am. Chem. Soc.* 40(9):1361-1403.
DOI: 10.1021/ja02242a004.

Open-access source used: Swenson, H. and Stadie, N.P. (2019). Langmuir's
Theory of Adsorption: A Centennial Review. *Langmuir* 35(16):5409-5426.
DOI: 10.1021/acs.langmuir.9b00154. Eq. 1 (PDF p. 2):

    theta = K * P / (1 + K * P)

where theta = fractional surface coverage, K = Langmuir equilibrium constant
(affinity, Pa^-1), P = equilibrium partial pressure (Pa). The absolute
adsorbed amount is q = q_m * theta, giving the canonical form:

    q_CO2 = q_m * b * P / (1 + b * P)

with b = K (bar^-1) and q_m the monolayer saturation capacity (mmol/g).

The `1` in `(1 + b*P)` is a structural literal from the kinetic-theory
balance of adsorption and desorption rates (Swenson & Stadie 2019, §2,
PDF p. 2). It is NOT a tuneable constant — it is the defining structural
fingerprint of the single-site Langmuir isotherm.

Task framing (v2 Type II)
--------------------------
Cluster = (MOF, T_K) pair. Each cluster has 13 pressure points from a
single GCMC isotherm at fixed temperature for one MOF (DDEC charge,
UFF force field). Temperature is a cluster-level covariate (constant
within each cluster), NOT an input to predict(). Pressure P_bar is the
sole input to predict().

Data source: CRAFTED v2.0.1 (Lopes Oliveira et al. 2023, Scientific Data
10:230, DOI: 10.1038/s41597-023-02116-z; Zenodo DOI: 10.5281/zenodo.10120180).
GCMC-simulated CO2 adsorption on MOFs from CoRE-MOF-2014. Raw isotherm
files give pressure [Pa] and absolute CO2 uptake mean_volume [mol/kg]
(= mmol/g numerically). Converted to P_bar = P_Pa / 1e5.

LAW_CONSTANTS -- paper-published, frozen
----------------------------------------
None. The Langmuir FORM (the KP/(1+KP) hyperbola) is the scientific claim.
Both q_m and b are per-cluster (per-isotherm) material properties, refit
for every (MOF, T) combination; no universal numerical values are published.

OTHER_CONSTANTS -- universal / structural factors
-------------------------------------------------
(empty.) The `1` in `(1 + b*P)` is a fixed structural literal from the
kinetic-theory monolayer-saturation balance; it is NOT a tuneable constant.

LOCAL_FITTABLE -- per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- q_m : monolayer saturation capacity (mmol/g, > 0). The maximum absolute
        CO2 uptake at infinite pressure for one (MOF, T) cluster. Across
        CRAFTED MOFs (DDEC+UFF), values span roughly 0.5-35 mmol/g.
- b   : Langmuir affinity / equilibrium constant (bar^-1, > 0). Equivalent
        to Langmuir's K (Eq. 1 of Swenson & Stadie 2019). Decreases with
        temperature (higher T -> weaker adsorption). Across CRAFTED CO2
        isotherms (DDEC+UFF), b spans roughly 0.001-100 bar^-1.

init = None on both: fit() builds its own deterministic, data-derived start
via the Henry's-law slope (initial linear regime) and the maximum observed
uptake (plateau estimate for q_m). Single-start least_squares is sufficient
because the Langmuir form is globally convex in (q_m, b) over positive
domains for positive, monotone-increasing data (Swenson & Stadie 2019, §2).
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["P_bar"]
PAPER_REF = "summary_formula_swenson_stadie_2019.md"
EQUATION_LOC = (
    "Swenson & Stadie (2019) Eq. 1, PDF p. 2 -- theta = K*P / (1 + K*P); "
    "absolute uptake q_CO2 = q_m * theta = q_m * b * P / (1 + b * P). "
    "Originally: Langmuir (1918) J. Am. Chem. Soc. 40(9):1361-1403."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "q_m": {"init": None},
    "b":   {"init": None},
}


def _q_co2(P, q_m, b):
    """Langmuir isotherm: q_CO2 = q_m * b * P / (1 + b * P)."""
    bP = b * np.asarray(P, dtype=float)
    return q_m * bP / (1.0 + bP)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the Langmuir isotherm.

    Deterministic, data-derived start:
      q_m0 = 1.1 * max(y_fit)   -- slightly above observed max (plateau est.)
      b0   = henry_slope / q_m0
             where henry_slope = median(y/P) over the lowest-pressure quartile
             (Henry's-law regime estimate: q ~ q_m * b * P for b*P << 1).

    Single-start Levenberg-Marquardt / trust-region is sufficient because
    the Langmuir loss is unimodal over (q_m > 0, b > 0) for strictly
    positive, monotone-increasing data (Swenson & Stadie 2019, §2).
    """
    P = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # --- data-derived start ---
    y_max = float(np.max(y)) if len(y) > 0 else 1.0
    q_m0 = 1.1 * y_max if y_max > 0.0 else 1.0

    # Henry's-law slope from lowest quartile of pressure points
    if len(P) > 0:
        P_thresh = np.percentile(P, 25.0)
        low_mask = P <= P_thresh
        if not low_mask.any():
            low_mask = np.ones(len(P), dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(P[low_mask] > 0.0,
                              y[low_mask] / P[low_mask], 0.0)
        henry_slope = float(np.median(ratios[ratios > 0.0])) \
            if (ratios > 0.0).any() else 1e-3
    else:
        henry_slope = 1e-3

    b0 = henry_slope / q_m0 if q_m0 > 0.0 else 1e-3
    if not np.isfinite(b0) or b0 <= 0.0:
        b0 = 1e-3

    # Bounds: q_m in (0, 500] mmol/g; b in (0, 1e4] bar^-1
    # Upper q_m of 500 mmol/g is very generous (physical max ~40 mmol/g in CRAFTED).
    # Upper b of 1e4 bar^-1 covers the strongest adsorbers at low temperature.
    param_lo = [1e-6, 1e-8]
    param_hi = [500.0, 1e4]

    p0 = [
        min(max(q_m0, param_lo[0]), param_hi[0]),
        min(max(b0,   param_lo[1]), param_hi[1]),
    ]

    def residual(p):
        return _q_co2(P, p[0], p[1]) - y

    try:
        sol = least_squares(
            residual, p0,
            bounds=(param_lo, param_hi),
            method="trf",
            max_nfev=4000,
        )
        q_m, b = sol.x
        if not np.all(np.isfinite([q_m, b])):
            raise RuntimeError("non-finite fit result")
        return {"q_m": float(q_m), "b": float(b)}
    except Exception:                                   # noqa: BLE001
        return {"q_m": float(p0[0]), "b": float(p0[1])}


def predict(X: np.ndarray, q_m: float, b: float) -> np.ndarray:
    """Langmuir isotherm: q_CO2 = q_m * b * P / (1 + b * P).

    X: (n, 1) -- column [P_bar].
    Returns q_CO2 in mmol/g.
    """
    P = np.asarray(X[:, 0], dtype=float)
    return _q_co2(P, q_m, b)
