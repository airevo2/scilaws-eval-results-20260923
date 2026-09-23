"""Heligman-Pollard 8-parameter law of mortality (model 1).

Heligman & Pollard (1980), Journal of the Institute of Actuaries 107(1)
pp. 49-80, Eq. 1 — a single closed-form expression for the full age
pattern of human mortality:

    q_x / p_x = A^((x+B)^C) + D * exp(-E * (ln x - ln F)^2) + G * H^x

with p_x = 1 - q_x. Three additive components:
  comp 1  A^((x+B)^C)            — rapidly declining childhood mortality
  comp 2  D*exp(-E*(ln x-ln F)^2) — lognormal young-adult accident hump
  comp 3  G*H^x                  — Gompertz senescent rise

The benchmark target is the central death rate: from the odds q/p the
HMD relation gives q = (q/p)/(1+q/p) and M(x) = q / (1 - q/2), so

    log_m_full = ln M(x) = ln( q / (1 - q/2) ).

All eight parameters A-H are per-population fits — Heligman & Pollard
refit them by nonlinear least squares separately for each dataset; none
is a universal constant. This is a Type II task: A-H are LOCAL_FITTABLE,
fitted per country cluster.

The `year` input is not consumed: the HP law models the age pattern; a
country's within-cluster year variation is absorbed into the fitted A-H.

Caveats (documented in data/report.md):
- HP was validated on ages 0-85; ages 86-95 in HMD are Kannisto-smoothed
  extrapolations and the HP curve is extrapolated there.
- At x=0 the comp-2 lognormal kernel needs ln(0); the hump vanishes at
  age 0, so comp 2 is set to 0 for x < 0.5.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The HP functional form is the scientific claim; A-H are per-cluster
fits (Heligman & Pollard 1980 §5; Table 1 gives six per-dataset fits).

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.)

LOCAL_FITTABLE — per-cluster, computed by fit() via nonlinear least squares
---------------------------------------------------------------------------
- A : childhood-mortality level (~q_1).
- B : age displacement of infant mortality.
- C : rate of childhood-mortality decline.
- D : severity (amplitude) of the accident hump.
- E : spread (inverse width) of the accident hump.
- F : location (centre age) of the accident hump.
- G : base level of senescent (Gompertz) mortality.
- H : Gompertz growth factor of senescent mortality.
init = None on all eight: fit() builds its own data-derived start
(a Gompertz OLS on adult ages seeds G, H; the rest use HP-typical values).
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["age"]
PAPER_REF = "summary_formula_heligman_1980.md"
EQUATION_LOC = (
    "Heligman & Pollard (1980) Eq. 1, PDF p. 1 / journal p. 49 — "
    "q/p = A^((x+B)^C) + D*exp(-E*(ln x - ln F)^2) + G*H^x; "
    "central rate M = q/(1 - q/2)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "A": {"init": None}, "B": {"init": None}, "C": {"init": None},
    "D": {"init": None}, "E": {"init": None}, "F": {"init": None},
    "G": {"init": None}, "H": {"init": None},
}

# Parameter bounds for the nonlinear fit (order A, B, C, D, E, F, G, H).
_LO = [1e-7, 0.0,  0.01, 0.0,  0.05, 0.5,  1e-10, 1.0]
_HI = [0.9,  10.0, 2.0,  0.9,  100.0, 90.0, 0.5,   2.0]


def _log_m(age, A, B, C, D, E, F, G, H):
    """Heligman-Pollard log central death rate at given ages."""
    age = np.asarray(age, dtype=float)
    comp1 = A ** ((age + B) ** C)
    x_safe = np.clip(age, 1e-9, None)
    comp2 = D * np.exp(-E * (np.log(x_safe) - np.log(F)) ** 2)
    comp2 = np.where(age < 0.5, 0.0, comp2)          # accident hump vanishes at age 0
    comp3 = G * H ** age
    odds = np.clip(comp1 + comp2 + comp3, 0.0, None)
    q = odds / (1.0 + odds)
    q = np.clip(q, 0.0, 1.0 - 1e-9)
    M = np.clip(q / (1.0 - q / 2.0), 1e-300, None)
    return np.log(M)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Nonlinear least-squares fit of the 8 HP parameters to log_m_full.

    Data-derived start: a Gompertz OLS of log_m on age over adult ages
    (40-90, where the senescent term dominates) seeds G and H; the
    childhood and accident-hump parameters start from HP-typical values.
    """
    age = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # Seed G, H from a Gompertz line on adult ages: log m ~ ln G + age*ln H.
    adult = (age >= 40) & (age <= 90)
    if adult.sum() >= 5:
        coef, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(adult.sum()), age[adult]]), y[adult], rcond=None)
        G0 = float(np.clip(np.exp(coef[0]), 1e-10, 0.5))
        H0 = float(np.clip(np.exp(coef[1]), 1.0, 2.0))
    else:
        G0, H0 = 5e-5, 1.10

    # HP-typical starts (Heligman & Pollard 1980 Table 1 mid-ranges).
    p0 = [5e-4, 0.01, 0.12, 1e-3, 8.0, 20.0, G0, H0]
    p0 = [float(np.clip(v, lo, hi)) for v, lo, hi in zip(p0, _LO, _HI)]
    names = ["A", "B", "C", "D", "E", "F", "G", "H"]

    def residual(p):
        return _log_m(age, *p) - y

    try:
        sol = least_squares(residual, p0, bounds=(_LO, _HI),
                            method="trf", max_nfev=4000)
        return {n: float(v) for n, v in zip(names, sol.x)}
    except Exception:                                # noqa: BLE001 — graceful fallback
        # Degrade to a Gompertz-only fit: no childhood/hump terms.
        return {"A": 1e-7, "B": 0.01, "C": 0.12, "D": 0.0,
                "E": 8.0, "F": 20.0, "G": G0, "H": H0}


def predict(X: np.ndarray, A: float, B: float, C: float, D: float,
            E: float, F: float, G: float, H: float) -> np.ndarray:
    """log_m_full = ln M(x) from the Heligman-Pollard law.

    X: (n, 1) — column 0 is age.
    """
    return _log_m(X[:, 0], A, B, C, D, E, F, G, H)
