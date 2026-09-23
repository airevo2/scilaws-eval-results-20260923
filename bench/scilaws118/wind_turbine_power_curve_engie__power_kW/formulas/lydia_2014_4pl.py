"""Lydia et al. (2014) 4-parameter logistic wind power curve (Type I, frozen).

Lydia, Suresh Kumar, Selvakumar, Prem Kumar (2014), *Renewable and
Sustainable Energy Reviews* 30:452-460, §4.1.7 Eq. 29 (PDF p. 6),
applies the four-parameter logistic (4PL) function to wind-turbine
power-curve modelling:

    P(u) = a * (1 + m * exp(-u / tau)) / (1 + n * exp(-u / tau))

where u is hub-height wind speed, a is the asymptote (≈ rated power),
m and n control the curvature, and tau controls the inflection point.
Unlike the piecewise Carrillo cubic, the 4PL is smooth everywhere —
it captures the cut-in transition, the cubic climb, AND the rated
saturation in one continuous form — and so generally extrapolates
better to the saturation regime, as the v2 train/test split confirms.

The paper publishes NO universal (a, m, n, tau) — these are
turbine-specific fits.  For this Type I baseline, `a` is frozen to
the Senvion MM82 datasheet rated power (2050 kW), and (m, n, tau)
are pre-fit on the v2 TRAIN wind-speed band (4.69 - 7.48 m/s, 204k
rows).

LAW_CONSTANTS — frozen
----------------------
- A    =  2050.0   (Senvion MM82 datasheet rated power, kW)
- M    = -10.0     (pre-fit on v2 train; hits constrained bound but
                    stable across multi-start)
- N    =  114.51   (pre-fit on v2 train)
- TAU  =   1.7212  (pre-fit on v2 train, m/s)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["wind_speed_mps"]
PAPER_REF = "summary_formula_lydia_2014.md"
EQUATION_LOC = (
    "Lydia et al. 2014 §4.1.7 Eq. 29, PDF p. 6 — 4-parameter logistic; "
    "a fixed at Senvion MM82 rated power (2050 kW), (m, n, tau) "
    "pre-fit on v2 train band."
)

LAW_CONSTANTS = {
    "A":    2050.0,
    "M":   -10.0,
    "N":    114.5147,
    "TAU":  1.7212,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A: float = 2050.0, M: float = -10.0,
            N: float = 114.5147, TAU: float = 1.7212) -> np.ndarray:
    """P = A · (1 + M·exp(-u/TAU)) / (1 + N·exp(-u/TAU))."""
    u = np.asarray(X[:, 0], dtype=float)
    expo = np.exp(-u / TAU)
    return A * (1.0 + M * expo) / (1.0 + N * expo)
