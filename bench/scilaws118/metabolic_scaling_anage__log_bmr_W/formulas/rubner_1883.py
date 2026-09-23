"""Rubner (1883) surface-area BMR-vs-mass law (Type I, frozen).

Rubner (1883), Zeitschrift fur Biologie 19:535-562, advanced the
"surface law" of metabolism: basal heat production is proportional to
body-surface area, which for geometrically similar animals scales as
mass to the 2/3 power.  The functional form is therefore

    B = B0 * M^b,  with  b = 2/3 ≈ 0.6667.

Rubner's surface law was the dominant interspecific scaling theory
before Kleiber (1932) showed empirically that b is closer to 3/4 than
2/3; the 2/3 vs 3/4 controversy continued through the 20th century and
remains a touchstone in allometric scaling debates (e.g.,
Heusner 1982 vs Feldman & McMahon 1983).

Rubner himself did not tabulate a universal B0 in W/g^(2/3) — his
analyses were per-species in kcal / m^2 of body surface / day.  For
this Type I benchmark we freeze B0 to the value that makes Rubner and
Kleiber agree at the geometric-mean training mass (M = 391 g, the
sqrt of the train mass band 39 g - 3924 g):

    log10(B0_Rubner) = log10(B0_Kleiber) + (0.75 - 0.6667) * log10(391)
                     = -1.713 + 0.0833 * 2.592
                     = -1.713 + 0.216
                     = -1.497.

This calibration makes the two baselines numerically equivalent at the
center of the train distribution but diverge on the extrapolation OOD
test set — the benchmark's central question.  The calibration is
documented here rather than refit at evaluation time, so the value is
a fixed LAW_CONSTANT.

LAW_CONSTANTS — frozen
----------------------
- B_EXPONENT = 0.6667  (Rubner 1883 surface law, 2/3)
- LOG_B0     = -1.497  (calibrated to Kleiber 1932 at M = 391 g)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["body_mass_g"]
PAPER_REF = "summary_formula_rubner_1883.md"
EQUATION_LOC = (
    "Rubner 1883 surface-area scaling B ∝ M^(2/3); prefactor calibrated "
    "to Kleiber 1932 at the geometric-mean train mass (M = 391 g)."
)

LAW_CONSTANTS = {
    "B_EXPONENT": 0.6667,
    "LOG_B0":    -1.497,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray,
            B_EXPONENT: float = 0.6667,
            LOG_B0: float = -1.497) -> np.ndarray:
    """log10(B) = LOG_B0 + B_EXPONENT * log10(M)."""
    M = np.asarray(X[:, 0], dtype=float)
    return LOG_B0 + B_EXPONENT * np.log10(M)
