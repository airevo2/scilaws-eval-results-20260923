"""Coughlin et al. (2018) BNS ejecta-velocity fit.

Coughlin et al. (2018) Appendix E, Eq. E9 (PDF p. 15) gives a single-formula
fit for the NR ejecta velocity in units of c:

    vej = e * M1 * (f*C1 + 1) / M2 + e * M2 * (f*C2 + 1) / M1 + g,

with universal fitted constants e = -0.3292, f = -1.633, g = 0.720 (PDF p.
15, line below Eq. E9). The fit is recalibrated on a larger NR catalogue
than Dietrich and Ujevic (2017) and is symmetric under (M1, C1) <-> (M2, C2)
by construction. The three coefficients are stored in LAW_CONSTANTS:
universal across the calibration corpus, so the harness re-fits them once
on train.csv with the published values as OTHER_CONSTANTS initial values.

Symbol map: paper M1, M2 -> CSV M1, M2 (M_sun); paper C1, C2 -> CSV C1, C2
(dimensionless compactness); output in units of c.

Validity caveat: the paper uses a flat prior 0 <= vej <= 0.3 c on the
sampled velocity (PDF p. 4); the formula's calibration domain is the
compiled NR set listed at PDF p. 15 (Dietrich 2017b, Hotokezaka 2013,
Sekiguchi 2016, Bovard 2017, Shibata 2017, Ciolfi 2017, etc.).
"""

import numpy as np

USED_INPUTS = ["M1", "M2", "C1", "C2"]
PAPER_REF = "summary_formula_coughlin_2018.md"
EQUATION_LOC = "Eq. E9, p. 15"
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {"e": -0.3292, "f": -1.633, "g": 0.720}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X, e=LAW_CONSTANTS["e"], f=LAW_CONSTANTS["f"], g=LAW_CONSTANTS["g"]):
    X = np.asarray(X, dtype=float)
    M1 = X[:, 0]
    M2 = X[:, 1]
    C1 = X[:, 2]
    C2 = X[:, 3]
    return e * M1 * (f * C1 + 1.0) / M2 + e * M2 * (f * C2 + 1.0) / M1 + g
