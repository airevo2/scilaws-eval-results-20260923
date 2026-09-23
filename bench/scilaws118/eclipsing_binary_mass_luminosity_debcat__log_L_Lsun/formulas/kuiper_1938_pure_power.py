"""Kuiper (1938) pure power-law M-L relation — weakest rung.

Kuiper (1938), Astrophys. J. 88:472, "The Empirical Mass-Luminosity
Relation", first established that main-sequence stellar luminosities
scale approximately as a power of stellar mass.  In modern log-log
form:

    log10(L / L_sun) = ALPHA · log10(M / M_sun),

with ALPHA empirically fit across the available stellar sample.
Kuiper's 1938 fit gave ALPHA ≈ 3.5; later compilations refined the
slope to ≈ 3.7-4.4 depending on the mass range and sample.  For this
Type I baseline ALPHA is pre-fit on the v2 train (460 stars,
0.1 < M/M_sun < 2.4) by linear regression in log-log space, yielding
ALPHA ≈ 4.37.

Kuiper's pure power law is the simplest M-L baseline; it does not
account for (i) the well-documented break in slope around M ≈ 0.5 M_sun
where low-mass stars transition from radiative to convective energy
transport, nor (ii) the flattening of the M-L slope at very high
masses (M > 10 M_sun) where radiation pressure dominates.

LAW_CONSTANTS — frozen, pre-fit on v2 train
-------------------------------------------
- ALPHA = 4.3706    pre-fit power-law slope on v2 train log-log

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["log_M_Msun"]
PAPER_REF = "summary_formula_kuiper_1938.md"
EQUATION_LOC = (
    "Kuiper 1938 ApJ 88:472, empirical M-L power law.  ALPHA pre-fit "
    "on v2 train (460 stars, M = 0.1-2.4 M_sun) by linear log-log OLS."
)

LAW_CONSTANTS = {
    "ALPHA": 4.1390,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, ALPHA: float = 4.1390) -> np.ndarray:
    """log10(L/L_sun) = ALPHA · log10(M/M_sun)."""
    log_M = np.asarray(X[:, 0], dtype=float)
    return ALPHA * log_M
