"""1/v law — low-energy s-wave capture background (middle rung).

Below an isolated s-wave resonance, the neutron-capture cross section of
a compound-nucleus reaction varies as the inverse neutron speed,
sigma ∝ 1/v ∝ 1/sqrt(E) (Bohr-Wheeler 1939 / Wigner-Eisenbud
compound-nucleus picture).  In log space:

    log10(sigma) = A_V - 0.5 * log10(E),

with the slope FIXED at -1/2 by the 1/v physics; only the intercept A_V
(the cross-section scale) is fit.

This captures the thermal background trend (test r^2 ~ 0.37, rmse ~ 0.85)
but cannot represent the 4.9 eV resonance: at the resonance the measured
cross section spikes to ~33,000 barns, far above the smooth 1/v curve.
Recovering the 1/v slope of -1/2 is the first physical insight; the
resonance pole (next rung) is the second.

LAW_CONSTANTS — frozen, pre-fit on v2 train (slope fixed at -1/2)
----------------------------------------------------------------
- A_V = 1.3066   (log10 cross-section scale; sigma at E = 1 eV ~ 20 b)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).  The -0.5 exponent is the structural 1/v constant.
"""

import numpy as np

USED_INPUTS = ["E_eV"]
PAPER_REF = "summary_exfor_au197.md"
EQUATION_LOC = (
    "1/v law log10(sigma) = A_V - 0.5*log10(E) (Bohr-Wheeler 1939 "
    "compound-nucleus s-wave limit; summary_exfor_au197.md (i)).  "
    "Slope fixed at -1/2; intercept A_V pre-fit on v2 train."
)

LAW_CONSTANTS = {
    "A_V": 1.3066,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A_V: float = 1.3066) -> np.ndarray:
    """log10(sigma) = A_V - 0.5*log10(E); 1/v law."""
    E = np.asarray(X[:, 0], dtype=float)
    return A_V - 0.5 * np.log10(E)
