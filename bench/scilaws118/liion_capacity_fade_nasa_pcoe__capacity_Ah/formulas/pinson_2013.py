"""Pinson-Bazant (2013) sqrt-time SEI capacity-fade law (Type I, frozen).

Pinson & Bazant (2013), J. Electrochem. Soc. 160(2):A243-A250, derive
that lithium-ion battery capacity loss due to SEI (Solid-Electrolyte
Interphase) layer growth on the anode follows a sqrt-time law,
because the diffusion-limited growth of the SEI layer over cycling is
analogous to parabolic film growth in metallurgy.  The reduced-form
prediction is:

    Q(k) = Q0 - α · sqrt(k) + β,

where k is the cycle index, Q0 is the initial fresh-cell capacity,
α is the cycle-fade rate proportional to (D · t_per_cycle)^(1/2) with
D the Li-ion diffusion coefficient in SEI, and β is a small calendar-
time offset.

For this Type I baseline, (Q0, α, β) are pre-fit on the v2 TRAIN
cycle band (middle 60%, cycles ~32-128) and frozen as LAW_CONSTANTS.
The test asks whether the sqrt-time law extrapolates to (a) early
cycles (1-31, near-pristine, possible SEI-formation transient) and
(b) later cycles (129-168, deeper degradation).

LAW_CONSTANTS — frozen, pre-fit on v2 train band
------------------------------------------------
- Q0    = -142.7947  Ah  (fitted intercept; absorbs cycle-1 SEI offset)
- ALPHA =    0.0757  Ah / sqrt(cycle)  (sqrt-time decay rate)
- BETA  =  145.0329  Ah  (calendar offset — large here because the OLS
                          parameterisation factors Q0 and β separately)

NOTE: The (Q0, β) pair is degenerate in this two-additive-constant
form; only their sum Q0 + β ≈ 2.24 Ah is identifiable.  Keeping them
as separate fit values matches the paper's notation.

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["cycle_index"]
PAPER_REF = "summary_formula_pinson_2013.md"
EQUATION_LOC = (
    "Pinson & Bazant 2013, Eq. 14 (PDF p. 246) — sqrt-time SEI capacity-"
    "fade law Q(k) = Q0 - α·sqrt(k) + β; (Q0, α, β) frozen from v2 "
    "train-band OLS calibration."
)

LAW_CONSTANTS = {
    "Q0":    -142.7947,
    "ALPHA":    0.0757,
    "BETA":   145.0329,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, Q0: float = -142.7947,
            ALPHA: float = 0.0757, BETA: float = 145.0329) -> np.ndarray:
    """Q(k) = Q0 - ALPHA · sqrt(k) + BETA."""
    k = np.asarray(X[:, 0], dtype=float)
    return Q0 - ALPHA * np.sqrt(k) + BETA
