"""He et al. (2011) double-exponential capacity-fade law (Type I, frozen).

He, Williard, Osterman & Pecht (2011), J. Power Sources 196:10314-10321,
fit a double-exponential to lithium-ion battery capacity-fade data:

    Q(k) = a · exp(b · k) + c · exp(d · k),

with cycle index k.  The two exponentials phenomenologically capture
the two-stage decay observed in Li-ion cells: a fast initial SEI
formation transient (one exponential with rate ~10^{-2} per cycle)
and a slower long-term degradation (the other exponential with rate
~10^{-3} per cycle).  The (a, b, c, d) coefficients are cell-specific
and the paper publishes per-cell fits.

For this Type I baseline, (A, B, C, D) are pre-fit on the v2 TRAIN
cycle band (middle 60%, cycles ~32-128) and frozen as LAW_CONSTANTS.
The four parameters give the double-exp more flexibility than
Pinson's three-parameter sqrt form, but on this small dataset the
extra parameter risks overfitting the train band.

LAW_CONSTANTS — frozen, pre-fit on v2 train band
------------------------------------------------
- A =  2.0288    Ah         (initial-fast-decay amplitude)
- B = -0.003649  1/cycle    (initial-fast-decay rate)
- C =  0.0042    Ah         (slow-recovery component amplitude)
- D =  0.02736   1/cycle    (slow-recovery rate)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["cycle_index"]
PAPER_REF = "summary_formula+dataset_he_2011.md"
EQUATION_LOC = (
    "He et al. 2011, Eq. 9 — double-exponential Q(k) = a·exp(b·k) + "
    "c·exp(d·k); (a, b, c, d) frozen from v2 train-band nonlinear LS."
)

LAW_CONSTANTS = {
    "A":  2.0288,
    "B": -0.003649,
    "C":  0.0042,
    "D":  0.02736,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A: float = 2.0288, B: float = -0.003649,
            C: float = 0.0042, D: float = 0.02736) -> np.ndarray:
    """Q(k) = A · exp(B·k) + C · exp(D·k)."""
    k = np.asarray(X[:, 0], dtype=float)
    return A * np.exp(B * k) + C * np.exp(D * k)
