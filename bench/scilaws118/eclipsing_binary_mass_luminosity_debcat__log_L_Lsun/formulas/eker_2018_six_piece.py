"""Eker et al. (2018) six-piece classical M-L relation — paper-published slopes.

Eker, Bakış, Bilir, Soydugan, Steer, Soydugan, Bakış, Aliçavuş, Aslan,
Alpsoy (2018), "Interrelated Main-Sequence Mass-Luminosity, Mass-Radius
and Mass-Effective Temperature Relations", arXiv:1807.02568 (submitted
to MNRAS).  Calibrated a six-piece piecewise-linear M-L relation in
log10(M)-log10(L) space, with break points at mass boundaries
M = 0.45, 0.72, 1.05, 2.40, 7.00 M_sun corresponding to physically
meaningful transitions in stellar energy generation (PDF Table 4,
p. 18; physical motivation §4.1, pp. 24-26).

Within each of the six mass domains, the relation is
    log10(L/L_sun) = a · log10(M/M_sun) + b
with (a, b) fit by Eker et al. on 509 main-sequence eclipsing-binary
components compiled from their updated DEBCAT catalogue (pre-2018).

The published (a, b) per mass domain are reproduced verbatim below
(Eker 2018 Table 4, PDF p. 18):

| Domain (M_sun)     | a       | b      | sigma (dex) |
|--------------------|---------|--------|-------------|
| 0.179 < M ≤ 0.45  | 2.028   | -0.976 | 0.076       |
| 0.45  < M ≤ 0.72  | 4.572   | -0.102 | 0.109       |
| 0.72  < M ≤ 1.05  | 5.743   | -0.007 | 0.129       |
| 1.05  < M ≤ 2.40  | 4.329   | +0.010 | 0.140       |
| 2.40  < M ≤ 7.00  | 3.967   | +0.093 | 0.165       |
| 7.00  < M ≤ 31    | 2.865   | +1.105 | 0.152       |

This is the STRONGEST published M-L baseline available.  v2 uses the
paper-published (a, b) values directly — they are PAPER constants,
not refit on v2 train.  Because the v2 train/test split is
time-based (pre-2018 train / post-2018 test) and Eker's 509-star fit
sample is all pre-2018, the post-2018 v2 test stars are guaranteed
NOT in Eker's fit window — using Eker's published slopes on the v2
test set is a CLEAN held-out validation.

For test stars below 0.179 M_sun or above 31 M_sun (outside Eker's
fit window), this baseline extrapolates the boundary bin (no
re-extrapolation beyond paper validity).

LAW_CONSTANTS — frozen, Eker 2018 paper-published (Table 4)
-----------------------------------------------------------
12 slopes + intercepts (a, b) for 6 mass domains.

OTHER_CONSTANTS — frozen domain boundaries
------------------------------------------
5 break-point masses (Eker 2018 Table 4 + §4.1 physical motivation).

LOCAL_FITTABLE
--------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["log_M_Msun"]
PAPER_REF = "summary_formula+dataset_eker_2018.md"
EQUATION_LOC = (
    "Eker 2018 Table 4, PDF p. 18 — six-piece piecewise-linear M-L "
    "with paper-published (a, b) for each of 6 mass domains.  Domain "
    "boundaries 0.45/0.72/1.05/2.40/7.00 M_sun from §4.1, pp. 24-26."
)

LAW_CONSTANTS = {
    # Bin 1: 0.179 - 0.45 M_sun (ultra low-mass)
    "A1":  2.028,  "B1": -0.976,
    # Bin 2: 0.45 - 0.72 M_sun (very low-mass)
    "A2":  4.572,  "B2": -0.102,
    # Bin 3: 0.72 - 1.05 M_sun (low-mass)
    "A3":  5.743,  "B3": -0.007,
    # Bin 4: 1.05 - 2.40 M_sun (intermediate-mass)
    "A4":  4.329,  "B4":  0.010,
    # Bin 5: 2.40 - 7.00 M_sun (high-mass)
    "A5":  3.967,  "B5":  0.093,
    # Bin 6: 7.00 - 31 M_sun (very high-mass)
    "A6":  2.865,  "B6":  1.105,
}
OTHER_CONSTANTS = {
    "M_BREAK_1": 0.45,
    "M_BREAK_2": 0.72,
    "M_BREAK_3": 1.05,
    "M_BREAK_4": 2.40,
    "M_BREAK_5": 7.00,
}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray,
            A1: float = 2.028, B1: float = -0.976,
            A2: float = 4.572, B2: float = -0.102,
            A3: float = 5.743, B3: float = -0.007,
            A4: float = 4.329, B4: float =  0.010,
            A5: float = 3.967, B5: float =  0.093,
            A6: float = 2.865, B6: float =  1.105,
            M_BREAK_1: float = 0.45, M_BREAK_2: float = 0.72,
            M_BREAK_3: float = 1.05, M_BREAK_4: float = 2.40,
            M_BREAK_5: float = 7.00) -> np.ndarray:
    """Six-piece M-L: log10(L/L_sun) = a_i · log10(M/M_sun) + b_i in domain i."""
    log_M = np.asarray(X[:, 0], dtype=float)
    M = 10.0 ** log_M
    pred = np.zeros_like(log_M)
    # Use vectorised conditions
    bin1 = M <= M_BREAK_1
    bin2 = (M > M_BREAK_1) & (M <= M_BREAK_2)
    bin3 = (M > M_BREAK_2) & (M <= M_BREAK_3)
    bin4 = (M > M_BREAK_3) & (M <= M_BREAK_4)
    bin5 = (M > M_BREAK_4) & (M <= M_BREAK_5)
    bin6 = M > M_BREAK_5
    pred[bin1] = A1 * log_M[bin1] + B1
    pred[bin2] = A2 * log_M[bin2] + B2
    pred[bin3] = A3 * log_M[bin3] + B3
    pred[bin4] = A4 * log_M[bin4] + B4
    pred[bin5] = A5 * log_M[bin5] + B5
    pred[bin6] = A6 * log_M[bin6] + B6
    return pred
