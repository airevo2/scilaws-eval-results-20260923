"""Nedora et al. (2020/2022) two-parameter polynomial disk-mass fit.

Nedora et al. 2020 (CQG 39, 015008, 2022) Eq. (6), Table VII for M_disk:

    M_disk = b0 + b1*q + b2*Lambda_tilde
           + b3*q**2 + b4*q*Lambda_tilde
           + b5*Lambda_tilde**2

Recommended coefficients are the M0RefSet & M0/M1Set row of Table VII
(nedora_2020.txt PDF lines 2489-2492; Sec. V recommends
"the Eq. (6) calibrated with datasets with the most advanced physics
input, i.e., MO/M1Set and MORefSet"):

    b0 = -1.85, b1 = 2.59, b2 = 7.07e-4,
    b3 = -7.33e-1, b4 = -8.08e-4, b5 = 2.75e-7.

Output is M_disk in M_sun (linear, NOT log10). The six coefficients above
were re-verified byte-for-byte against the M0/M1Set row of Table VII
(nedora_2020.txt L2490-2492) during the 2026-05-29 audit — they are correctly
transcribed. The paper is INTERNALLY INCONSISTENT on the output scale: Fig. 8
says "the calibration was performed for log10(M_disk)", but the results tables
write "Mdisk = P22(q, Λ̃)" (linear) directly. The linear reading is the only
physically admissible one here: applying P22 as 10**P22 on the released
(q, Λ̃) ranges yields M_disk ≈ 1-5 M_sun (impossible — disk masses are
≤ 0.31 M_sun), whereas the linear reading yields ~0.04-0.3 M_sun, matching the
M0RefSet mean disk mass of (0.12 ± 0.05) M_sun reported in PDF Sec. V. We
therefore apply P22 linearly, consistent with the paper's own "Mdisk = P22"
results notation.

Convention discrepancy with the released CSV. The paper defines
q = M_A/M_B with M_A >= M_B, so q >= 1. The released CSV uses
q = M_1/M_2 with M_1 <= M_2, so 0 < q <= 1. The predict body inverts q:
q_paper = 1.0 / q_csv. Lambda_tilde is symmetric under (1<->2) by
construction (Eq. 1 of Nedora 2020) and does not change.

The polynomial may extrapolate to negative values at extreme inputs;
no floor is imposed by the paper.

Calibration domain: q in [1, 2.06], Lambda_tilde in [50, 3196],
M_chirp in [1.04, 1.74] M_sun, on the cumulative MORefSet+MO/M1Set
calibration set.

All six coefficients are universal across the dataset; LOCAL_FITTABLE is
empty for this Setting 1 Type I task.
"""

import numpy as np

USED_INPUTS = ["q", "Lambda_tilde"]
PAPER_REF = "summary_formula_dataset_nedora_2020.md"
EQUATION_LOC = "Eq. 6 + Table VII MO/M1Set row, p. 13"
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "b0": -1.85,
    "b1": 2.59,
    "b2": 7.07e-4,
    "b3": -7.33e-1,
    "b4": -8.08e-4,
    "b5": 2.75e-7,
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X, b0=LAW_CONSTANTS["b0"], b1=LAW_CONSTANTS["b1"], b2=LAW_CONSTANTS["b2"],
            b3=LAW_CONSTANTS["b3"], b4=LAW_CONSTANTS["b4"], b5=LAW_CONSTANTS["b5"]):
    q_csv = np.asarray(X[:, 0], dtype=float)
    Lt = np.asarray(X[:, 1], dtype=float)
    # Paper convention: q_paper = M_A / M_B >= 1; released CSV uses q <= 1.
    q = 1.0 / q_csv
    return (b0 + b1 * q + b2 * Lt
            + b3 * q * q + b4 * q * Lt + b5 * Lt * Lt)
