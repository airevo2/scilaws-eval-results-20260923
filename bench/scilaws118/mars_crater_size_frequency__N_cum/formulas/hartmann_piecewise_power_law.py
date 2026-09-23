"""Hartmann-style piecewise power-law crater SFD — best rung.

Hartmann's production function (HPF) is the historical alternative to
Neukum's polynomial NPF.  Hartmann (1984, 1999, 2005; reviewed in
Hartmann & Neukum 2001, Space Science Reviews 96:165) represents the
cumulative crater size-frequency distribution as a sequence of straight
power-law segments in log-log space, joined at break diameters where the
SFD slope changes.  Hartmann & Neukum (2001, §3) note that the HPF and
NPF agree to ~30% over most of the diameter range but diverge by a
factor 2-3 at D ~ 2-11 km -- the multi-slope region that motivates a
piecewise representation.

This baseline implements a continuous three-segment broken power law in
log10(D) with two break diameters:

    log10 N(>=D) = C0 + S1·u                              (u = log10 D <= K1)
                 = C0 + S1·K1 + S2·(u - K1)              (K1 < u <= K2)
                 = C0 + S1·K1 + S2·(K2-K1) + S3·(u-K2)   (u > K2)

The three slopes S1, S2, S3 and the two break points (K1, K2 in log10 D)
are pre-fit on the v2 train rows.  The fitted break diameters land at
D1 ~ 1.9 km and D2 ~ 31 km -- physically the simple-to-complex crater
transition (Ivanov 2001 places D* ~ 7 km but the SFD slope break is
sample-dependent) and the onset of the steep large-crater branch.

Because the slopes adapt to the actual Robbins & Hynek (2012) Mars
global counts (rather than being imported from the lunar-derived NPF
shape), this rung achieves the lowest error of the three, including on
the range-OOD test (large basins): the third segment's steep slope
extrapolates the basin branch correctly (test nmse_log ~ 0.012, vs the
NPF's ~0.12 and the single power law's ~2.8).

LAW_CONSTANTS — frozen, pre-fit on v2 train (56 bins, D < 128 km)
-----------------------------------------------------------------
- C0 = -2.582651     intercept (log10 N at D = 1 km)
- S1 = -1.583928     slope, D < 10^K1 (~1.9 km)
- S2 = -1.075627     slope, 10^K1 < D < 10^K2 (~1.9-31 km)
- S3 = -2.590839     slope, D > 10^K2 (~31 km; large-crater branch)
- K1 =  0.282920     first break in log10 D (D1 = 1.918 km)
- K2 =  1.485987     second break in log10 D (D2 = 30.62 km)

6 LAW_CONSTANTS total.

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["D_lower_km"]
PAPER_REF = "summary_supporting_hartmann_2001.md"
EQUATION_LOC = (
    "Hartmann piecewise power-law production function (Hartmann 1984/1999/2005; "
    "Hartmann & Neukum 2001 SSR 96:165, §3, Fig. 2): continuous broken power "
    "law in log10 D with two break diameters.  Slopes + breaks pre-fit on v2 "
    "train."
)

LAW_CONSTANTS = {
    "C0": -2.582651,
    "S1": -1.583928,
    "S2": -1.075627,
    "S3": -2.590839,
    "K1":  0.282920,
    "K2":  1.485987,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray,
            C0: float = -2.582651, S1: float = -1.583928,
            S2: float = -1.075627, S3: float = -2.590839,
            K1: float = 0.282920, K2: float = 1.485987) -> np.ndarray:
    """Continuous 3-segment broken power law in log10 D; Hartmann-style SFD."""
    D = np.asarray(X[:, 0], dtype=float)
    u = np.log10(D)
    log_N = C0 + S1 * u
    log_N = np.where(u > K1, C0 + S1 * K1 + S2 * (u - K1), log_N)
    log_N = np.where(u > K2, C0 + S1 * K1 + S2 * (K2 - K1) + S3 * (u - K2), log_N)
    return 10.0 ** log_N
