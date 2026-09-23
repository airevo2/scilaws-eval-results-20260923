"""Emig & Peltonen (2020) Lambert-W running performance formula.

NOTE: this module requires scipy.special.lambertw (W_{-1} real branch).

Citation: Emig, T. & Peltonen, J. (2020). Human running performance from
real-world big data. Nature Communications 11:4936.
DOI: 10.1038/s41467-020-18737-6. Eq. (1), PDF p. 3.

Formula (Eq. 1, PDF p. 3)
--------------------------
The fastest time T(d) for covering distance d is:

    T(d) = -(t_c / gamma_l) * (d / d_c) / W_{-1}[-(d/d_c) * exp(-1/gamma_l)]

where:
    d_c = v_m * t_c        [crossover distance, metres]
    t_c = 360 s            [universal crossover time; fixed for all subjects;
                            PDF p. 3 text and Methods p. 8:
                            "tc = 6 min is a good approximation on average"]
    v_m   : crossover velocity (m/s) — smallest velocity eliciting MAP
    gamma_l : endurance decay parameter (dimensionless)
    W_{-1}  : lower real branch of the Lambert W-function
              (real-valued for z in [-1/e, 0), which holds for all d and
               typical (v_m, gamma_l) values; PDF p. 3)

The SR target is average velocity:
    v_bar(d) = d / T(d)

LAW_CONSTANTS — the formula's defining coefficients (per sex)
-------------------------------------------------------------
The two characteristic parameters of Eq. (1) are the crossover velocity
v_m (aerobic-power index) and the endurance-decay rate gamma_l. They are
the formula's defining coefficients and the SR discovery target: Eq. (1)
is meaningless without recovering them. The paper fits them by
least-squares to race data and reports only fitting bounds
(2 m/s <= v_m <= 7 m/s, 0.039 <= gamma_l <= 0.135, PDF p. 8 Methods:
"We numerically minimized the sum of the squared ... 2 m s-1 <= vm <=
7 m s-1, 0.039 <= gamma_l <= 0.135"). Following the gold-contract axis
(defining-coefficient vs given), a coefficient being fit-from-data does
NOT bar it from LAW — every LAW constant is fit by someone; the right
test is whether it is the formula's characteristic parameter. These four
are (directly analogous to Riegel's per-sex a/b, also least-squares fits
to world records), so they are LAW_CONSTANTS.

Because Emig & Peltonen 2020 publish no frozen IAAF-record table, the
benchmark fits Eq. (1) to this task's own train split (aerobic regime,
distance_m >= 800 m) per sex by least-squares; the recovered values
(verified to reproduce by re-fitting on data/train.csv) are:

  vm_M      = 5.891400  m/s   crossover velocity, men
  gamma_l_M = 0.078037        endurance decay, men   (within paper bounds 0.039-0.135)
  vm_F      = 5.336700  m/s   crossover velocity, women
  gamma_l_F = 0.075702        endurance decay, women (within paper bounds 0.039-0.135)

The harness supplies them via predict(X, **LAW_CONSTANTS).

Note: fitting on all training distances including sprints (60-400 m),
which are outside the formula's validity domain (d < d_c), drives
gamma_l to high values (~0.12) and degrades test-set predictions.
Fitting only within the aerobic regime (d >= 800 m) respects the
formula's stated validity and gives better extrapolation to ultra
distances (test set: 20 km - 100 km).

OTHER_CONSTANTS (fixed structural constant of the model)
--------------------------------------------------------
  tc = 360.0 s   crossover time scale; fixed for all subjects, not a
                 defining coefficient (PDF p. 3 "tc = 6 min"; Methods p. 8)

Type designation: Type I.
Each row (one race distance x one sex) is independent. The formula
applies globally to all rows; v_m and gamma_l are fit once per sex
across all world records. No per-cluster fitting. LOCAL_FITTABLE = {}
(empty dict, per v2 contract).

Column mapping (released CSV -> formula variables)
--------------------------------------------------
  X[:,0] = distance_m   -> d     [m]
  X[:,1] = sex_M        -> sex_M [1 for male, 0 for female]

The released CSV encodes sex as a binary numeric: sex_M = 1 for male, 0 for
female. No one-hot expansion — the harness passes the two columns directly.

Caveats
-------
- The W_{-1} argument z = -(d/d_c) * exp(-1/gamma_l) must lie in [-1/e, 0).
  Numerically clipped to avoid complex drift from floating-point noise.
- For d < d_c (sprint distances), the formula is outside its validity
  domain (PDF p. 3: "condition d >= d_c is always satisfied for the race
  distances considered here"). These rows are included in the benchmark
  as an extrapolation probe; predictions may be less accurate.
- Correction note: the summary_formula_emig_2020.md originally listed
  gamma_l bounds as "1.039-5" — this is WRONG. The correct PDF Methods
  p. 8 value is 0.039 <= gamma_l <= 0.135. This module uses the correct
  values. The summary has been corrected.
"""

import numpy as np
from scipy.special import lambertw

USED_INPUTS = ["distance_m", "sex_M"]
PAPER_REF = "summary_formula_emig_2020.md"
EQUATION_LOC = "Emig & Peltonen 2020 Eq. (1), PDF p. 3 (bounds from Methods, PDF p. 8)"

# === LAW_CONSTANTS — the formula's defining coefficients (per sex) ===
# v_m (crossover velocity) and gamma_l (endurance decay) of Eq. (1).
# Fit by least-squares to the train aerobic regime (verified to reproduce
# on data/train.csv); paper gives only bounds, PDF p. 8 Methods.
LAW_CONSTANTS = {
    "vm_M":      5.891400,   # m/s   crossover velocity, men
    "gamma_l_M": 0.078037,   #       endurance decay, men
    "vm_F":      5.336700,   # m/s   crossover velocity, women
    "gamma_l_F": 0.075702,   #       endurance decay, women
}

OTHER_CONSTANTS = {
    "tc": 360.0,   # crossover time [s] — fixed structural constant; PDF p. 3 ("tc = 6 min"), Methods p. 8
}

LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters


def _emig_velocity(d, vm, gamma_l, tc=360.0):
    """Vectorised Emig velocity for one sex class.

    Parameters
    ----------
    d       : array of race distances [m].
    vm      : crossover velocity [m/s].
    gamma_l : endurance decay parameter (paper bounds 0.039-0.135, PDF p. 8).
    tc      : crossover time [s], fixed at 360 s (PDF p. 3).

    Returns
    -------
    Array of average velocities [m/s].
    """
    d = np.asarray(d, dtype=float)
    dc = vm * tc
    z = -(d / dc) * np.exp(-1.0 / gamma_l)
    # W_{-1} real for z in [-1/e, 0). Clip to avoid complex drift from float noise.
    z_clipped = np.clip(z, -1.0 / np.e + 1e-12, -1e-300)
    W = np.real(lambertw(z_clipped, k=-1))
    T = -(tc / gamma_l) * (d / dc) / W
    return d / T


def predict(X: np.ndarray, vm_M: float, gamma_l_M: float,
            vm_F: float, gamma_l_F: float) -> np.ndarray:
    """Predict average running velocity (m/s) from race distance and sex.

    Parameters
    ----------
    X : np.ndarray, shape (n, 2)
        Columns:
          0 — distance_m  [m]
          1 — sex_M       [1 for male, 0 for female]
    vm_M, gamma_l_M : Eq. (1) defining coefficients for males (LAW).
    vm_F, gamma_l_F : Eq. (1) defining coefficients for females (LAW).

    Returns
    -------
    np.ndarray, shape (n,) — predicted average velocity in m/s.
    """
    X = np.asarray(X, dtype=float)
    d = X[:, 0]
    sex_M = X[:, 1].astype(bool)
    sex_F = ~sex_M

    tc = OTHER_CONSTANTS["tc"]
    v_out = np.empty(len(d), dtype=float)
    if np.any(sex_M):
        v_out[sex_M] = _emig_velocity(d[sex_M], vm_M, gamma_l_M, tc=tc)
    if np.any(sex_F):
        v_out[sex_F] = _emig_velocity(d[sex_F], vm_F, gamma_l_F, tc=tc)
    return v_out
