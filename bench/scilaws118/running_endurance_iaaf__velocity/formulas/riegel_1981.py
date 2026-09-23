"""Riegel (1981) power-law endurance equation.

Citation: Riegel, P. S. (1981). Athletic Records and Human Endurance.
American Scientist 69(3): 285-290. Table 1, PDF p. 3 (article p. 287).

Formula (Table 1, PDF p. 3)
---------------------------
The race-time vs distance relationship follows a power law:

    T(d) = a * (d / 1000)^b       [d in m, T in minutes]

Rearranging to the SR target (average velocity in m/s):

    v(d) = d / (T_s(d))
         = d / (60 * a * (d / 1000)^b)
         = (1000^b / (60 * a)) * d^(1 - b)

LAW_CONSTANTS (Table 1, PDF p. 3 — Riegel 1981 endurance equation)
--------------------------------------------------------------------
Four paper-published constants, two per sex:
  a_M = 2.299   running men, distance in km, T in minutes
  b_M = 1.07732 running men
  a_F = 2.598   running women
  b_F = 1.08283 running women

These are Riegel's own least-squares fit values to world records at
publication time (Table 1, "Running, men" and "Running, women" rows).

OTHER_CONSTANTS (unit conversion factors)
------------------------------------------
  unit_factor = 1000.0   converts d from metres to km (denominator)
  time_factor = 60.0     converts T from minutes to seconds

Type designation: Type I.
Each row (one race distance x one sex) is independent. The formula
applies globally to all rows within each sex class. No per-cluster
fitting. LOCAL_FITTABLE = {} (empty dict, per v2 contract).

Column mapping (released CSV -> formula variables)
--------------------------------------------------
  X[:,0] = distance_m   -> d      [m]
  X[:,1] = sex_M        -> sex_M  [1 for male, 0 for female]

The released CSV encodes sex as a binary numeric: sex_M = 1 for male, 0 for
female. No one-hot expansion — the harness passes the two columns directly.

Caveats
-------
Riegel (1981) calibrated this formula on records from ~1.5 km (1 mile)
to 42.2 km (marathon). Sprint distances (< 800 m) and ultra distances
(> 42.2 km) lie outside the original calibration range. Predictions at
60-400 m (sprint) or 50-100 km (ultra) are extrapolations.
"""

import numpy as np

USED_INPUTS = ["distance_m", "sex_M"]
PAPER_REF = "summary_formula_riegel_1981.md"
EQUATION_LOC = "Riegel 1981 Table 1, PDF p. 3 (article p. 287)"

LAW_CONSTANTS = {
    "a_M": 2.299,     # running men, d in km, T in minutes — Table 1, PDF p. 3
    "b_M": 1.07732,   # running men — Table 1, PDF p. 3
    "a_F": 2.598,     # running women, d in km, T in minutes — Table 1, PDF p. 3
    "b_F": 1.08283,   # running women — Table 1, PDF p. 3
}

OTHER_CONSTANTS = {
    "unit_factor": 1000.0,  # converts d from metres to kilometres in T formula
    "time_factor": 60.0,    # converts T from minutes to seconds
}

LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, a_M: float, b_M: float,
            a_F: float, b_F: float) -> np.ndarray:
    """Predict average running velocity (m/s) from race distance and sex.

    Parameters
    ----------
    X : np.ndarray, shape (n, 2)
        Columns:
          0 — distance_m  [m]
          1 — sex_M       [1 for male, 0 for female]
    a_M, b_M : Riegel 1981 power-law coefficients for males.
    a_F, b_F : Riegel 1981 power-law coefficients for females.

    Returns
    -------
    np.ndarray, shape (n,) — predicted average velocity in m/s.
    """
    X = np.asarray(X, dtype=float)
    d = X[:, 0]       # distance in metres
    sex_M = X[:, 1]   # 1 for male, 0 for female
    sex_F = 1.0 - sex_M   # complementary binary indicator

    # Select per-row parameters based on sex indicator
    a = sex_M * a_M + sex_F * a_F
    b = sex_M * b_M + sex_F * b_F

    # T(d_km) = a * (d_km)^b  [minutes];  d_km = d / 1000
    # T_s(d)  = 60 * a * (d / 1000)^b   [seconds]
    # v(d)    = d / T_s = (1000^b / (60 * a)) * d^(1 - b)
    unit_factor = OTHER_CONSTANTS["unit_factor"]
    time_factor = OTHER_CONSTANTS["time_factor"]
    v = (unit_factor ** b / (time_factor * a)) * (d ** (1.0 - b))
    return v
