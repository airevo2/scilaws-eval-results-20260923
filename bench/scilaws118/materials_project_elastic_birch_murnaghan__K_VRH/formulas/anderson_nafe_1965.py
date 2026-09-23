"""Anderson-Nafe (1965) log-linear bulk modulus scaling.

Anderson, O. L. & Nafe, J. E. (1965). The bulk modulus-volume relationship
for oxide compounds and related geophysical problems. J. Geophys. Res. 70,
3951. DOI: 10.1029/JZ070i016p03951.

Anderson & Nafe documented that for isoelectronic families of oxide
compounds, the bulk modulus follows a log-linear relationship with
atomic volume:

    log10(K) = a - (4/3) * log10(V_atomic)

The exponent -4/3 on log10(V_atomic) is the key LAW_CONSTANT. Adapted
to the full de Jong 2015 Materials Project dataset, the intercept a is
fit per crystal system on data/train.csv.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
- anderson_nafe_exponent: -4/3 = -1.33333...
  Source: Anderson & Nafe 1965 J. Geophys. Res. 70:3951. Documented in
  reference/summary_birch_murnaghan_kvrh.md §4.

OTHER_CONSTANTS — per-class intercepts fit on train.csv
--------------------------------------------------------
THESE ARE FIT-ON-DATA, NOT PAPER-PUBLISHED VALUES. The per-class
intercepts a_by_csid are fit on data/train.csv by log10-linear
regression (slope fixed at -4/3, intercept free). They live in
OTHER_CONSTANTS (NOT LAW_CONSTANTS) per the v2 FM-C10d contract — a
LAW_CONSTANT must be a paper-published number, while a dataset-fit
auxiliary number belongs in OTHER_CONSTANTS.

Anderson & Nafe's 1965 paper gave a universal slope (-4/3) per the
isoelectronic-oxide-family fit, but the per-family intercepts vary
with family (different `a` for SiO2-family, MgO-family, Al2O3-family,
…). The Materials Project elastic dataset does not partition by
isoelectronic family, so we instead group by Bravais crystal system
(a structural proxy). This is a defensible adaptation but the
resulting `a_by_csid` values are dataset-specific and dataset-fit;
the only Anderson-Nafe paper-frozen quantity in this baseline is the
exponent -4/3 (in LAW_CONSTANTS).

Crystal system id mapping (from prep_data.py):
  1 = triclinic    (no train examples; fallback = global)
  2 = monoclinic
  3 = orthorhombic
  4 = tetragonal
  5 = trigonal
  6 = hexagonal
  7 = cubic

Type designation: Type I — LOCAL_FITTABLE = {}, no fit() method.

Mapping: V_atomic_A3 (col 1) and crystal_system_id (col 2) from USED_INPUTS.
"""

import numpy as np

USED_INPUTS = ["V_atomic_A3", "crystal_system_id"]
PAPER_REF = "summary_birch_murnaghan_kvrh.md"
EQUATION_LOC = "Anderson & Nafe 1965 J. Geophys. Res. 70:3951 — log10(K) = a - (4/3)*log10(V); documented in summary_birch_murnaghan_kvrh.md §4"

LAW_CONSTANTS = {
    "anderson_nafe_exponent": -4.0 / 3.0,  # = -1.33333... — Anderson & Nafe 1965
}

OTHER_CONSTANTS = {
    # Per-class intercepts a fit on train.csv (slope fixed at -4/3).
    # crystal_system_id -> a (see summary_birch_murnaghan_kvrh.md Table §4)
    "a_by_csid_1": 3.677239,   # triclinic fallback = global (no train examples)
    "a_by_csid_2": 3.671468,   # monoclinic
    "a_by_csid_3": 3.674551,   # orthorhombic
    "a_by_csid_4": 3.705846,   # tetragonal
    "a_by_csid_5": 3.609841,   # trigonal
    "a_by_csid_6": 3.675350,   # hexagonal
    "a_by_csid_7": 3.675828,   # cubic
    "a_global":    3.677239,   # fallback for unrecognised crystal_system_id
}

LOCAL_FITTABLE = {}   # Type I


def predict(X: np.ndarray, anderson_nafe_exponent: float) -> np.ndarray:
    """K_VRH (GPa) from Anderson-Nafe log-linear scaling.

    X: (n, 2) — columns [V_atomic_A3, crystal_system_id] per USED_INPUTS.
    The LAW exponent arrives via the harness call predict(X, **LAW_CONSTANTS);
    the per-crystal-system intercepts are read from the OTHER_CONSTANTS dict.
    Returns K_VRH in GPa.
    """
    V = np.asarray(X[:, 0], dtype=float)
    csid = np.asarray(X[:, 1], dtype=np.int64)

    a_table = {
        1: OTHER_CONSTANTS["a_by_csid_1"],
        2: OTHER_CONSTANTS["a_by_csid_2"],
        3: OTHER_CONSTANTS["a_by_csid_3"],
        4: OTHER_CONSTANTS["a_by_csid_4"],
        5: OTHER_CONSTANTS["a_by_csid_5"],
        6: OTHER_CONSTANTS["a_by_csid_6"],
        7: OTHER_CONSTANTS["a_by_csid_7"],
    }
    a_global = OTHER_CONSTANTS["a_global"]
    a = np.full(csid.shape, a_global, dtype=np.float64)
    for k, v in a_table.items():
        a[csid == k] = v

    V_safe = np.where(V > 0, V, 1e-9)
    log10K = a + anderson_nafe_exponent * np.log10(V_safe)
    return np.power(10.0, log10K)
