"""Patalas 1984 thermocline-depth scaling, reproduced in:
Boehrer, B. and Schultze, M. 2008.  "Stratification of Lakes."
Reviews of Geophysics 46, RG2005.  DOI: 10.1029/2006RG000210.
Section 2.1, paragraph [17], PDF page 3.

Formula
-------
    z_epi = a * A_s ** b

    where
      A_s   = lake surface area in km^2
      z_epi = epilimnion (thermocline) depth in m
      a     = 4.6  m    (LAW: Patalas 1984 via Boehrer & Schultze 2008 p.3 [17])
      b     = 0.205     (LAW: Patalas 1984 via Boehrer & Schultze 2008 p.3 [17])

Verbatim quote from PDF page 3, paragraph [17]:
  "The most central regression originates from Patalas [1984]
   zepi = 4.6 A^0.205 which is close to previously used formulas from
   Ventz [1972] and Fachbereichsstandard [1983] ... The differences
   between fitted curves (factor 1.5) give a good impression of the
   accuracy at which epilimnion thickness can be parameterized with
   surface area only."

LAW_CONSTANTS
-------------
  a_patalas = 4.6   — leading coefficient (m), Patalas 1984 calibration,
              reproduced at Boehrer & Schultze 2008 PDF page 3 paragraph [17].
  b_patalas = 0.205 — power-law exponent (dimensionless), same source.

OTHER_CONSTANTS
---------------
  (none — the formula is dimensionally self-contained in SI-adjacent units
   km^2 and m; no additional conversion factors are needed.)

Type designation
----------------
  TYPE I.  Single global power law; no per-lake fitted parameters.
  LOCAL_FITTABLE = {} (empty).  The formula applies to every lake-year row
  independently.

Column mapping
--------------
  Paper notation  | Released CSV column
  A               | A_s_km2   (lake surface area, km^2)
  z_epi           | z_t_m     (thermocline depth, m) — SR target, column 0

Caveats
-------
  - Validity range: 0.1 to ~1000 km^2 (temperate Northern Hemisphere lakes;
    Boehrer & Schultze 2008 paragraph [17]).  Prep_data.py enforces this range.
  - Published uncertainty: "factor 1.5" between competing power-law fits
    (Boehrer & Schultze 2008 p.3 [17]). Expected RMSE ~4-5 m, R^2 ~ 0.1-0.2.
  - The column Z_max_m is exposed in the released CSV but is NOT consumed by
    this formula.  It is retained so a future formula (e.g. capped variant) can
    use it.  Per data_spec S2-P3, all columns any candidate baseline might need
    must appear in released data.
  - Secondary baseline rationale (multi-baseline mandate S2-P7): Stage 1
    confirmed only one closed-form baseline with a citable OA source. The
    Patalas formula is paywalled and cited only through this OA paper. Competing
    regressions cited in the same paragraph (Ventz 1972, Fachbereichsstandard
    1983) lack explicit leading constants in the OA paper and are not separately
    citable.  Single-baseline shipping per data_spec §5.6 is therefore the
    correct outcome; this is documented here and in reference/summary_formula_*.md.
"""

import numpy as np

USED_INPUTS   = ["A_s_km2"]
PAPER_REF     = "summary_formula_boehrer_2008.md"
EQUATION_LOC  = "Section 2.1, paragraph [17], PDF page 3"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a_patalas": 4.6,    # m; Boehrer & Schultze 2008 PDF page 3 paragraph [17]
    "b_patalas": 0.205,  # dimensionless; same source
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}    # no additional conversion factors needed

LOCAL_FITTABLE = {}     # Type I — no per-cluster fitting


def predict(X: np.ndarray, a_patalas: float, b_patalas: float) -> np.ndarray:
    """Predict thermocline depth using the Patalas (1984) power law.

    Gold predict() style: LAW constants arrive as named params via the harness
    `predict(X, **LAW_CONSTANTS)`; no default argument values; structural
    literals (the column index) stay inline.

    Parameters
    ----------
    X : np.ndarray, shape (n, 1)
        Column 0 = A_s_km2 (lake surface area, km^2).
    a_patalas, b_patalas : float
        The two LAW_CONSTANTS forwarded by the harness.

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted thermocline depth in metres.
    """
    A_s = X[:, 0]
    return a_patalas * np.power(A_s, b_patalas)
