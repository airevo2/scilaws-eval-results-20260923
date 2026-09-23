"""Neukum (2001) Production Function (NPF) — paper-published polynomial shape.

Neukum, Ivanov & Hartmann (2001), "Cratering Records in the Inner Solar
System in Relation to the Lunar Reference System", Space Science Reviews
96:55-86, DOI 10.1023/A:1011989004263.  The NPF replaces piecewise
power-law segments with a single 11th-degree polynomial in log10(D),
fitted simultaneously across the full diameter range (PDF p. 60, Eq. 2):

    log10 N(>=D) = a0 + sum_{j=1}^{11} a_j * [log10 D]^j

where D is crater diameter in km and N is the cumulative crater density
in km^-2.  The eleven shape coefficients a1..a11 are PAPER-PUBLISHED
(Neukum 2001 Table I, the "new" Ivanov et al. 2000 set, corrected at the
large-crater end using a 6700-crater Orientale-basin count); they encode
the universal lunar-derived production-function shape and are explicitly
described as fixed structural constants ("always the same regardless of
surface or time", Neukum 2001 §3).  Only the intercept a0 carries the
absolute crater-density scale, which depends on surface age via the
lunar chronology Eq. 5; for a given surface a0 is set so the polynomial
matches the observed N(1).

For this Type I baseline a1..a11 are frozen at their Neukum-2001
paper values and a0 is pre-fit on the v2 train rows (it absorbs the
Mars global surface age / Mars-Moon bolide-flux scaling).

Key scientific caveat (why this is the MIDDLE, not the best, rung):
the NPF shape was calibrated on lunar counts and scaled to Mars.  On the
Robbins & Hynek (2012) global Mars catalog the lunar-derived polynomial
is ~1 dex too steep at small diameters (D = 1-2 km) relative to the
observed Mars global SFD -- the global Martian surface is an age-mixed
average including resurfaced terrains, not a single pristine production
surface.  A single amplitude offset a0 therefore cannot align the shape,
leaving a residual the data-fit piecewise rung captures but the NPF
cannot (test nmse_log ~ 0.12 vs the piecewise rung's ~0.012).  This
mirrors the eclipsing-binary task, where the train-tuned Henry two-piece
slightly beats the paper-published Eker six-piece.

LAW_CONSTANTS — frozen
----------------------
- a0           pre-fit on v2 train (sets Mars global amplitude / age)
- a1 .. a11    Neukum 2001 Table I "new" (Ivanov 2000) paper-published
               polynomial shape coefficients

12 LAW_CONSTANTS total.

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["D_lower_km"]
PAPER_REF = "summary_formula_neukum_2001.md"
EQUATION_LOC = (
    "Neukum 2001 SSR 96:55, Eq. 2 (PDF p. 60) + Table I 'new' (Ivanov 2000) "
    "coefficients (PDF p. 62): log10 N = a0 + sum_{j=1}^{11} a_j (log10 D)^j.  "
    "a1..a11 paper-published (frozen); a0 pre-fit on v2 train."
)

LAW_CONSTANTS = {
    "A0": -1.693486,   # pre-fit on v2 train (Mars global amplitude)
    "A1": -3.557528,   # Neukum 2001 Table I "new" (Ivanov 2000) — paper-published
    "A2":  0.781027,
    "A3":  1.021521,
    "A4": -0.156012,
    "A5": -0.444058,
    "A6":  0.019977,
    "A7":  0.086850,
    "A8": -0.005874,
    "A9": -0.006809,
    "A10": 8.25e-4,
    "A11": 5.54e-5,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray,
            A0: float = -1.693486, A1: float = -3.557528, A2: float = 0.781027,
            A3: float = 1.021521, A4: float = -0.156012, A5: float = -0.444058,
            A6: float = 0.019977, A7: float = 0.086850, A8: float = -0.005874,
            A9: float = -0.006809, A10: float = 8.25e-4, A11: float = 5.54e-5) -> np.ndarray:
    """log10 N(>=D) = A0 + sum_{j=1}^{11} A_j (log10 D)^j; Neukum NPF."""
    D = np.asarray(X[:, 0], dtype=float)
    L = np.log10(D)
    a = [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]
    log_N = sum(a[j] * L ** j for j in range(12))
    return 10.0 ** log_N
