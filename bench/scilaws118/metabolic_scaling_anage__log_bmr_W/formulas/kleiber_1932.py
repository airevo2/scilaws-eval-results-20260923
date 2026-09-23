"""Kleiber (1932) interspecific 3/4-power BMR-vs-mass law (Type I, frozen).

Kleiber (1932), Hilgardia 6(11):315-353, derived the canonical mammalian
basal metabolic rate scaling law

    B = B0 * M^b,

with b = 3/4 (paginated p. 320 / PDF p. 8 — recommended structural
value) and B0 = 71.2 kcal/(24 hr * kg^0.75) for humans (paginated p. 345
/ PDF p. 33).  Converting to SI / g units:

    71.2 kcal/(day * kg^0.75)
      = 71.2 * 4184 J / 86400 s / kg^0.75
      = 3.448 W/kg^0.75
      = 3.448 / 1000^0.75 W/g^0.75
      = 0.01938 W/g^0.75,

so on the log10 scale

    log10(B0) ≈ -1.713  (W/g^0.75).

Kleiber 1932 framed B0 as a per-species-group quantity (Table 1) and
did not tabulate one universal interspecific B0; the human-derived
value is widely cited as the canonical "Kleiber number" because
mammalian B0 values across Kleiber's 13 species groups cluster within
~20% of it.  We freeze both constants as LAW_CONSTANTS — they are
paper-published numerical values, not refit on the v2 train data.

This is a TYPE I baseline: LOCAL_FITTABLE is empty, no fit() function,
predict() is invoked once on the full test set with LAW_CONSTANTS only.
The benchmark question is whether these frozen Kleiber 1932 constants
extrapolate from the train mass band (39 g - 3.9 kg) out to extreme
mammals (3.7 g pygmy mouse - 3.7 t elephant) in the test set.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
- B_EXPONENT = 0.75   (Kleiber 1932 paginated p. 320 — recommended 3/4)
- LOG_B0     = -1.713 (Kleiber 1932 paginated p. 345 — human formula,
                       expressed as log10(W/g^0.75))

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["body_mass_g"]
PAPER_REF = "summary_formula_kleiber_1932.md"
EQUATION_LOC = (
    "Kleiber 1932 paginated p. 320 (recommended b = 3/4) and "
    "p. 345 (human B0 = 71.2 kcal/(day·kg^0.75) → 0.01938 W/g^0.75)."
)

LAW_CONSTANTS = {
    "B_EXPONENT": 0.75,
    "LOG_B0":    -1.713,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray,
            B_EXPONENT: float = 0.75,
            LOG_B0: float = -1.713) -> np.ndarray:
    """log10(B) = LOG_B0 + B_EXPONENT * log10(M)."""
    M = np.asarray(X[:, 0], dtype=float)
    return LOG_B0 + B_EXPONENT * np.log10(M)
