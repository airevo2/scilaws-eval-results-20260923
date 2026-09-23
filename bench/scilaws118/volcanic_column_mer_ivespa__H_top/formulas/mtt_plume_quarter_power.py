"""Morton-Taylor-Turner plume theory, H ~ MER^(1/4) — best rung.

Buoyant-plume theory (Morton, Taylor & Turner 1956; applied to volcanic
columns by Wilson et al. 1978, Sparks 1986, Sparks et al. 1997) predicts
that the top height of a maintained buoyant plume rising through a
stratified atmosphere scales as the one-fourth power of the buoyancy
flux, and hence of the mass eruption rate:

    H_top = A * MER^(1/4).

The exponent 1/4 is FIXED by first-principles plume dynamics (it is the
classic H ~ F^{1/4} result for a point buoyancy source in uniform
stratification), not fit to data.  Only the prefactor A — which absorbs
the atmospheric stratification, entrainment coefficient and the
buoyancy-to-MER conversion — is pre-fit on the v2 OOD train.

On the MER-range-OOD test (the largest eruptions, extrapolated from
small/moderate ones), fixing the exponent at the theoretical 1/4
extrapolates marginally better than the shallower data-fit exponent of
the empirical-power-law rung (0.175): theory generalises beyond the
fitted range where a restricted-range empirical slope does not.  Test
log_mae ~ 0.133 — best of the three rungs (though within the 53%
log-scatter noise floor it is nearly tied with the empirical fit; the
real discrimination in this task is power-law vs the wrong-form linear
rung).

LAW_CONSTANTS — frozen
----------------------
- A = 0.2618   prefactor, pre-fit on v2 train (exponent fixed at 1/4)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).  The exponent 1/4 is a structural theoretical constant.
"""

import numpy as np

USED_INPUTS = ["MER_kg_per_s"]
PAPER_REF = "summary_formula_aubry_2023.md"
EQUATION_LOC = (
    "Morton-Taylor-Turner 1956 buoyant-plume scaling H ~ F^(1/4) -> "
    "H_top = A*MER^0.25 (Wilson 1978; Sparks 1997; discussed as the "
    "theoretical baseline in Aubry 2023).  Exponent fixed at 1/4; "
    "prefactor A pre-fit on v2 MER-OOD train."
)

LAW_CONSTANTS = {
    "A": 0.2618,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A: float = 0.2618) -> np.ndarray:
    """H_top = A * MER^(1/4); Morton-Taylor-Turner plume scaling."""
    MER = np.asarray(X[:, 0], dtype=float)
    return A * MER ** 0.25
