"""Slichter (1898) porosity-power-law permeability model — log10(k / mD).

Type I — pooled Norwegian shelf core-plug calibration.

Verbatim from Cabalar and Akbulut (2016) Table 4 (PDF p.10):

    k = 1e-2 * (g/nu) * phi^3.287 * d10^2

where phi is porosity, g/nu is gravity/kinematic viscosity (~9.77 s^-1 cm^-1
at 20 deg C), and d10 is the 10th-percentile grain diameter (mm).
Validity limits: 0.01 < d10 < 5.0 mm.

For the Norwegian shelf benchmark, grain size d10 is not available. The g/nu
and d10^2 factors plus the pre-factor 1e-2 and unit conversion are absorbed
into a single global log-space constant C_GLOBAL, pre-fitted once on the
released train split. The Slichter exponent 3.287 is paper-published and
used directly in predict().

In log10 space:

    log10(k / mD) = 3.287 * log10(phi) + C_GLOBAL

=== Type I demotion lineage ===
Originally Type II with `LOCAL_FITTABLE = {"C_log": {"init": None}}` and a
per-cluster C_log refit per lithology. E2 audit (2026-05-28) fired FM-T7 due
to K_train=4 lithologies (< K>=9 floor). Per HANDOFF §6b: demoted to Type I —
pre-fit C_log globally on data/train.csv, freeze as LAW_CONSTANT, remove fit(),
set LOCAL_FITTABLE = {}. C_GLOBAL was pre-fitted by closed-form linear LS:

    C_GLOBAL = mean(log_k_train - 3.287 * log10(phi_train))
             = 3.634470

Pre-fit metric on the 22,246-row test split:
    RMSE = 1.1686,   R^2 = 0.4893.

Type: Type I.
- LOCAL_FITTABLE is empty.
- predict() runs with only LAW_CONSTANTS; no per-cluster fitting.

Column mapping:
  phi  -> CSV column "phi" (dimensionless, fraction).
  log_k -> CSV column "log_k" (log10 of permeability in mD).

LAW_CONSTANTS — paper-published or dataset-frozen, all used in predict()
---------------------------------------------------------------------------
- N_EXPONENT = 3.287  : Slichter porosity exponent, paper-published verbatim
  from Cabalar-Akbulut 2016 Table 4 (PDF p.10). Used in predict().
- C_GLOBAL = 3.634470 : dataset-effective log-space constant absorbing the
  Slichter pre-factor 1e-2, g/nu, mean d10^2, and the cm/s -> mD unit
  conversion. Pre-fitted once on data/train.csv by closed-form linear LS in
  log space. Used in predict().

OTHER_CONSTANTS — paper-published companion constants (docstring provenance)
----------------------------------------------------------------------------
- SLICHTER_PRE_FACTOR = 1e-2 : Slichter pre-factor verbatim from Cabalar-
  Akbulut 2016 Table 4 (PDF p.10). NOT used numerically in predict() because
  d10 is unavailable in this dataset; absorbed into C_GLOBAL. Kept here for
  paper provenance per FM-C10a.

LOCAL_FITTABLE — empty (Type I)
"""

import numpy as np

USED_INPUTS = ["phi"]
PAPER_REF = "summary_formula_dataset_cabalar_2016.md"
EQUATION_LOC = (
    "Cabalar and Akbulut (2016) Table 4, PDF p.10 — "
    "k = 1e-2 * (g/v) * phi^3.287 * d10^2  "
    "(grain-size factor absorbed into C_GLOBAL for d10-unavailable datasets)."
)

LAW_CONSTANTS = {
    "N_EXPONENT": 3.287,       # Slichter porosity exponent; Cabalar-Akbulut 2016 Table 4 PDF p.10
    # Pre-fit global log-space pre-factor for the Norwegian shelf train split.
    # Absorbs 1e-2, g/nu, mean(d10^2), and cm/s -> mD unit conversion.
    "C_GLOBAL":   3.634470,
}
OTHER_CONSTANTS = {
    # Slichter pre-factor verbatim from Cabalar-Akbulut 2016 Table 4 PDF p.10.
    # Not numerically used in predict() because d10 is unavailable; absorbed
    # into C_GLOBAL. Kept here for paper provenance per FM-C10a.
    "SLICHTER_PRE_FACTOR": 1e-2,
}
LOCAL_FITTABLE = {}            # Type I — no per-cluster parameters, no fit()

_EPS = 1e-6


def predict(
    X: np.ndarray,
    N_EXPONENT: float = 3.287,
    C_GLOBAL: float   = 3.634470,
) -> np.ndarray:
    """log10(k/mD) = N_EXPONENT * log10(phi) + C_GLOBAL.

    X: (n, 1) — column [phi].
    Both LAW_CONSTANTS are paper-frozen / dataset-frozen.
    """
    phi = np.clip(np.asarray(X[:, 0], dtype=float), _EPS, 1.0 - _EPS)
    return N_EXPONENT * np.log10(phi) + C_GLOBAL
