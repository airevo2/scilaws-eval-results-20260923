"""Kozeny-Carman (1937) phi-only permeability model — log10(k / mD).

Type I — pooled Norwegian shelf core-plug calibration.

Original derivation: Kozeny (1927) Sitzungsber. Akad. Wiss. Wien 136:271-306;
Carman (1937) Trans. Inst. Chem. Eng. 15:150-166.

Canonical form (as tabulated in Cabalar & Akbulut 2016 Table 4, PDF p.10):

    k = 8.3e-3 * (g/nu) * [phi^3 / (1-phi)^2] * d10^2

where phi is porosity (dimensionless), g/nu is gravity/kinematic viscosity
(~9.77 s^-1 cm^-1 at 20 deg C), and d10 is the 10th-percentile grain diameter
(cm). The pre-factor 8.3e-3 is the paper-published Kozeny-Carman calibration.

For the Norwegian shelf benchmark, grain size d10 is not available. The g/nu
and d10^2 factors plus the pre-factor 8.3e-3 and the unit conversion (cm/s ->
mD) are collapsed into a single global log-space constant C_GLOBAL, which is
**pre-fitted once on this dataset's train split** (the demoted Type I form
mirrors Hack 1957 in this repo: a paper-published structural form whose
unit-conversion / dataset-effective constant is computed once and then frozen
as a LAW_CONSTANT).

In log10 space:

    log10(k / mD) = 3 * log10(phi) - 2 * log10(1 - phi) + C_GLOBAL

The structural exponents 3 (numerator) and 2 (denominator) are canonical KC
values.

=== Type I demotion lineage ===
Originally Type II with `LOCAL_FITTABLE = {"C_log": {"init": None}}` and a
per-cluster C_log refit per lithology (6 clusters total: 4 siliciclastic
train + 2 carbonate test). E2 audit (2026-05-28) fired FM-T7 because
K_train=4 lithologies is below the K>=9 demotion floor. Per the FM-T7 recipe
(HANDOFF §6b: "pre-fit LOCAL globally, freeze as LAW, remove fit(), set
LOCAL_FITTABLE = {}"), this baseline was demoted to Type I. C_GLOBAL was
pre-fitted on the 51,908-row train split by closed-form linear LS:

    C_GLOBAL = mean(log_k_train - 3*log10(phi_train) + 2*log10(1-phi_train))
             = 3.220265

Pre-fit metric on the 22,246-row test split:
    RMSE = 1.1512,   R^2 = 0.5044.

Type: Type I.
- LOCAL_FITTABLE is empty.
- predict() runs with only LAW_CONSTANTS; no per-cluster fitting.

Column mapping:
  phi  -> CSV column "phi" (dimensionless, fraction).
  log_k -> CSV column "log_k" (log10 of permeability in mD).

LAW_CONSTANTS — paper-published or dataset-frozen, all used in predict()
---------------------------------------------------------------------------
- C_GLOBAL = 3.220265 : dataset-effective log-space constant absorbing the KC
  pre-factor 8.3e-3, g/nu, mean d10^2, and the cm/s -> mD unit conversion.
  Pre-fitted once on data/train.csv (the released training split) by closed-
  form linear LS in log space. Frozen as the dataset's Norwegian-shelf
  calibrated KC pre-factor (Hack-1957 pattern: paper-published functional form
  + one dataset-effective constant pre-baked into LAW). Used in predict().

OTHER_CONSTANTS — paper-published companion constants (docstring provenance)
----------------------------------------------------------------------------
- KC_PRE_FACTOR = 8.3e-3 : the canonical Kozeny-Carman pre-factor verbatim
  from Cabalar & Akbulut 2016 Table 4, PDF p.10. NOT used numerically in
  predict() because grain size d10 is unavailable in this dataset; absorbed
  into C_GLOBAL above. Kept here for paper provenance per FM-C10a (LAW vs
  OTHER classification: LAW = used in predict(); OTHER = docstring-only
  companion or universal physical constant).

LOCAL_FITTABLE — empty (Type I)
"""

import numpy as np

USED_INPUTS = ["phi"]
PAPER_REF = "summary_formula_dataset_cabalar_2016.md"
EQUATION_LOC = (
    "Cabalar and Akbulut (2016) Table 4, PDF p.10 — "
    "k = 8.3e-3 * (g/v) * [phi^3/(1-phi)^2] * d10^2  "
    "(grain-size factor absorbed into C_GLOBAL for d10-unavailable datasets)."
)

LAW_CONSTANTS = {
    # Pre-fit global log-space pre-factor for the Norwegian shelf train split.
    # Absorbs 8.3e-3, g/nu, mean(d10^2), and cm/s -> mD unit conversion.
    "C_GLOBAL": 3.220265,
}
OTHER_CONSTANTS = {
    # KC pre-factor verbatim from Cabalar-Akbulut 2016 Table 4 PDF p.10.
    # Not numerically used in predict() because d10 is unavailable; absorbed
    # into C_GLOBAL. Kept here for paper provenance per FM-C10a.
    "KC_PRE_FACTOR": 8.3e-3,
}
LOCAL_FITTABLE = {}            # Type I — no per-cluster parameters, no fit()

_EPS = 1e-6


def predict(X: np.ndarray, C_GLOBAL: float = 3.220265) -> np.ndarray:
    """log10(k/mD) = 3*log10(phi) - 2*log10(1-phi) + C_GLOBAL.

    X: (n, 1) — column [phi].
    C_GLOBAL is dataset-frozen LAW_CONSTANT (see module docstring).
    """
    phi = np.clip(np.asarray(X[:, 0], dtype=float), _EPS, 1.0 - _EPS)
    return 3.0 * np.log10(phi) - 2.0 * np.log10(1.0 - phi) + C_GLOBAL
