"""Terzaghi (1925) permeability formula — log10(k / mD).

Type I — pooled Norwegian shelf core-plug calibration.

Verbatim from Cabalar and Akbulut (2016) Table 4 (PDF p.10):

    k = 0.0084 * (g/nu) * [(phi - 0.13) / (1-phi)^(2/3)] * d10^2

where phi is porosity, g/nu is gravity/kinematic viscosity, and d10 is the
10th-percentile grain diameter (mm). No validity limits stated in Table 4,
but the (phi - 0.13) numerator becomes <= 0 for phi <= 0.13, where the model
is structurally undefined in log space.

For the Norwegian shelf benchmark, grain size d10 is not available. The g/nu
and d10^2 factors plus the pre-factor 0.0084 and unit conversion are absorbed
into a single global log-space constant C_GLOBAL, pre-fitted on the train
rows with phi > 0.13.

In log10 space:

    log10(k / mD) = log10(phi - 0.13) - (2/3) * log10(1 - phi) + C_GLOBAL

The porosity threshold 0.13 and the denominator exponent 2/3 are paper-
published constants. predict() clips phi to a small floor above 0.13 to keep
log() finite for out-of-domain rows; this produces large negative predicted
log_k for phi << 0.13 (the formula is honestly inapplicable there).

=== Type I demotion lineage ===
Originally Type II with `LOCAL_FITTABLE = {"C_log": {"init": None}}` and a
per-cluster C_log refit per lithology. E2 audit (2026-05-28) fired FM-T7 due
to K_train=4 lithologies (< K>=9 floor). Per HANDOFF §6b: demoted to Type I —
pre-fit C_log globally on data/train.csv (restricted to phi > 0.13 rows),
freeze as LAW_CONSTANT, remove fit(), set LOCAL_FITTABLE = {}. C_GLOBAL was
pre-fitted on 38,684 of 51,908 train rows (those with phi > 0.13) by
closed-form linear LS:

    C_GLOBAL = mean(log_k - log10(phi - 0.13) + (2/3) * log10(1 - phi))
             = 2.761932

Pre-fit metric on the 22,246-row test split (full pooled):
    RMSE = 1.6646,   R^2 = -0.0363.

The poor pooled test R^2 is honest: Terzaghi's (phi - 0.13) numerator goes
negative for ~26% of test rows (phi <= 0.13), so the model is physically
inapplicable on the full pooled dataset. Kept in the bank as a documented
formal baseline to allow ladder comparison with the other two phi-only
formulas (Kozeny-Carman, Slichter), both of which produce smooth log10()
behaviour across the full phi range.

Type: Type I.
- LOCAL_FITTABLE is empty.
- predict() runs with only LAW_CONSTANTS; no per-cluster fitting.

Column mapping:
  phi  -> CSV column "phi" (dimensionless, fraction).
  log_k -> CSV column "log_k" (log10 of permeability in mD).

LAW_CONSTANTS — paper-published or dataset-frozen, all used in predict()
---------------------------------------------------------------------------
- N_THRESHOLD = 0.13   : porosity correction baseline, paper-published verbatim
  from Cabalar-Akbulut 2016 Table 4 (PDF p.10). Enters numerator as
  (phi - 0.13); used in predict().
- C_GLOBAL    = 2.761932 : dataset-effective log-space constant absorbing the
  Terzaghi pre-factor 0.0084, g/nu, mean d10^2, and the cm/s -> mD unit
  conversion. Pre-fitted once on the 38,684 train rows with phi > 0.13 by
  closed-form linear LS in log space. Used in predict().

OTHER_CONSTANTS — paper-published companion constants (docstring provenance)
----------------------------------------------------------------------------
- TERZAGHI_PRE_FACTOR = 0.0084 : Terzaghi pre-factor verbatim from Cabalar-
  Akbulut 2016 Table 4 (PDF p.10). NOT used numerically in predict() because
  d10 is unavailable; absorbed into C_GLOBAL. Kept here for paper provenance
  per FM-C10a.

LOCAL_FITTABLE — empty (Type I)
"""

import numpy as np

USED_INPUTS = ["phi"]
PAPER_REF = "summary_formula_dataset_cabalar_2016.md"
EQUATION_LOC = (
    "Cabalar and Akbulut (2016) Table 4, PDF p.10 — "
    "k = 0.0084 * (g/v) * [(phi - 0.13) / (1-phi)^(2/3)] * d10^2  "
    "(grain-size factor absorbed into C_GLOBAL for d10-unavailable datasets)."
)

LAW_CONSTANTS = {
    "N_THRESHOLD": 0.13,       # porosity correction baseline; Cabalar-Akbulut 2016 Table 4 PDF p.10
    # Pre-fit global log-space pre-factor for the Norwegian shelf train split
    # (restricted to phi > 0.13 rows where the formula is defined).
    "C_GLOBAL":    2.761932,
}
OTHER_CONSTANTS = {
    # Terzaghi pre-factor verbatim from Cabalar-Akbulut 2016 Table 4 PDF p.10.
    # Not numerically used in predict() because d10 is unavailable; absorbed
    # into C_GLOBAL. Kept here for paper provenance per FM-C10a.
    "TERZAGHI_PRE_FACTOR": 0.0084,
}
LOCAL_FITTABLE = {}            # Type I — no per-cluster parameters, no fit()

_EPS = 1e-6


def predict(
    X: np.ndarray,
    N_THRESHOLD: float = 0.13,
    C_GLOBAL: float    = 2.761932,
) -> np.ndarray:
    """log10(k/mD) = log10(phi - N_THRESHOLD) - (2/3)*log10(1-phi) + C_GLOBAL.

    X: (n, 1) — column [phi].
    Values with phi <= N_THRESHOLD are clipped to (phi - N_THRESHOLD = _EPS)
    before taking the log; predictions there are physically meaningless but
    numerically finite (large-negative log_k) so the harness can compute a
    pooled R^2 honestly.
    """
    phi = np.asarray(X[:, 0], dtype=float)
    num = np.clip(phi - N_THRESHOLD, _EPS, None)
    den = np.clip(1.0 - phi, _EPS, None)
    return np.log10(num) - (2.0 / 3.0) * np.log10(den) + C_GLOBAL
