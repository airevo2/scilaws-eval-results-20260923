"""Hofstadter (1956) historical dipole null model for G_E^p / G_D.

Citation: Hofstadter, Rev. Mod. Phys. 28, 214 (1956). Formula reference:
Table I, row IV (exponential charge density), PDF p. 5.

Formula
-------
Hofstadter's Table I (PDF p. 5) identifies the exponential charge density
rho(r) ~ exp(-r/a) as the best-fit proton model. The corresponding
Born-approximation form factor is:

    F(qa) = (1 + (qa)^2 / 12)^(-2)

Reading this in modern notation with Lambda^2 = 12*(hbar*c)^2/a^2,
the proton electric Sachs form factor takes the dipole form:

    G_E^p(Q^2) = (1 + Q^2 / Lambda^2)^(-2)

The standard dipole reference G_D adopted by the field uses the same
functional form with Lambda^2 = 0.71 GeV^2 (Arrington 2007 p. 7;
Bradford 2006 Eq. 2, p. 2; Ye 2018 p. 6). Therefore:

    GE_over_GD = G_E^p / G_D = (1 + Q^2/Lambda^2)^(-2) / (1 + Q^2/0.71)^(-2)

When Lambda^2 = 0.71 GeV^2, this ratio is identically 1 for all Q^2.

Documented degeneracy: the Hofstadter 1956 dipole baseline is by design a
null / span model — it predicts a flat ratio of exactly 1. This is the
field's earliest and simplest approximation, appropriate as the lower rung
of the baseline ladder. It is retained to (a) document the historical
starting point, (b) span the space of "no Q^2 dependence" predictions for
SR comparison, and (c) test whether a symbolic regression system identifies
the richer structure in the data beyond a flat constant. The degeneracy is
acknowledged here; the harness will naturally rank this baseline below
Arrington 2007 and Bradford 2006 in measured RMSE.

LAW_CONSTANTS
--------------
    {} — empty. The dipole form is the claim itself; the ratio ≡ 1 for
    the canonical Lambda^2. There are no fitted paper constants beyond the
    structural shape of F(qa). The Lambda^2 = 0.71 GeV^2 value is a
    field-wide conventional choice, not a parameter Hofstadter himself
    calibrated in this paper (he used a, not Lambda^2).

OTHER_CONSTANTS
----------------
    Lambda2 = 0.71 GeV^2 — canonical dipole mass scale shared across all
                            baselines. Value stated in: Bradford 2006 Eq. (2)
                            PDF p. 2, Arrington 2007 PDF p. 7, Ye 2018 PDF p. 6.
                            Used both as G_E^p dipole scale and as the G_D
                            normalisation denominator, so the ratio is 1.

Type designation: Type I. No per-experiment or per-cluster refit. The ratio
is identically 1 regardless of Q^2. LOCAL_FITTABLE is empty. No fit() function.

Column mapping: Q2_GeV2 (released CSV col 1) -> Q^2 in GeV^2.
"""

import numpy as np

USED_INPUTS = ["Q2_GeV2"]
PAPER_REF = "summary_formula_dataset_hofstadter_1956.md"
EQUATION_LOC = (
    "Hofstadter 1956 Table I row IV, PDF p. 5 (exponential density dipole form); "
    "canonical Lambda^2=0.71 GeV^2 from Bradford 2006 Eq.(2) PDF p.2 / "
    "Arrington 2007 PDF p.7"
)

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {}     # Dipole ratio ≡ 1 is the structural claim; no fitted value

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "Lambda2": 0.71,   # GeV^2 — canonical dipole scale (Bradford 2006 Eq.2 PDF p.2)
}

LOCAL_FITTABLE = {}    # Type I — no per-cluster parameters


def predict(X: np.ndarray) -> np.ndarray:
    """Predicted GE_over_GD under the Hofstadter 1956 dipole approximation.

    For Lambda^2 = 0.71 GeV^2 (the canonical value used in both G_E and G_D),
    the ratio is identically 1 for all Q^2. This is the null / span model.

    X: (n, 1) — column Q2_GeV2.
    No LAW_CONSTANTS to pass.
    """
    n = X.shape[0]
    return np.ones(n, dtype=float)
