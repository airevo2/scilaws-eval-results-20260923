"""Manara et al. (2017) — log L_acc vs. log L_star, Eq. (3), single power-law.

Manara, C. F. et al. (2017), A&A 604, A127.
DOI: 10.1051/0004-6361/201630147
Equation (3), PDF page 10 (=== PAGE 10 === in .txt, line 1972).

Formula (single power-law, linmix Bayesian linear regression fit to the
Chamaeleon I disk-bearing sample):

    log(L_acc / L_sun) = intercept + slope * log(L_star / L_sun)

Fit parameters: R² = 0.69, BIC = 219, AIC = 214.
1σ dispersion = 0.67 ± 0.08 dex around the best fit.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
From Eq. (3), PDF page 10 (Manara et al. 2017 A&A 604, A127):

    slope     = 1.9    (±0.1)
    intercept = -0.8   (±0.2)

Derived from a nearly complete Chamaeleon I disk-bearing sample
(~100 stars, age ~2–3 Myr, d=160 pc) using the linmix Bayesian
linear regression tool.

OTHER_CONSTANTS — none needed (log-log linear formula, dimensionally clean)

Type designation: Type I — universal constants fit once to the full
Chamaeleon I sample; no per-cluster free parameters. LOCAL_FITTABLE = {}.

Column mapping:
    Paper: log(L_star/L_sun) → released CSV column: logL_star
    Paper: log(L_acc/L_sun)  → released CSV column: logL_acc (target)

Validity domain: Chamaeleon I pre-main-sequence disk-bearing stars;
L_star ~ 0.001–10 L_sun (M_star ~ 0.03–2 M_sun), age ~2–3 Myr.

Caveat: This baseline is calibrated on Chamaeleon I, not Lupus. The
slope (1.9) is slightly steeper than the Alcalá 2017 Lupus fit (1.26 or
1.31). The benchmark data are Lupus YSOs, so this baseline probes
cross-region generalisation of the L_acc–L_star relationship.
"""

import numpy as np

USED_INPUTS = ["logL_star"]
PAPER_REF = "summary_formula+dataset_manara_2017.md"
EQUATION_LOC = "Eq. 3, PDF p. 10 (manara_2017.pdf)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "slope":     1.9,    # PDF p. 10, Eq. (3): (1.9 ± 0.1)
    "intercept": -0.8,   # PDF p. 10, Eq. (3): (−0.8 ± 0.2)
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE  = {}    # Type I


def predict(X: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Predict log(L_acc/L_sun) from log(L_star/L_sun) via Manara+2017 Eq. (3).

    X: (n, 1) — column logL_star [log10 L_sun].
    Returns: (n,) array of predicted logL_acc [log10 L_sun].
    """
    logL_star = np.asarray(X[:, 0], dtype=float)
    return slope * logL_star + intercept
