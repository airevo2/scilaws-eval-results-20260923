"""Manara et al. (2017) — log L_acc vs. log L_star, Eq. (2), broken power-law.

Manara, C. F. et al. (2017), A&A 604, A127.
DOI: 10.1051/0004-6361/201630147
Equation (2), PDF page 10 (=== PAGE 10 === in .txt).

Formula (broken power-law / segmented line fit via scipy.optimize.curve_fit
on the Chamaeleon I disk-bearing sample):

    log(L_acc/L_sun) = -0.55 + 2.08 * log(L_star/L_sun)  if log(L_star) <= -0.34
    log(L_acc/L_sun) = -1.04 + 0.63 * log(L_star/L_sun)  if log(L_star) >  -0.34

Break point: log(L_star/L_sun) = -0.34 (L_star ≈ 0.46 L_sun, M_star ≈ 0.3 M_sun).

The segmented model is slightly preferred by AIC (208 vs 212 for single
power-law) and achieves R² = 0.78 vs 0.70, but BIC (217) does not prefer
it over the single power-law (BIC = 217 for both). The paper concludes
both models are plausible.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
From Eq. (2), PDF page 10 (Manara et al. 2017 A&A 604, A127):

    slope_low     = 2.08   — slope below break point (log L_star ≤ -0.34)
    intercept_low = -0.55  — intercept below break point
    slope_high    = 0.63   — slope above break point (log L_star > -0.34)
    intercept_high = -1.04  — intercept above break point
    log_L_break   = -0.34  — break point in log(L_star/L_sun)

These are the best-fit parameters: θ0 = -0.55, θ1 = 2.08, θ2 = -1.45
where θ2 is the change in slope (not the second slope directly).
The second slope = θ1 + θ2 = 2.08 + (-1.45) = 0.63, and
intercept_high satisfies continuity at x_c = -0.34:
  intercept_high = -0.55 + 2.08*(-0.34) - 0.63*(-0.34) = -0.55 - 0.4913 = -1.04
(consistent with Eq. 2 in the PDF).

OTHER_CONSTANTS — none needed

Type designation: Type I — universal broken power-law fit to Chamaeleon I
sample; no per-cluster free parameters. LOCAL_FITTABLE = {}.

Column mapping:
    Paper: log(L_star/L_sun) → released CSV column: logL_star
    Paper: log(L_acc/L_sun)  → released CSV column: logL_acc (target)

Caveat: The break at log L_star = -0.34 corresponds to M_star ≈ 0.3 M_sun
(the boundary between very low-mass and higher-mass T Tauri stars). The
steep slope (2.08) below the break reflects that low-mass stars have
proportionally much lower accretion luminosities. Applied to the Lupus
benchmark dataset (a different star-forming region), this probes
cross-region generalisation of the broken power-law interpretation.
"""

import numpy as np

USED_INPUTS = ["logL_star"]
PAPER_REF = "summary_formula+dataset_manara_2017.md"
EQUATION_LOC = "Eq. 2, PDF p. 10 (manara_2017.pdf)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "slope_low":      2.08,   # PDF p. 10, Eq. (2): θ1 = 2.08
    "intercept_low":  -0.55,  # PDF p. 10, Eq. (2): θ0 = -0.55
    "slope_high":     0.63,   # PDF p. 10, Eq. (2): θ1 + θ2 = 2.08 + (-1.45) = 0.63
    "intercept_high": -1.04,  # PDF p. 10, Eq. (2): computed from continuity at x_c=-0.34
    "log_L_break":    -0.34,  # PDF p. 10, Eq. (2): x_c = -0.34
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE  = {}    # Type I


def predict(X: np.ndarray, slope_low: float, intercept_low: float,
            slope_high: float, intercept_high: float,
            log_L_break: float) -> np.ndarray:
    """Predict log(L_acc/L_sun) via Manara+2017 Eq. (2) broken power-law.

    X: (n, 1) — column logL_star [log10 L_sun].
    Returns: (n,) array of predicted logL_acc [log10 L_sun].
    """
    logL_star = np.asarray(X[:, 0], dtype=float)
    low_mask = logL_star <= log_L_break
    result = np.where(
        low_mask,
        slope_low * logL_star + intercept_low,
        slope_high * logL_star + intercept_high,
    )
    return result
