"""PFAS soil sorption: empirical log KOC — log KOW linear correlation.

Citation: Fabregat-Palau, J.; Ershadi, A.; Finkel, M.; Rigol, A.; Vidal, M.;
Grathwohl, P. Modeling PFAS Sorption in Soils Using Machine Learning.
Environ. Sci. Technol. 2025, 59(15), 7678–7687. DOI: 10.1021/acs.est.4c13284.

Formula (Eq. from Figure 2 caption, PDF p. 6):

    log KOC = 0.58 · log KOW + 0.06    (n=47 PFAS species, r²=0.83)

Combined with the standard partitioning relationship
    log Kd = log KOC + log10(foc)

the closed-form prediction is:

    log_kd = 0.58 · log_Kow + 0.06 + log10(foc)

LAW_CONSTANTS (paper-published, frozen; scored by judge channel):
    slope     = 0.58  — PDF p. 6, Figure 2 caption:
                        "log KOC = 0.58 log KOW + 0.06; n = 47; r2 = 0.83"
    intercept = 0.06  — PDF p. 6, Figure 2 caption: ibid.

OTHER_CONSTANTS (structural; not scored):
    None needed — log10 is a standard math operator, foc already dimensionless.

Type designation: Type I — each row is an independent (PFAS, soil) measurement
pair from a batch equilibrium experiment. The Koc-Kow regression coefficients
are globally fitted across all 47 PFAS species (no per-cluster parameters).
LOCAL_FITTABLE = {} (empty).

Column mapping (paper notation → released CSV columns):
    log KOW → log_Kow   (dimensionless)
    foc      → foc       (dimensionless = Corg/100)
    log KOC  → intermediate; not directly in CSV

Caveats:
- Coefficients 0.58 and 0.06 were regressed on per-PFAS average log KOC vs
  log KOW (PDF p. 6, Fig. 2); four outlier PFAS were excluded from that fit
  (6:2 FtSaAm, C6/6 PFPiA, C6/8 PFPiA, C8/8 PFPiA).
- The formula treats all PFAS uniformly; it ignores speciation and
  subfamily-specific sorption behaviour.
- The paper notes (PDF p. 6) the PFAS slope (0.58) is lower than PAH (0.97)
  and HOC (1.10) regressions, attributed to the charged character of most PFAS
  at environmental pH.
- When foc → 0, log10(foc) → -∞. predict() clips foc at 1e-9.
"""

import numpy as np

USED_INPUTS = ["log_Kow", "foc"]
PAPER_REF   = "summary_dataset_fabregat_2025.md"
EQUATION_LOC = "Fig. 2 caption, PDF p. 6: log KOC = 0.58 log KOW + 0.06"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "slope":     0.58,   # PDF p. 6, Fig. 2 caption: regression slope
    "intercept": 0.06,   # PDF p. 6, Fig. 2 caption: regression intercept
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}    # log10 is a math operator; foc is already dimensionless

LOCAL_FITTABLE = {}     # Type I — no per-cluster parameters; no fit()


def predict(X: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Predict log10(Kd) from log_Kow and foc.

    Parameters
    ----------
    X : np.ndarray, shape (n, 2)
        Column 0 = log_Kow (log10 octanol-water partition coefficient).
        Column 1 = foc (fraction organic carbon, dimensionless).
    slope : float
        Slope in log KOC = slope * log KOW + intercept.
        LAW value: 0.58 (Fabregat-Palau 2025, PDF p. 6, Fig. 2 caption).
    intercept : float
        Intercept in log KOC = slope * log KOW + intercept.
        LAW value: 0.06 (Fabregat-Palau 2025, PDF p. 6, Fig. 2 caption).

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted log10(Kd) in log10(L/kg).
    """
    log_kow = np.asarray(X[:, 0], dtype=float)
    foc     = np.clip(np.asarray(X[:, 1], dtype=float), 1e-9, None)
    return slope * log_kow + intercept + np.log10(foc)
