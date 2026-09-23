"""Reference baseline: Temesi et al. 2023, Materials 16, 2311.

Citation:
    Temesi, O.; Varga, L.K.; Li, X.; Vitos, L.; Chinh, N.Q.
    "Estimation of Shear Modulus and Hardness of High-Entropy Alloys Made
    from Early Transition Metals Based on Bonding Parameters."
    Materials 2023, 16, 2311. DOI: 10.3390/ma16062311 (CC-BY; PMC10059814).

Formula — Equation 12, PDF page 7 of 10 (Materials 2023, 16, 2311):

    HV_fitted = -122.18 + 109.75 × VEC - 11.23 × ΔHmix

in units of kgf/mm² (VEC dimensionless, ΔHmix in kJ/mol).

LAW_CONSTANTS (all from Eq. 12, PDF p. 7):
    a       = -122.18  kgf/mm²   regression intercept
    b_VEC   =  109.75  kgf/mm²   coefficient on VEC
    c_dHmix =  -11.23  kgf/mm² per kJ/mol   coefficient on dHmix

    Text line 1147 in temesi_2023_hardness_bonding.txt:
        "HVfitted = −122.18 + 109.75 × VEC − 11.23 × ∆Hmix"

OTHER_CONSTANTS: none (the formula is dimensionally consistent by construction).

Type: Type I. All three LAW constants are paper-stated coefficients from a single
globally-calibrated multiple-linear regression on 36 ETM-based RHEA samples (Table 2,
PDF p. 7). No per-cluster fitting is required or declared.

Column mapping:
    VEC   -> "VEC"   (valence electron concentration, dimensionless)
    dHmix -> "dHmix" (Miedema mixing enthalpy, kJ/mol)

Caveats:
    The paper calibrated Eq. 12 specifically on early-transition-metal (ETM)
    BCC RHEAs (VEC range 3.5-6.5). Applying it to the full Gorsse 2018 dataset
    (VEC 4.2-11.7, including FCC 3d-TM HEAs) is an OOD extrapolation.
    The frozen coefficients from the paper are used here as LAW_CONSTANTS;
    fitting them on train.csv would improve RMSE but would no longer be the
    published equation.
"""

import numpy as np

USED_INPUTS = ["VEC", "dHmix"]
PAPER_REF   = "summary_formula_temesi_2023_hardness_bonding.md"
EQUATION_LOC = "Eq. 12, PDF p. 7 (Materials 2023, 16, 2311)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a":       -122.18,   # regression intercept, kgf/mm² — Eq. 12, PDF p. 7
    "b_VEC":    109.75,   # coefficient on VEC, kgf/mm²   — Eq. 12, PDF p. 7
    "c_dHmix":  -11.23,   # coefficient on dHmix, kgf/mm² per kJ/mol — Eq. 12, PDF p. 7
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}   # none needed; formula is dimensionally self-contained

LOCAL_FITTABLE = {}    # Type I: globally calibrated, no per-cluster refit


def predict(X: np.ndarray, a: float, b_VEC: float,
            c_dHmix: float) -> np.ndarray:
    """Predict Vickers hardness from VEC and dHmix.

    X : (n, 2) array — columns = [VEC, dHmix] in USED_INPUTS order.
    The three LAW coefficients (a, b_VEC, c_dHmix) arrive as named params
    via predict(X, **LAW_CONSTANTS), gold-style — no default values.
    Returns predicted HV in kgf/mm².
    """
    VEC   = X[:, 0]
    dHmix = X[:, 1]
    return a + b_VEC * VEC + c_dHmix * dHmix
