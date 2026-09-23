"""Giordano, Russell & Dingwell (2008) multicomponent silicate-melt viscosity model.

Giordano D., Russell J.K., Dingwell D.B. (2008).
"Viscosity of magmatic liquids: A model."
Earth and Planetary Science Letters 271:123-134.
DOI: 10.1016/j.epsl.2008.03.038.
PDF: reference/giordano_2008.pdf (from UBC EOAS server, CC-accessible)

Model (Eq. 1, p. 126 of the PDF):

    log10(eta) = A + B / (T(K) - C)

where B and C are linear combinations of compositional terms (mol% oxides):

    B = sum(bi * Mi) + sum(b1j * M1j * M2j)          (Eq. 2, p. 126)
    C = sum(ci * Ni) + c11 * (N1_11 - N2_11)          (Eq. 3, p. 126)

All 18 LAW_CONSTANTS from Table 1, p. 126:

    A = -4.55 (universal Arrhenian limit, +/-0.21, Table 1 caption)

B-coefficients (Table 1, p. 126):
    b1  = 159.6   (SiO2 + TiO2)
    b2  = -173.3  (Al2O3)
    b3  = 72.1    (FeO(T) + MnO + P2O5)
    b4  = 75.7    (MgO)
    b5  = -39.0   (CaO)
    b6  = -84.1   (Na2O + V)               [V = H2O + F2O-1]
    b7  = 141.5   (V + ln(1 + H2O))        [note: H2O here is mol%]
    b11 = -2.43   (SiO2+TiO2) * FM         [FM = FeO(T)+MnO+MgO]
    b12 = -0.91   (SiO2+TA+P2O5) * (NK+H2O)  [TA = TiO2+Al2O3; NK = Na2O+K2O]
    b13 = 17.6    Al2O3 * NK

C-coefficients (Table 1, p. 126):
    c1  = 2.75    (SiO2)
    c2  = 15.7    (TA = TiO2 + Al2O3)
    c3  = 8.3     (FM = FeO(T) + MnO + MgO)
    c4  = 10.2    (CaO)
    c5  = -12.3   (NK = Na2O + K2O)
    c6  = -99.5   ln(1 + V)
    c11 = 0.30    (Al2O3 + FM + CaO - P2O5) * (NK + V)

Note: The paper uses mole-fraction oxides expressed as mol% (0-100).
The dataset oxide columns (sio2, tio2, al2o3, feo, fe2o3, mno, mgo, cao,
na2o, k2o, p2o5, h2o) are already in mol% — no conversion needed.

FeO(T): The dataset provides feo and fe2o3 separately. Giordano 2008
treats all iron as FeO equivalent: FeO(T) = feo + 2*fe2o3 (converting
Fe2O3 to FeO molar basis: 1 mol Fe2O3 -> 2 mol FeO; p. 125 §3 of
Giordano 2008 states "We use the oxide mol% as a chemical basis and
treat all iron as FeO").
Since fe2o3 is small in most natural melts, the approximation FeO(T) = feo
is also defensible; we use the full conversion feo + 2*fe2o3 for accuracy.

F2O-1 (fluorine): not present in the dataset columns. We set F=0 so
V = H2O (volatile total = water only). This is the most common case in
the database; fluorine-bearing measurements are a small subset.

Type I baseline (global law): no LOCAL_FITTABLE parameters.
This provides a global multicomponent prediction using T and composition.
"""

from __future__ import annotations

import numpy as np

# Inputs consumed by predict(); all must be columns in test_fit / test_test
# group_id is not a predictive input; log_viscosity is the target.
USED_INPUTS = [
    "T",
    "sio2", "tio2", "al2o3", "feo", "fe2o3", "mno", "mgo", "cao",
    "na2o", "k2o", "p2o5", "h2o",
]
PAPER_REF    = "summary_formula_giordano_2008.md"
EQUATION_LOC = "Eqs. 1-3 + Table 1, PDF p. 126 (Giordano, Russell & Dingwell 2008, EPSL 271:123-134)"

# All values are from Table 1, p. 126 of Giordano et al. (2008)
# EPSL 271:123-134. Numbers in parentheses in Table 1 are 95% CI on last digit.
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    # Universal pre-exponential (Arrhenian limit), Table 1 caption, p. 126
    "A":   -4.55,
    # B-polynomial coefficients (K), Table 1 left column, p. 126
    "b1":  159.6,    # SiO2 + TiO2
    "b2":  -173.3,   # Al2O3
    "b3":  72.1,     # FeO(T) + MnO + P2O5
    "b4":  75.7,     # MgO
    "b5":  -39.0,    # CaO
    "b6":  -84.1,    # Na2O + V  (V = H2O + F2O-1)
    "b7":  141.5,    # V + ln(1 + H2O)
    "b11": -2.43,    # (SiO2 + TiO2) * FM  [FM = FeO(T)+MnO+MgO]
    "b12": -0.91,    # (SiO2 + TA + P2O5) * (NK + H2O)  [TA = TiO2+Al2O3; NK = Na2O+K2O]
    "b13": 17.6,     # Al2O3 * NK
    # C-polynomial coefficients (K), Table 1 right column, p. 126
    "c1":  2.75,     # SiO2
    "c2":  15.7,     # TA = TiO2 + Al2O3
    "c3":  8.3,      # FM = FeO(T) + MnO + MgO
    "c4":  10.2,     # CaO
    "c5":  -12.3,    # NK = Na2O + K2O
    "c6":  -99.5,    # ln(1 + V)
    "c11": 0.30,     # (Al2O3 + FM + CaO - P2O5) * (NK + V)
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}   # No unit-conversion or structural constants needed
LOCAL_FITTABLE  = {}   # Global law: no per-cluster fitting (Type I; v2 dict-schema)


def predict(X: np.ndarray, **law_constants) -> np.ndarray:
    """Predict log10(eta / Pa.s) from the Giordano 2008 model.

    Parameters
    ----------
    X : ndarray, shape (n, len(USED_INPUTS))
        Columns in order: T, sio2, tio2, al2o3, feo, fe2o3, mno, mgo, cao,
        na2o, k2o, p2o5, h2o.
        T in K; oxide columns in mol% (0-100 scale).
    law_constants : mapping
        Must contain every key in LAW_CONSTANTS (passed by harness).

    Returns
    -------
    ndarray shape (n,)  — log10(eta / Pa.s)

    Notes
    -----
    Formula (Eqs. 1-3, Table 1, Giordano et al. 2008, p. 126):
        log10(eta) = A + B / (T - C)
    where B and C are compositional sums using mol% oxides.
    """
    # Unpack law constants
    A   = float(law_constants["A"])
    b1  = float(law_constants["b1"])
    b2  = float(law_constants["b2"])
    b3  = float(law_constants["b3"])
    b4  = float(law_constants["b4"])
    b5  = float(law_constants["b5"])
    b6  = float(law_constants["b6"])
    b7  = float(law_constants["b7"])
    b11 = float(law_constants["b11"])
    b12 = float(law_constants["b12"])
    b13 = float(law_constants["b13"])
    c1  = float(law_constants["c1"])
    c2  = float(law_constants["c2"])
    c3  = float(law_constants["c3"])
    c4  = float(law_constants["c4"])
    c5  = float(law_constants["c5"])
    c6  = float(law_constants["c6"])
    c11 = float(law_constants["c11"])

    X = np.asarray(X, dtype=float)

    # --- Column extraction (matching USED_INPUTS order) ---
    T    = X[:, 0]   # K
    sio2 = X[:, 1]   # mol%
    tio2 = X[:, 2]
    al2o3 = X[:, 3]
    feo   = X[:, 4]
    fe2o3 = X[:, 5]
    mno   = X[:, 6]
    mgo   = X[:, 7]
    cao   = X[:, 8]
    na2o  = X[:, 9]
    k2o   = X[:, 10]
    p2o5  = X[:, 11]
    h2o   = X[:, 12]

    # --- Derived compositional groups (mol% basis) ---
    # FeO(T): total iron as FeO; 1 mol Fe2O3 = 2 mol FeO (molar basis)
    # Giordano 2008 p. 125 §3: "We use the oxide mol% as a chemical basis
    # and treat all iron as FeO" — so Fe2O3 mol% is converted to FeO molar.
    feot = feo + 2.0 * fe2o3

    # V = H2O + F2O-1 (volatile total); F=0 in this dataset
    V = h2o  # + 0 (no fluorine column)

    # TA = TiO2 + Al2O3
    TA = tio2 + al2o3

    # FM = FeO(T) + MnO + MgO
    FM = feot + mno + mgo

    # NK = Na2O + K2O
    NK = na2o + k2o

    # --- B parameter (Eq. 2, p. 126) ---
    # B = sum(bi * Mi) + sum(b1j * M1j * M2j)
    B = (
        b1  * (sio2 + tio2)           # b1: SiO2 + TiO2
        + b2  * al2o3                  # b2: Al2O3
        + b3  * (feot + mno + p2o5)   # b3: FeO(T) + MnO + P2O5
        + b4  * mgo                    # b4: MgO
        + b5  * cao                    # b5: CaO
        + b6  * (na2o + V)             # b6: Na2O + V
        + b7  * (V + np.log(1.0 + h2o))  # b7: V + ln(1 + H2O)
        + b11 * (sio2 + tio2) * FM    # b11: (SiO2+TiO2) * FM
        + b12 * (sio2 + TA + p2o5) * (NK + h2o)  # b12: (SiO2+TA+P2O5)*(NK+H2O)
        + b13 * al2o3 * NK             # b13: Al2O3 * NK
    )

    # --- C parameter (Eq. 3, p. 126) ---
    # C = sum(ci * Ni) + c11 * (Al2O3 + FM + CaO - P2O5) * (NK + V)
    C = (
        c1  * sio2                     # c1: SiO2
        + c2  * TA                     # c2: TA = TiO2 + Al2O3
        + c3  * FM                     # c3: FM = FeO(T)+MnO+MgO
        + c4  * cao                    # c4: CaO
        + c5  * NK                     # c5: NK = Na2O + K2O
        + c6  * np.log(1.0 + V)        # c6: ln(1 + V)
        + c11 * (al2o3 + FM + cao - p2o5) * (NK + V)  # c11: cross-term
    )

    # --- VFT prediction ---
    # The Giordano 2008 model has two physically-defined bounds documented in
    # the paper itself, which we use to clip extrapolation outside the model's
    # domain of validity (compositions far from its silicate-melt calibration
    # set, e.g. SiO2=12/Al2O3=44/CaO=44 ca-aluminate, or pure Al2O3):
    #
    #   Upper bound: log10(eta) = 12 at the glass-transition temperature Tg
    #   (Giordano 2008 p. 130, Eq. discussion: "Tg ... taken, here, as the
    #   temperature corresponding to a viscosity of 10^12 Pa s"). Above this
    #   the material is glass, not melt — the VFT form is undefined.
    #
    #   Lower bound: log10(eta) = A = -4.55, the high-T Arrhenian limit
    #   (Giordano 2008 p. 127, Table 1 caption: "A = -4.55 ... high-T limit
    #   to viscosity"). Below this the model returns its own asymptotic floor.
    #
    # For compositions outside the calibration set, T can approach or fall
    # below the computed C (Tg), where the VFT form B/(T-C) becomes singular.
    # In such cases the model's physical interpretation is that the melt has
    # reached its glass transition — log10(eta) >= 12. We clip to this range.
    denom = T - C
    safe_denom = np.where(np.abs(denom) > 1e-3, denom, np.sign(denom + 1e-10) * 1e-3)
    raw = A + B / safe_denom

    # When T <= C, the VFT model is below its glass-transition floor; the
    # physically correct value is the Tg ceiling (log10 eta = 12). Treat T<=C
    # and any prediction exceeding the Tg ceiling as glass-transitioned.
    # Use the paper's published Tg convention (log10 eta = 12) as the upper
    # bound and A as the lower bound, both directly from Giordano 2008.
    TG_LOG10_ETA = 12.0   # Giordano 2008 p. 130 (glass-transition convention)
    A_FLOOR      = A      # Giordano 2008 p. 127 / Table 1 caption (high-T limit)

    out = np.where(denom <= 0, TG_LOG10_ETA, raw)
    return np.clip(out, A_FLOOR, TG_LOG10_ETA)


# Type I global law: no fit() — Giordano 2008 publishes all 18 LAW_CONSTANTS
# as paper-frozen values (Table 1, p.126). The harness validate_contract()
# explicitly forbids a fit() when LOCAL_FITTABLE is empty; an empty no-op
# fit() would be a contract violation.
