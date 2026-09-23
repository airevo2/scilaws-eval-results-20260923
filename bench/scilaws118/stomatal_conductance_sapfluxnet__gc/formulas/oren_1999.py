"""Oren et al. (1999) log-linear stomatal conductance model.

Oren, R., Sperry, J. S., Katul, G. G., Pataki, D. E., Ewers, B. E.,
Phillips, N., & Schäfer, K. V. R. (1999). Survey and synthesis of
intra- and interspecific variation in stomatal sensitivity to vapour
pressure deficit. Plant, Cell and Environment, 22(12), 1515–1526.
DOI: 10.1046/j.1365-3040.1999.00513.x

Log-linear stomatal conductance formula (Eq. 1, PDF p. 2):

    Gc = Gc_ref − m · ln(D)

anchored at D = 1 kPa so that ln(1) = 0 and Gc_ref = Gc(D=1 kPa).

Interspecific sensitivity rule (Figs. 3 & 4b, PDF pp. 3 & 7):

    m ≈ 0.6 · Gc_ref  (mesic species; 0.59–0.60 empirical slope)

This 0.6 interspecific slope is a real Oren (1999) finding, but it is a
property of the *cross-species* m-vs-Gc_ref scatter, NOT a coefficient
the shipped two-parameter form consumes: the model fitted here is the
unconstrained log-linear law that re-fits Gc_ref AND m independently per
cluster, so 0.6 enters neither fit() nor predict(). The collapsed
one-parameter variant m = 0.6·Gc_ref is a DIFFERENT (not-shipped) form.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The shipped form is the unconstrained two-parameter law
Gc = Gc_ref − m·ln(D); after the per-cluster fit() params (Gc_ref, m)
are removed, no fixed cross-cluster *fitted* constant remains — the FORM
itself is the whole claim (MANUAL §1.1: invariant-AND-fitted → LAW;
invariant-but-not-consumed-here → not a LAW). LAW_CONSTANTS = {} is
therefore correct (form-discovery Type II, like co2_adsorption / ecoli /
cobb_douglas). The 0.6 interspecific slope is documented above as the
paper's central finding but is not consumed by this two-parameter form.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
None. The 1 in ln(1) = 0 is a structural consequence of the anchor
choice (D = 1 kPa), not an independent numeric constant.

LOCAL_FITTABLE — per-cluster, computed by fit() via OLS on ln(D)
-----------------------------------------------------------------
- Gc_ref : conductance at D = 1 kPa; equals regression intercept b;
           units [cm³ cm⁻² h⁻¹ kPa⁻¹].
- m      : stomatal sensitivity = −dGc/d(ln D); equals |slope| of
           linear regression of Gc on ln(D); positive for physiological
           stomatal closure; units [cm³ cm⁻² h⁻¹ kPa⁻¹ per ln(kPa)].

Type II decision: Gc_ref and m are fit per tree / per species stand from
each cluster's (D, Gc_proxy) time series. Rows across trees are not
interchangeable. Cluster = individual tree (tree_code).

Column mapping:
  paper D → released CSV vpd  [kPa]
  paper Gc / GS → released CSV Gc_proxy [cm³ cm⁻² h⁻¹ kPa⁻¹]

Caveats:
  - m may be negative for drought-stressed or xeric trees; the
    evaluation layer flags such clusters.
  - Gc_proxy = mean(Js) / mean(D) uses daily daytime means; it is
    proportional to (but not identical to) stomatal conductance in SI
    units (see PROVENANCE.md §Conductance derivation).
  - init = None: fit() uses closed-form OLS (exact, no iteration needed).
"""

import numpy as np
from scipy.linalg import lstsq

USED_INPUTS = ["vpd"]
PAPER_REF = "summary_formula_oren_1999.md"
EQUATION_LOC = (
    "Oren et al. 1999 Eq. 1, PDF p. 2: Gc = Gc_ref - m * ln(D); "
    "interspecific rule m ≈ 0.6 * Gc_ref: leaf-level Fig. 3 (r²=0.92, "
    "n=23, PDF p. 3) and sap-flux Fig. 4b (r²=0.84, n=31, PDF p. 7)."
)

# No LAW_CONSTANTS: the shipped two-parameter form Gc = Gc_ref − m·ln(D)
# re-fits both Gc_ref and m per cluster, so it holds no fixed cross-cluster
# fitted coefficient. Oren's 0.6 interspecific slope (Fig. 3 leaf 0.60 r²=0.92
# n=23; Fig. 4b sap-flux 0.59 r²=0.84 n=31; theoretical Eq. 4 ≈0.59) describes
# the cross-species m-vs-Gc_ref scatter and is NOT consumed by this form.
LAW_CONSTANTS = {}

OTHER_CONSTANTS = {}

LOCAL_FITTABLE = {
    "Gc_ref": {"init": None},
    "m":      {"init": None},
}


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Closed-form OLS fit of Gc = Gc_ref - m * ln(D).

    The model is linear in (Gc_ref, m) via z = ln(D):
        Gc = Gc_ref * 1 + m * (−z)
    OLS on design matrix [1, −ln(D)] gives the exact MLE.

    X_fit : (n, 1) — column [vpd] in kPa.
    y_fit : (n,)   — Gc_proxy in cm³ cm⁻² h⁻¹ kPa⁻¹.

    Returns {"Gc_ref": float, "m": float}.
    LAW_CONSTANTS is empty, so the harness calls fit(X_fit, y_fit) with no
    law kwargs; both returned params are per-cluster LOCAL_FITTABLE.
    """
    D = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    valid = (D > 1e-9) & np.isfinite(D) & np.isfinite(y)
    if valid.sum() < 2:
        return {"Gc_ref": float(np.nanmedian(y)) if y.size else 1.0, "m": 0.0}

    D_v = D[valid]
    y_v = y[valid]

    lnD = np.log(D_v)
    A_mat = np.column_stack([np.ones_like(lnD), -lnD])   # (n, 2)

    coef, _, _, _ = lstsq(A_mat, y_v)
    return {"Gc_ref": float(coef[0]), "m": float(coef[1])}


def predict(X: np.ndarray, Gc_ref: float, m: float) -> np.ndarray:
    """Two-parameter log-linear stomatal conductance.

    Gc = Gc_ref - m * ln(D)

    X: (n, 1) — column [vpd] in kPa.
    Gc_ref, m: per-cluster fitted LOCAL parameters.

    Returns: (n,) Gc_proxy in the same units as the training target.
    """
    D = np.asarray(X[:, 0], dtype=float)
    return Gc_ref - m * np.log(np.clip(D, 1e-9, None))
