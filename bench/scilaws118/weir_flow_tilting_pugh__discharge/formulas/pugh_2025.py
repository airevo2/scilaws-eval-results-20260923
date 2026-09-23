"""Pugh (2025) dimensionless weir-discharge equation applied to a tilting weir.

Pugh, J. E., Venayagamoorthy, S. K., Gates, T. K., Berni, C., & Rastello, M.
(2025). Demystifying the discharge coefficient for flow over thin weirs and
sills. *Flow*, 5, E31. DOI: 10.1017/flo.2025.10022.

---

**Vertical-weir formula (paper Eq. 2.3 + Eq. 5.1):**

From Buckingham Pi analysis (Eq. 2.3):

    Fr_h = q / (sqrt(g) * h^(3/2))  =  phi(h/p)

where q = Q/b is discharge per unit channel width [m² s⁻¹].

The empirical linear fit for the weir regime (Eq. 5.1, PDF p. E31-13):

    Fr_h  ≈  0.576 + 0.071 * (h/p)

with LAW_CONSTANTS 0.576 (intercept) and 0.071 (slope) valid for a vertical
sharp-crested weir (theta = 90°), Re_h > 3.5e4, h/p < 5.

---

**Extension to the tilting-weir dataset (Pugh et al. 2024, JHE):**

The companion Pugh 2024 tilting-weir dataset covers inclination angles
theta = 25.7° to 90°.  Pugh 2025 §1 (Impact Statement) notes explicitly that
the tilting weir "operates dynamically between the bounding cases discussed in
this study" (vertical weir and free overfall).  At theta < 90° the effective
crest geometry and nappe structure deviate from the vertical-weir assumptions
behind Eq. 5.1: the intercept and slope of the linear Fr_h(h/p) relation
change with theta.

Consequently, the *functional form* Fr_h = a0 + a1*(h/p) is the scientific
claim from Pugh 2025, while the vertical-weir published constants (0.576 and
0.071) are NOT universal across inclinations.  Both are LOCAL_FITTABLE per
cluster (per experimental configuration, which maps onto distinct theta values
in the tilting-weir dataset).

Total discharge is recovered as:

    Q = Fr_h * sqrt(g) * h^(3/2) * b

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None — the only numerical values in Eq. 5.1 (0.576 and 0.071) are calibrated
for a vertical weir (theta = 90°).  They are cited as LAW_CONSTANTS in the
vertical-weir domain but are NOT valid across the full theta range of the
Pugh 2024 tilting-weir dataset; treating them as fixed would impose a systematic
bias for every non-vertical configuration.  They are therefore re-fit per
cluster.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
g = 9.81 m s⁻² (gravitational acceleration; paper uses 9.81, PDF p. E31-9,
§2 states "g is the gravitational constant (equal in this study to 9.81 m s⁻²)").

LOCAL_FITTABLE — per-cluster, fitted by fit() via OLS
------------------------------------------------------
- a0 : Fr_h intercept (dimensionless).  Vertical-weir anchor: 0.576 (Eq. 5.1).
- a1 : Fr_h slope w.r.t. h/p (dimensionless).  Vertical-weir anchor: 0.071.

init = None: fit() builds a data-derived start from the linear regression
of Fr_h on h/p; the problem is always linear in (a0, a1), so OLS gives the
exact solution with no nonlinear iteration needed.
"""

import numpy as np

USED_INPUTS = ["h_m", "p_m", "b_m"]
PAPER_REF = "summary_formula_pugh_2025.md"
EQUATION_LOC = (
    "Pugh 2025 Eq. 2.3 (Fr_h = q/(sqrt(g)*h^(3/2)) = phi(h/p)), PDF p. E31-9; "
    "Eq. 5.1 (Fr_h ≈ 0.576 + 0.071*h/p), PDF p. E31-13; "
    "g = 9.81 m s⁻², stated in §2 PDF p. E31-9."
)

# g = 9.81 m s⁻²: stated explicitly in Pugh 2025 §2, PDF p. E31-9.
LAW_CONSTANTS = {}
OTHER_CONSTANTS = {"g": 9.81}   # m s⁻²
_G = OTHER_CONSTANTS["g"]        # alias of the boxed gravity constant
LOCAL_FITTABLE = {
    "a0": {"init": None},  # Fr_h intercept; vertical-weir published value: 0.576
    "a1": {"init": None},  # Fr_h slope in h/p; vertical-weir published value: 0.071
}


def _Q(h, p, b, a0, a1):
    """Q = (a0 + a1*(h/p)) * sqrt(g) * h^(3/2) * b."""
    Fr_h = a0 + a1 * (h / p)
    return Fr_h * np.sqrt(_G) * h ** 1.5 * b


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """OLS fit of the linear Fr_h(h/p) model to per-cluster (Q, h, p, b) data.

    The model Q = (a0 + a1*(h/p)) * sqrt(g) * h^(3/2) * b is linear in (a0, a1)
    after dividing both sides by sqrt(g) * h^(3/2) * b, giving:

        Fr_h_obs = a0 + a1 * (h/p)

    where Fr_h_obs = Q / (sqrt(g) * h^(3/2) * b).

    OLS on this system is exact (no nonlinear iteration); least_squares is not
    used.  Fallback: if fewer than 2 rows, return vertical-weir published anchors
    (Pugh 2025 Eq. 5.1).
    """
    h = np.asarray(X_fit[:, 0], dtype=float)
    p = np.asarray(X_fit[:, 1], dtype=float)
    b = np.asarray(X_fit[:, 2], dtype=float)
    Q_obs = np.asarray(y_fit, dtype=float)

    denom = np.sqrt(_G) * h ** 1.5 * b
    # Guard against zero denominators (should not occur for valid data)
    good = (denom > 0) & np.isfinite(denom) & np.isfinite(Q_obs)
    if good.sum() < 2:
        # Vertical-weir anchor from Pugh 2025 Eq. 5.1
        return {"a0": 0.576, "a1": 0.071}

    Fr_h_obs = Q_obs[good] / denom[good]
    r = h[good] / p[good]                  # h/p

    # OLS: [1, r] @ [a0, a1] = Fr_h_obs
    A = np.column_stack([np.ones(good.sum()), r])
    try:
        coef, _, _, _ = np.linalg.lstsq(A, Fr_h_obs, rcond=None)
        a0, a1 = float(coef[0]), float(coef[1])
        if not (np.isfinite(a0) and np.isfinite(a1)):
            raise ValueError("non-finite OLS solution")
        return {"a0": a0, "a1": a1}
    except Exception:                       # noqa: BLE001
        return {"a0": 0.576, "a1": 0.071}


def predict(X: np.ndarray, a0: float, a1: float) -> np.ndarray:
    """Q = (a0 + a1*(h/p)) * sqrt(g) * h^(3/2) * b  [m³ s⁻¹].

    X: (n, 3) — columns [h_m, p_m, b_m].
    """
    h = np.asarray(X[:, 0], dtype=float)
    p = np.asarray(X[:, 1], dtype=float)
    b = np.asarray(X[:, 2], dtype=float)
    return _Q(h, p, b, a0, a1)
