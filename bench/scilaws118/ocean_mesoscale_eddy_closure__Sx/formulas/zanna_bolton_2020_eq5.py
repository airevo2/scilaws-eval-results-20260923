"""Zanna and Bolton (2020) Eq. (5) — barotropic single-coefficient closure.

Citation: Zanna & Bolton, Geophys. Res. Lett., 47, e2020GL088376 (2020),
DOI: 10.1029/2020GL088376. Eq. (5), PDF p. 6.

Formula (zonal component):
    Sx = kappa_BT * [ d(zeta^2)/dx  -  d(zeta*D)/dx  +  d(zeta*D~)/dy ]
       = kappa_BT * ( d_zeta2_dx - d_zetaD_dx + d_zetaDtilde_dy )

This is the single-kappa simplification of the full Eq. (4) expression, derived
by approximating all six independent RVM weights as equal: w_i ≈ kappa_BT.
The coefficient of variation across the six weights is 14.2%; the approximation
introduces an average error of 14.2% in the predicted forcing. The expression
captures ~55.6% of variance in Sx in the paper's MITgcm barotropic validation.

LAW_CONSTANTS (paper p. 6, Eq. (5)):
  kappa_BT = -4.87e8 m^2  — barotropic single-kappa approximation. Value stated
  explicitly at PDF p. 6: "w_i ≈ kappa_BT = -4.87e10^8 m^2" (= -4.87e8 m^2).

OTHER_CONSTANTS:
  (none) — the formula is dimensionally self-consistent without any auxiliary
  scalars; all required field dimensions are carried by the derivative columns.

Type designation: Type I — kappa_BT is a single global scalar (not per-cluster
refit). LOCAL_FITTABLE = {}. All 68 608 rows are from one simulation; no
natural cluster identifier.

Column mapping (paper notation -> released CSV column names):
  d/dx(zeta^2)  -> d_zeta2_dx      [s^-2 m^-1]
  d/dx(zeta*D)  -> d_zetaD_dx      [s^-2 m^-1]
  d/dy(zeta*D~) -> d_zetaDtilde_dy [s^-2 m^-1]

Caveats:
  - kappa_BT was fit on an MITgcm barotropic double-gyre run (3.75 km, filtered
    to 30 km). The benchmark data use a pyqg two-layer QG eddy run (Ross 2023
    §2.1). Ross 2023 §4 reports R^2 ~ 0.3-0.5 for ZB2020 on this pyqg dataset,
    somewhat below the original ~0.56, reflecting resolution and domain
    differences.
  - Sign: kappa_BT < 0 reflects the backscatter / inverse-cascade nature of
    the parameterization (energy injection into resolved scales).
"""

import numpy as np

USED_INPUTS = ["d_zeta2_dx", "d_zetaD_dx", "d_zetaDtilde_dy"]
PAPER_REF = "summary_formula_dataset_zanna_2020.md"
EQUATION_LOC = "Zanna & Bolton 2020 Eq. (5), PDF p. 6; kappa_BT value PDF p. 6"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "kappa_BT": -4.87e8,   # m^2; barotropic single-kappa; ZB2020 PDF p. 6 Eq. (5)
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}       # no auxiliary scalars needed
LOCAL_FITTABLE = {}        # Type I — no per-cluster parameters


def predict(X: np.ndarray, kappa_BT: float) -> np.ndarray:
    """Predict zonal subgrid eddy forcing Sx [m s^-2].

    X: (n, 3) — columns [d_zeta2_dx, d_zetaD_dx, d_zetaDtilde_dy].
    kappa_BT: barotropic single-coefficient [m^2].
    """
    d_zeta2_dx      = np.asarray(X[:, 0], dtype=float)
    d_zetaD_dx      = np.asarray(X[:, 1], dtype=float)
    d_zetaDtilde_dy = np.asarray(X[:, 2], dtype=float)
    return kappa_BT * (d_zeta2_dx - d_zetaD_dx + d_zetaDtilde_dy)
