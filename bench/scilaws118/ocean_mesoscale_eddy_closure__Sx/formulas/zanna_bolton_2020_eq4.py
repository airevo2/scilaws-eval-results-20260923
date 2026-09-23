"""Zanna and Bolton (2020) Eq. (4) — full RVM barotropic expression (zonal component).

Citation: Zanna & Bolton, Geophys. Res. Lett., 47, e2020GL088376 (2020),
DOI: 10.1029/2020GL088376. Eq. (4), PDF p. 6.

Formula (zonal component only — this baseline predicts Sx):
    Sx = w0 * d(zeta^2)/dx  -  w1 * d(zeta*D)/dx  +  w2 * d(zeta*D~)/dy
       = w0 * d_zeta2_dx  -  w1 * d_zetaD_dx  +  w2 * d_zetaDtilde_dy

The full vector expression (Eq. 4, PDF p. 6) includes both Sx and Sy components
with six independent RVM weights (w0..w5); this module implements the zonal
component with the three relevant weights w0, w1, w2. The meridional component
(Sy) is out of scope for this benchmark task.

LAW_CONSTANTS (paper p. 6, Eq. (4) and text immediately following):
  w0 = -4.096e8 m^2  — ZB2020 PDF p. 6, stated: "w0 = -4.096 x 10^8"
  w1 = -5.483e8 m^2  — ZB2020 PDF p. 6, stated: "w1 = -5.483 x 10^8"
  w2 = -4.384e8 m^2  — ZB2020 PDF p. 6, stated: "w2 = -4.384 x 10^8"
  (Uncertainties per weight < 10%; coefficient of variation = 14.2%.)

OTHER_CONSTANTS:
  (none) — all unit factors are absorbed into the published weight values.

Type designation: Type I — weights w0..w2 are single global scalars from the
paper's RVM regression on one MITgcm simulation. LOCAL_FITTABLE = {}.

Column mapping (paper notation -> released CSV column names):
  d/dx(zeta^2)  -> d_zeta2_dx      [s^-2 m^-1]
  d/dx(zeta*D)  -> d_zetaD_dx      [s^-2 m^-1]
  d/dy(zeta*D~) -> d_zetaDtilde_dy [s^-2 m^-1]

Caveats:
  - The minus sign on the w1 term tracks the paper's tensor structure: the
    (1,1) diagonal of the Reynolds-stress tensor involves zeta^2 - zeta*D,
    while the off-diagonal term involves zeta*D~ (see ZB2020 §3.2).
  - Published weights are from the barotropic MITgcm run; the coefficient of
    variation 14.2% motivates the single-kappa Eq. (5) approximation.
  - This baseline and Eq. (5) share the same functional form but different
    coefficients; on the pyqg eddy data both attain R^2 ~ 0.3-0.5 (Ross 2023 §4).
"""

import numpy as np

USED_INPUTS = ["d_zeta2_dx", "d_zetaD_dx", "d_zetaDtilde_dy"]
PAPER_REF = "summary_formula_dataset_zanna_2020.md"
EQUATION_LOC = "Zanna & Bolton 2020 Eq. (4), PDF p. 6 (zonal component; weights w0..w2)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "w0": -4.096e8,   # m^2; ZB2020 PDF p. 6 Eq. (4)
    "w1": -5.483e8,   # m^2; ZB2020 PDF p. 6 Eq. (4)
    "w2": -4.384e8,   # m^2; ZB2020 PDF p. 6 Eq. (4)
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}       # no auxiliary scalars
LOCAL_FITTABLE = {}        # Type I — no per-cluster parameters


def predict(X: np.ndarray, w0: float, w1: float, w2: float) -> np.ndarray:
    """Predict zonal subgrid eddy forcing Sx [m s^-2].

    X: (n, 3) — columns [d_zeta2_dx, d_zetaD_dx, d_zetaDtilde_dy].
    w0, w1, w2: independent RVM weights [m^2] from Eq. (4).
    """
    d_zeta2_dx      = np.asarray(X[:, 0], dtype=float)
    d_zetaD_dx      = np.asarray(X[:, 1], dtype=float)
    d_zetaDtilde_dy = np.asarray(X[:, 2], dtype=float)
    # Note: explicit minus sign on w1 term per paper's tensor structure
    return w0 * d_zeta2_dx - w1 * d_zetaD_dx + w2 * d_zetaDtilde_dy
