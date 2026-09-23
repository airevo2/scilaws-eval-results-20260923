"""Ross et al. (2023) Eq. (A7) — ZB2020 baroclinic-shape closure (partial form).

Citation: Ross, Li, Perezhogin, Fernandez-Granda, Zanna,
J. Advances Modeling Earth Systems, 15, e2022MS003258 (2023),
DOI: 10.1029/2022MS003258. Eq. (A7), PDF pp. 26-27.

Full formula (Ross 2023 Eq. A7, PDF p. 26 — vector form, zonal component):
    S_u^ZB2020 ≈ kappa_ZB2020 * div( [[-zeta*D, zeta*D~],
                                       [zeta*D~,  zeta*D]] )
                  + I * (1/2) * kappa_ZB2020 * grad(zeta^2 + D^2 + D~^2)

Expanding the zonal (x) component:
    Sx ≈ kappa * ( -d(zeta*D)/dx + d(zeta*D~)/dy
                   + (1/2) * d(zeta^2 + D^2 + D~^2)/dx )
       = kappa * ( -d_zetaD_dx + d_zetaDtilde_dy
                   + (1/2)*d_zeta2_dx + (1/2)*d_D2_dx + (1/2)*d_Dtilde2_dx )

PARTIAL APPROXIMATION: The benchmark data_raw/build_data.py computes only the
three primary ZB2020 basis derivatives: d_zeta2_dx, d_zetaD_dx, d_zetaDtilde_dy.
The terms d_D2_dx and d_Dtilde2_dx are absent from the released data columns.
This module therefore implements the PARTIAL form dropping those two terms:

    Sx ≈ kappa * ( -d_zetaD_dx + d_zetaDtilde_dy + (1/2)*d_zeta2_dx )

This partial form coincides with ZB2020 Eq. (6) applied to the barotropic
component only (Eq. 6 decomposes as: S_u^BC = 2*S_u^BT + I*kappa_BC*grad(D^2+D~^2);
the last term is dropped here).

LAW_CONSTANTS (ZB2020 PDF p. 6, Eq. (6) text):
  kappa_ZB2020 = -8.723e8 m^2  — baroclinic single-kappa from ZB2020.
  ZB2020 states: "mean value kappa_BC = -8.723 × 10^8 m^2" (PDF p. 6).
  Ross 2023 (PDF p. 27, Eq. A8 text) notes that for online tests, kappa_ZB2020
  is fit empirically; the -8.723e8 value is the physically grounded paper anchor.

OTHER_CONSTANTS:
  (none) — formula is dimensionally self-consistent given derivative columns.

Type designation: Type I — kappa_ZB2020 is a global scalar. LOCAL_FITTABLE = {}.

Column mapping (paper notation -> released CSV column names):
  d/dx(zeta*D)  -> d_zetaD_dx      [s^-2 m^-1]
  d/dy(zeta*D~) -> d_zetaDtilde_dy [s^-2 m^-1]
  d/dx(zeta^2)  -> d_zeta2_dx      [s^-2 m^-1]

Caveats:
  - Missing d_D2_dx and d_Dtilde2_dx terms reduce achievable skill. The
    dropped terms together represent the gradient of (D^2 + D~^2)/2, which
    for deformation-dominated flows can be significant.
  - The baroclinic kappa (-8.723e8 m^2) is larger in magnitude than the
    barotropic kappa (-4.87e8 m^2); applying it to this partial form on the
    barotropic-regime pyqg data may give poorer skill than Eq. (5).
  - Ross 2023 fits kappa empirically for online tests; the frozen ZB2020 value
    serves as the published constant baseline for this benchmark entry.
"""

import numpy as np

USED_INPUTS = ["d_zeta2_dx", "d_zetaD_dx", "d_zetaDtilde_dy"]
PAPER_REF = "summary_formula+dataset_ross_2023.md"
EQUATION_LOC = (
    "Ross 2023 Eq. (A7), PDF p. 26 (zonal component, partial — D^2 and D~^2 "
    "gradient terms absent from data); kappa_ZB2020 = kappa_BC from "
    "Zanna & Bolton 2020 Eq. (6), PDF p. 6"
)

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "kappa_ZB2020": -8.723e8,   # m^2; ZB2020 baroclinic kappa; ZB2020 PDF p. 6 Eq. (6) text
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}       # no auxiliary scalars
LOCAL_FITTABLE = {}        # Type I — no per-cluster parameters


def predict(X: np.ndarray, kappa_ZB2020: float) -> np.ndarray:
    """Predict zonal subgrid eddy forcing Sx [m s^-2].

    X: (n, 3) — columns [d_zeta2_dx, d_zetaD_dx, d_zetaDtilde_dy].
    kappa_ZB2020: baroclinic single-kappa [m^2] from ZB2020 Eq. (6) / Ross Eq. (A7).

    Implements the partial form of Ross 2023 Eq. (A7) omitting d_D2_dx and
    d_Dtilde2_dx, which are not present in the benchmark data.
    """
    d_zeta2_dx      = np.asarray(X[:, 0], dtype=float)
    d_zetaD_dx      = np.asarray(X[:, 1], dtype=float)
    d_zetaDtilde_dy = np.asarray(X[:, 2], dtype=float)
    return kappa_ZB2020 * (-d_zetaD_dx + d_zetaDtilde_dy + 0.5 * d_zeta2_dx)
