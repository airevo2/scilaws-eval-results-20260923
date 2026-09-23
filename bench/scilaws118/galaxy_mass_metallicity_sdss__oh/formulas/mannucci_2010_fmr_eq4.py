"""Mannucci et al. (2010) FMR — 1D projection in mu_0.32 (Eq. 4).

Mannucci, F. et al. 2010, MNRAS, 408, 2115.
Equation (4), PDF p. 9.

Formula
-------
    12 + log(O/H) = 8.90 + 0.39*x - 0.20*x**2 - 0.077*x**3 + 0.064*x**4

where:
    mu_0.32 = log10(M_star) - 0.32 * log10(SFR)   [Eq. 3, PDF p. 9]
    x       = mu_0.32 - 10

This is the 1D projection of the Fundamental Metallicity Relation (FMR) onto
the mu_0.32 axis. The value alpha=0.32 minimises the residual dispersion of
median metallicities of SDSS galaxies around the relation (Fig. 5, left panel,
PDF p. 9). The resulting 4th-order polynomial in x achieves a residual scatter
of ~0.05 dex. High-redshift galaxies (z < 2.5) follow this same relation
(Fig. 5, right panel, PDF p. 9).

The quantity mu_alpha = log(M_*) - alpha*log(SFR) reduces the 2D FMR surface
(Eq. 2) to a 1D projection; at alpha=0.32 the SFR-metallicity anticorrelation
and mass-metallicity correlation partially cancel, yielding tighter scatter
than either mass or SFR alone.

LAW_CONSTANTS — paper-published, frozen (PDF p. 9, Eqs. 3-4)
---------------------------------------------------------
    alpha = 0.32 : FMR projection coefficient in mu_alpha = log(M_*) -
                   alpha*log(SFR) — the value that MINIMISES the residual
                   metallicity scatter (Eq. 3 / Fig. 5, PDF p. 9). This is a
                   defining coefficient of the FMR projection: the whole
                   1D-projection construction and the g0..g4 polynomial below
                   are conditional on this alpha. (Was previously hidden as a
                   module global `_ALPHA` and consumed in predict() without
                   being a graded LAW constant — moved into LAW per the
                   field-classification re-audit 2026-05-30.)
    g0 = 8.90    : constant offset at x=0 — PDF p. 9 Eq. (4)
    g1 = 0.39    : linear-x coefficient — PDF p. 9 Eq. (4)
    g2 = -0.20   : quadratic-x coefficient — PDF p. 9 Eq. (4)
    g3 = -0.077  : cubic-x coefficient — PDF p. 9 Eq. (4)
    g4 = 0.064   : quartic-x coefficient — PDF p. 9 Eq. (4)

OTHER_CONSTANTS — none
---------------------------------------------------------
    (empty)

Type designation: Type I — each galaxy is an independent row; no per-cluster
fit; LOCAL_FITTABLE = {}.

Column mapping: log_Mstar -> log10(M_star/M_sun), log_SFR -> log10(SFR/[M_sun/yr]).
mu_0.32 = log_Mstar - 0.32 * log_SFR; x = mu_0.32 - 10.

Caveats: Mannucci et al. (2010) use the Kewley-Ellison R23 metallicity
calibration, while the released dataset uses MPA-JHU Bayesian strong-line
values (OH_P50). A systematic offset of ~0.06-0.08 dex is expected. The
1D projection compresses the 2D FMR into a single axis; at very high or low
SFR this compression introduces additional scatter compared to Eq. (2).
The polynomial is valid over the SDSS calibration range mu_0.32 ~ 8.5-11.5.
"""

import numpy as np

USED_INPUTS = ["log_Mstar", "log_SFR"]
PAPER_REF   = "summary_formula_dataset_mannucci_2010.md"
EQUATION_LOC = "Mannucci et al. 2010, MNRAS 408 2115, Eq. (4), PDF p. 9"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "alpha": 0.32,  # FMR projection coeff (mu_alpha) — Eq. 3, PDF p. 9
    "g0":  8.90,    # constant — PDF p. 9 Eq. (4)
    "g1":  0.39,    # linear x — PDF p. 9 Eq. (4)
    "g2": -0.20,    # quadratic x — PDF p. 9 Eq. (4)
    "g3": -0.077,   # cubic x — PDF p. 9 Eq. (4)
    "g4":  0.064,   # quartic x — PDF p. 9 Eq. (4)
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE  = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, alpha: float, g0: float, g1: float, g2: float,
            g3: float, g4: float) -> np.ndarray:
    """Predict 12+log(O/H) from the mu_0.32 FMR projection (Mannucci 2010 Eq. 4).

    X: (n, 2) — columns [log_Mstar, log_SFR].
    Returns: (n,) array of 12+log(O/H) in dex.

    Derivation:
        mu_alpha = log_Mstar - alpha * log_SFR  (Eq. 3, alpha=0.32)
        x        = mu_alpha - 10
        OH       = g0 + g1*x + g2*x**2 + g3*x**3 + g4*x**4  (Eq. 4)
    """
    log_Mstar = np.asarray(X[:, 0], dtype=float)
    log_SFR   = np.asarray(X[:, 1], dtype=float)
    mu = log_Mstar - alpha * log_SFR
    x  = mu - 10.0
    return g0 + g1 * x + g2 * x**2 + g3 * x**3 + g4 * x**4
