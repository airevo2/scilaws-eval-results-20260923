"""Hertz spherical-contact law — AFM indentation force F(h).

Mendova et al. (2024). *Why Some Cells Appear Stiffer than Others: AFM
Nanoindentation as an Analytical Tool to Assess the Mechanical Properties
of Liposomes.* International Journal of Molecular Sciences 25(13):7186.
DOI: 10.3390/ijms25137186. License: CC-BY-4.0.

The paper introduces the task formula in §1 (PDF p.2, main formula block) and
rederives it in full in Appendix A, culminating at eq. A26 (PDF p.10):

    F = (4/3) * E* * sqrt(R) * h^(3/2)

where:
  - h (also written delta) is the indentation depth (m, measured from
    contact point; positive into sample).
  - R is the spherical tip radius (m); for the zef18 dataset R = 1.864e-5 m
    (constant across all curves; supplied as the per-row input column R_m).
  - E* = E / (1 - nu^2) is the reduced Young's modulus (Pa).  For the
    zef18 pipeline nu = 0.5 is assumed fixed (PDF p.6, §4.4 of Mendova 2024
    citing Thomas et al. 2013), so E* = (4/3) E, but E* is fitted directly
    as the free per-cluster parameter, absorbing nu.
  - The prefactor 4/3 arises from the Boussinesq pressure-distribution
    integral (eqs. A25-A26, PDF p.10) and is an EXACT rational of the
    spherical Hertz form — derived, not fitted.
  - The exponents 3/2 on h and 1/2 on R are fixed by spherical contact
    geometry (eq. A24, PDF p.10: a^2 = R*h, contact radius from geometry).

Because the formula is linear in E*, the fit reduces to an analytic
ordinary-least-squares problem on the design matrix
    X_design = (4/3) * sqrt(R) * h^(3/2)
with E* as the single scalar coefficient.  No iterative optimiser is needed.

LAW_CONSTANTS — paper-published, frozen cross-cluster coefficients
------------------------------------------------------------------
(empty.)  This spherical Hertz law has NO fitted cross-cluster constant: the
functional FORM itself is the entire scientific claim, and the one per-cluster
quantity (E*) varies from curve to curve (-> LOCAL_FITTABLE).  The numerical
prefactor 4/3 is NOT a fitted defining coefficient — it is an EXACT rational
derived in closed form from the Boussinesq pressure integral (eq. A25 yields
F = (2/3)*p0*pi*a^2, which on substitution gives the 4/3 of eq. A26, PDF p.10).
It is therefore a STRUCTURAL rational of the form (like 3/2, 1/2), kept inline
in predict()/fit() — not declared in any field — exactly as the gold reference
keeps the `3` of its Coulomb prefactor and `2/3`, `Z^2/5` inline.  The paper
confirms it is exact/structural, not fitted (summary §7: "The prefactor 4/3 is
exact and universal ... inconsistent with the theory's structural derivation"
to absorb it into a free fitted coefficient).

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The exponents 3/2 (on h) and 1/2 (on R) are derived from the
spherical contact geometry (eq. A24, PDF p.10) and are not free constants
— they define the model family, not a particular numerical choice — so they
stay inline as structural exponents.

LOCAL_FITTABLE — per-cluster, fitted by fit() via analytic OLS
---------------------------------------------------------------
- E_star : reduced Young's modulus (Pa, > 0). Absorbs material stiffness
  and Poisson ratio (nu = 0.5 assumed). Per-curve values for the zef18
  zebrafish dataset are in the range ~0.3-2 kPa (PDF p.3, §2 of Mendova
  2024; PROVENANCE). Fitted per AFM force-indentation curve (each curve
  = one cluster).

init = None: fit() uses an analytic OLS solution (formula is linear in
E_star), so no iterative starting point is required.
"""

import numpy as np

USED_INPUTS = ["indentation_m", "R_m"]
PAPER_REF = "summary_formula_mendova_2024.md"
EQUATION_LOC = (
    "Mendova 2024 §1 main formula (PDF p.2) and Appendix A eq. A26 (PDF p.10): "
    "F = (4/3)*E**sqrt(R)*h^(3/2); prefactor from Boussinesq integral eqs. A25-A26."
)

# This spherical Hertz law has no fitted cross-cluster coefficient: the FORM is
# the claim; E* is per-cluster (LOCAL). The 4/3 prefactor and the 3/2 / 1/2
# exponents are EXACT structural rationals of the form, kept inline in
# predict()/fit() (gold convention — never declared as constants).
LAW_CONSTANTS = {}

OTHER_CONSTANTS = {}

LOCAL_FITTABLE = {
    "E_star": {"init": None},
}


def _design_col(h, R):
    """Scalar feature phi(h, R) such that F = E_star * phi.

    phi = (4/3) * sqrt(R) * h^(3/2). The 4/3 prefactor and the 3/2 / 1/2
    exponents are exact structural rationals of the spherical Hertz form
    (Appendix A, eqs. A24-A26, PDF p.10) — kept inline, not declared.
    """
    h_clamped = np.maximum(h, 0.0)        # Hertz formula applies only for h >= 0
    return (4.0 / 3.0) * np.sqrt(np.abs(R)) * h_clamped ** 1.5


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Analytic OLS fit of E_star from the Hertz spherical-contact law.

    Signature: fit(X_fit, y_fit, **LAW_CONSTANTS); LAW_CONSTANTS is empty
    (the 4/3 prefactor is an inline structural rational, not a LAW constant),
    so this fit() takes no LAW kwargs.

    Because F = E_star * [(4/3) * sqrt(R) * h^(3/2)], the formula is linear
    in E_star.  Optimal E_star is therefore the slope of a no-intercept OLS
    regression:

        E_star = sum(phi * F) / sum(phi^2),   phi = (4/3)*sqrt(R)*h^(3/2).

    This is exact, deterministic, and requires no iterative optimiser.
    E_star is clipped to [1.0, 1e9] Pa after the regression to ensure
    physicality.
    """
    h    = np.asarray(X_fit[:, 0], dtype=float)
    R    = np.asarray(X_fit[:, 1], dtype=float)
    y    = np.asarray(y_fit, dtype=float)

    phi = _design_col(h, R)

    denom = float(np.dot(phi, phi))
    if denom < 1e-30:
        # Degenerate case: all h = 0 (no in-contact points); return mid-range
        E_star = 1e3
    else:
        E_star = float(np.dot(phi, y) / denom)

    # Physical constraint: E* > 0; floor at 1 Pa to avoid non-physical values.
    E_star = float(np.clip(E_star, 1.0, 1e9))
    return {"E_star": E_star}


def predict(X: np.ndarray, E_star: float) -> np.ndarray:
    """Hertz spherical-contact force: F = (4/3) * E* * sqrt(R) * h^(3/2).

    X: (n, 2) — columns [indentation_m, R_m].
    The 4/3 prefactor and the 3/2 / 1/2 exponents are inline structural
    rationals of the spherical Hertz form (not declared constants).
    """
    h = np.asarray(X[:, 0], dtype=float)
    R = np.asarray(X[:, 1], dtype=float)
    phi = _design_col(h, R)
    return E_star * phi
