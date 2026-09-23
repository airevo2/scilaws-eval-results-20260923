"""Cauchy (1836) four-term dispersion formula — Eq (15), journal p. 206.

Cauchy, A.-L. (1836). *Mémoire sur la dispersion de la lumière*.
Prague: J. G. Calve. Nouveaux Exercices de Mathématiques, Vol. IV.
Public domain (copyright expired). No DOI.

Eq (15), journal p. 206 (PDF p. 220):

    Θᵢ = a + b·sᵢ² + c·sᵢ⁴ + δ·sᵢ⁶

where Θᵢ = θᵢ² = nᵢ² is the squared refractive index at Fraunhofer
line i, and sᵢ ∝ 1/λᵢ is the angular frequency. In modern wavelength
notation with λ in µm:

    n²(λ) = A + B/λ² + C/λ⁴ + D/λ⁶

This is the full four-term truncation of the master series Eq (4)
(journal p. 205, PDF p. 217). Cauchy subsequently argues (journal p. 215,
Tables I–III) that the sixth-power term δsᵢ⁶ is negligible within the
visible range (fourth-order finite differences Δ²sᵢ⁴ ≈ observation
error ~0.001), leading to the canonical three-term Eq (55) implemented in
cauchy_1836_eq55.py. The four-term form is retained here as a distinct
closed-form variant because it is the first fully-specified formula in the
paper and includes the additional δ/λ⁶ term.

The benchmark target is n; we return √(A + B/λ² + C/λ⁴ + D/λ⁶) after
clipping the radicand to a small positive floor.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The four-term polynomial FORM is the scientific claim. The
coefficients a, b, c, δ are per-material fit parameters in Cauchy's work;
no universal numerical values applicable to the Bond (1965) µm-scale data
are published. All four coefficients are LOCAL_FITTABLE.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The exponents 2, 4, 6 in B/λ² + C/λ⁴ + D/λ⁶ are fixed by the
even-powers molecular-frequency expansion (master series Eq (4), journal
p. 205); they are algebraic structure, not tunable constants.

LOCAL_FITTABLE — per-material, fitted by fit() via bounded linear LS
---------------------------------------------------------------------
- A : low-frequency intercept of n²; dimensionless; typically 1.5–7.0.
- B : 1/λ² coefficient (µm²); positive for normal dispersion;
      typical magnitude 0.001–0.1 µm².
- C : 1/λ⁴ coefficient (µm⁴); small; typical magnitude 1e-6–1e-3 µm⁴.
- D : 1/λ⁶ coefficient (µm⁶); sixth-order correction; Cauchy shows it is
      negligible in the visible range; may be positive or negative;
      typical magnitude 1e-8–1e-5 µm⁶.

fit() solves the overdetermined linear system
n²ᵢ = A + B/λᵢ² + C/λᵢ⁴ + D/λᵢ⁶ via numpy.linalg.lstsq.  The model is
linear in {A, B, C, D}, so no nonlinear optimisation is required.
"""

import numpy as np

USED_INPUTS = ["wavelength_um"]
PAPER_REF = "summary_formula_cauchy_1836.md"
EQUATION_LOC = (
    "Cauchy (1836) Eq (15), journal p. 206 (PDF p. 220): "
    "Θᵢ = a + b·sᵢ² + c·sᵢ⁴ + δ·sᵢ⁶; modern form n²(λ) = A + B/λ² + C/λ⁴ + D/λ⁶."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "A": {"init": None},
    "B": {"init": None},
    "C": {"init": None},
    "D": {"init": None},
}

# Small floor to prevent sqrt of a non-positive radicand near absorption bands.
_N2_FLOOR = 1e-6


def _n2(lam, A, B, C, D):
    """Squared refractive index: A + B/λ² + C/λ⁴ + D/λ⁶."""
    lam2 = lam * lam
    lam4 = lam2 * lam2
    return A + B / lam2 + C / lam4 + D / (lam4 * lam2)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Ordinary least-squares fit of A, B, C, D to observed n values.

    The Cauchy four-term formula is linear in {A, B, C, D} when the target is
    n²(λ), so we fit n²_obs = A + B/λ² + C/λ⁴ + D/λ⁶ via numpy.linalg.lstsq
    (minimum-norm least-squares).

    Note: with only a modest number of data points per material (14–24 in the
    Bond dataset) and four free parameters, the D column (1/λ⁶) can be nearly
    collinear with the C column (1/λ⁴), making D ill-determined. lstsq
    handles this gracefully via the minimum-norm solution, but D may be
    effectively zero on sparse datasets — consistent with Cauchy's own
    conclusion that the sixth-order term is negligible.

    Fallback: if lstsq fails or yields non-finite parameters, we return
    physically reasonable defaults (A=2.0, B=0.01, C=0.0, D=0.0).
    """
    lam = np.asarray(X_fit[:, 0], dtype=float)
    n_obs = np.asarray(y_fit, dtype=float)
    n2_obs = n_obs ** 2

    lam2 = lam * lam
    lam4 = lam2 * lam2
    # Design matrix [1, 1/λ², 1/λ⁴, 1/λ⁶]
    M = np.column_stack([
        np.ones_like(lam),
        1.0 / lam2,
        1.0 / lam4,
        1.0 / (lam4 * lam2),
    ])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(M, n2_obs, rcond=None)
        A, B, C, D = (float(coeffs[i]) for i in range(4))
        if not np.all(np.isfinite([A, B, C, D])):
            raise RuntimeError("non-finite coefficients")
        return {"A": A, "B": B, "C": C, "D": D}
    except Exception:                                  # noqa: BLE001
        return {"A": 2.0, "B": 0.01, "C": 0.0, "D": 0.0}


def predict(X: np.ndarray, A: float, B: float, C: float, D: float) -> np.ndarray:
    """Cauchy four-term refractive index: n(λ) = sqrt(A + B/λ² + C/λ⁴ + D/λ⁶).

    X: (n_samples, 1) — column [wavelength_um].

    Radicand is clipped to _N2_FLOOR before sqrt to avoid NaN on
    unphysical (negative n²) values outside the visible-range validity domain.
    """
    lam = np.asarray(X[:, 0], dtype=float)
    n2 = _n2(lam, A, B, C, D)
    return np.sqrt(np.clip(n2, _N2_FLOOR, None))
