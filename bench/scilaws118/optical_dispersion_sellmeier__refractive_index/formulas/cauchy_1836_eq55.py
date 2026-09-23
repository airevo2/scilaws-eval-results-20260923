"""Cauchy (1836) canonical three-term dispersion formula — Eq (55), journal p. 215.

Cauchy, A.-L. (1836). *Mémoire sur la dispersion de la lumière*.
Prague: J. G. Calve. Nouveaux Exercices de Mathématiques, Vol. IV.
Public domain (copyright expired). No DOI.

Eq (55), journal p. 215 (PDF p. 227):

    θ² = a + b·s² + c·s⁴

where θ = n (refractive index), s ∝ 1/λ (frequency / wavenumber). In
modern wavelength notation with λ in µm this reads:

    n²(λ) = A + B/λ² + C/λ⁴

Cauchy derives this by truncating the full four-term Eq (15) (see
cauchy_1836_eq15.py), justifying the truncation by showing that the sixth-
power finite differences Δ²sᵢ⁶ are of the same order as the observation
errors (~0.001 in the last digit of n) — Tables I–III, journal pp. 211–213,
PDF pp. 222–225.  Eq (57) (journal p. 217, PDF p. 229) then gives explicit
three-coefficient fits for water, crown-glass, and flint-glass in Cauchy's
normalised s-units.

The benchmark target is n (not n²); we return √(A + B/λ² + C/λ⁴) after
clipping the radicand to a small positive floor to guard against unphysical
values near UV/IR absorption edges outside the validated visible range.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The three-term polynomial FORM n²(λ) = A + B/λ² + C/λ⁴ is the
scientific contribution; Cauchy gives explicit fits only for water /
crown-glass / flint-glass in his own normalised s-units, which are not
directly transferable to the modern µm scale used in the Bond (1965)
dataset. All three coefficients are LOCAL_FITTABLE per material.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The powers 2 and 4 in B/λ² + C/λ⁴ are fixed by the even-powers
molecular-frequency expansion (Eq (12), journal p. 206); they are algebraic
structure, not tunable constants.

LOCAL_FITTABLE — per-material, fitted by fit() via bounded linear LS
---------------------------------------------------------------------
- A : low-frequency limit of n²; intercept; dimensionless; typically 1.5–7.0
      (n² for the materials in the Bond dataset ranges from ~2.2 for AlPO4
      to ~25 for GaP in the IR).
- B : second-order dispersion coefficient (µm²); positive for normal
      dispersion; typical magnitude 0.001–0.1 µm².
- C : fourth-order dispersion coefficient (µm⁴); may be positive or
      negative; very small relative to B; typical magnitude 1e-6–1e-3 µm⁴.

fit() solves the overdetermined linear system n²ᵢ = A + B/λᵢ² + C/λᵢ⁴
in the least-squares sense via numpy.linalg.lstsq — no nonlinear
optimisation needed (the model is linear in {A, B, C}).
"""

import numpy as np

USED_INPUTS = ["wavelength_um"]
PAPER_REF = "summary_formula_cauchy_1836.md"
EQUATION_LOC = (
    "Cauchy (1836) Eq (55), journal p. 215 (PDF p. 227): "
    "θ² = a + b·s² + c·s⁴; modern form n²(λ) = A + B/λ² + C/λ⁴."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "A": {"init": None},
    "B": {"init": None},
    "C": {"init": None},
}

# Small floor to prevent sqrt of a non-positive radicand near absorption bands.
_N2_FLOOR = 1e-6


def _n2(lam, A, B, C):
    """Squared refractive index: A + B/λ² + C/λ⁴."""
    lam2 = lam * lam
    return A + B / lam2 + C / (lam2 * lam2)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Ordinary least-squares fit of A, B, C to observed n values.

    The Cauchy three-term formula is linear in {A, B, C} when the target is
    n²(λ), so we fit n²_obs = A + B/λ² + C/λ⁴ via numpy.linalg.lstsq
    (minimum-norm least-squares, stable on typical optical datasets).

    Fallback: if lstsq fails or yields non-finite parameters, we return
    physically reasonable defaults (A=2.0, B=0.01, C=0.0) that give n ≈ 1.42
    at λ = 1 µm — a neutral, non-material-specific starting point.
    """
    lam = np.asarray(X_fit[:, 0], dtype=float)
    n_obs = np.asarray(y_fit, dtype=float)
    n2_obs = n_obs ** 2

    lam2 = lam * lam
    # Design matrix [1, 1/λ², 1/λ⁴]
    M = np.column_stack([np.ones_like(lam), 1.0 / lam2, 1.0 / (lam2 * lam2)])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(M, n2_obs, rcond=None)
        A, B, C = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        if not np.all(np.isfinite([A, B, C])):
            raise RuntimeError("non-finite coefficients")
        return {"A": A, "B": B, "C": C}
    except Exception:                                  # noqa: BLE001
        return {"A": 2.0, "B": 0.01, "C": 0.0}


def predict(X: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    """Cauchy three-term refractive index: n(λ) = sqrt(A + B/λ² + C/λ⁴).

    X: (n_samples, 1) — column [wavelength_um].

    Radicand is clipped to _N2_FLOOR before sqrt to avoid NaN on
    unphysical (negative n²) values outside the visible-range validity domain.
    """
    lam = np.asarray(X[:, 0], dtype=float)
    n2 = _n2(lam, A, B, C)
    return np.sqrt(np.clip(n2, _N2_FLOOR, None))
