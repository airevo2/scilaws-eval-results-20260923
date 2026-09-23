"""Combined-tidal-deformability post-merger peak-frequency fit — Vretinaris et al. 2020.

Citation: S. Vretinaris, N. Stergioulas, A. Bauswein, Phys. Rev. D 101 (2020) 084039
(arXiv:1910.10856). Equation (33), PDF p. 25.

Formula
-------
    f_peak * M_chirp [kHz * M_sun] = b0 + b1 * M_chirp + b2 * Lambda_tilde^(-1/2)

Solved for f_peak [kHz]:
    f_peak [kHz] = (b0 + b1 * M_chirp + b2 * Lambda_tilde^(-1/2)) / M_chirp

LAW_CONSTANTS — Eq. (33), body text, PDF p. 25; confirmed txt line 1349:
    "fpeak Mchirp = 1.392 − 0.108Mchirp + 51.70Λ̃^(−1/2)"
    b0 = 1.392    — constant offset in units of kHz * M_sun
    b1 = -0.108   — chirp-mass slope [dimensionless in kHz * M_sun / M_sun units]
    b2 = 51.70    — tidal coefficient [kHz * M_sun]
    Paper reports R^2 = 0.985 and maximum residual 0.302 kHz.

OTHER_CONSTANTS — none; the formula is dimensionally self-consistent with
    M_chirp in M_sun, Lambda_tilde dimensionless, output in kHz.

Type designation: Type I. The three coefficients are globally calibrated on a
combined CFC/SPH + CoRe BNS catalogue (no per-cluster refit).
LOCAL_FITTABLE is empty.

Column mapping (paper -> released-CSV columns):
    M_chirp        = M_chirp  [M_sun]   — chirp mass of the binary
    Lambda_tilde   = lamT     [dimensionless] — combined tidal deformability

Caveats:
    Published validity range: M_chirp in [1.06, 1.94] M_sun, Lambda_tilde > 0.
    Outside this range no published guarantee; model extrapolates.
"""

import numpy as np

USED_INPUTS = ["lamT", "M_chirp"]
PAPER_REF   = "summary_formula+dataset_vretinaris_2020.md"
EQUATION_LOC = "Eq. (33), PDF p. 25; txt line 1349"

# Three globally calibrated coefficients — Eq. (33), PDF p. 25.
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "b0":  1.392,   # constant [kHz * M_sun]; txt line 1349
    "b1": -0.108,   # chirp-mass slope; txt line 1349
    "b2": 51.70,    # tidal coefficient [kHz * M_sun]; txt line 1349
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}  # formula is dimensionally self-consistent

LOCAL_FITTABLE = {}  # Type I — no per-cluster parameters; no fit()


def predict(X: np.ndarray, b0: float, b1: float, b2: float) -> np.ndarray:
    """Predict f_peak [kHz] for each BNS merger row.

    X: (n, 2) array — columns [lamT, M_chirp] per USED_INPUTS.
    Returns: (n,) array of f_peak predictions in kHz.
    """
    X = np.asarray(X, dtype=float)
    lamT    = X[:, 0]
    M_chirp = X[:, 1]

    # Eq. (33): f_peak * M_chirp = b0 + b1 * M_chirp + b2 * Lambda_tilde^(-1/2)
    rhs = b0 + b1 * M_chirp + b2 * np.power(lamT, -0.5)
    return rhs / M_chirp
