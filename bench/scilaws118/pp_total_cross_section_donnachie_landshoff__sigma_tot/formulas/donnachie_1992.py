"""Donnachie & Landshoff (1992) Regge-pomeron total cross-section formula.

Donnachie, A. and Landshoff, P.V. (1992).  "Total cross sections."
Physics Letters B, 296(1-2), 227-232.  DOI: 10.1016/0370-2693(92)90832-O.
OA preprint: arXiv:hep-ph/9209205.

NOTE: donnachie_1992.pdf on disk is an image-only scan; all equations and
constants below were verified by direct PDF page inspection (Read tool) —
pdf_to_text.py produces near-empty output and cannot be used for this PDF.

=== FORMULA ===
Equation (2), PDF p. 3:

    sigma^TOT = X * s^epsilon + Y * s^{-eta}

where s is the Mandelstam variable in GeV^2. The first term arises from
pomeron exchange (rising at high energy); the second from rho/omega/f/a
Reggeon exchange (falling). Applied to pp scattering.

=== LAW_CONSTANTS — paper's scientific claim, frozen ===
From PDF p. 3 (simultaneous fit to pp and pbar-p data for sqrt(s) > 10 GeV):
    epsilon = 0.0808   — pomeron intercept minus 1 (effective power)
    eta     = 0.4525   — Reggeon power (effective)

From PDF p. 10, Figure 1(a) legend (verbatim):
    "pp: 21.70 s^{0.0808} + 56.08 s^{-0.4525}"
    X_pp = 21.70   — pomeron coupling coefficient for pp [mb]
    Y_pp = 56.08   — Reggeon coupling coefficient for pp [mb]

=== OTHER_CONSTANTS ===
None — s is provided directly as a column in the released CSV.

=== Type designation ===
Type I: the DL formula is a single global fit over all pp measurements;
X_pp, Y_pp, epsilon, eta are universal constants for pp scattering (no
per-cluster refit). LOCAL_FITTABLE is empty.

=== Column mapping ===
Paper symbol  | Units  | CSV column | Position
sigma^TOT     | mb     | sigma_tot  | Column 0 (output)
s             | GeV^2  | s          | Column 1 (input)

=== Caveats ===
- The DL formula was fit at sqrt(s) > 10 GeV (Eq. 2 context, PDF p. 3).
  The benchmark train split uses sqrt(s) in [5, 100) GeV; the test split
  uses sqrt(s) >= 100 GeV. At very high energies (LHC, cosmic ray > 7 TeV)
  the 1992 fit underestimates sigma_tot because LHC data were not available
  then. This is a documented hard OOD probe (see metadata.yaml note).
- s is the Mandelstam variable computed from p_lab via the kinematic
  relation s = 2*m_p^2 + 2*m_p*sqrt(p_lab^2 + m_p^2), m_p = 0.93827 GeV.
  Only s is released as an input column; p_lab is not needed by the formula.
"""

import numpy as np

USED_INPUTS = ["s"]
PAPER_REF = "summary_formula_donnachie_1992.md"
EQUATION_LOC = "Eq. (2), PDF p. 3; pp coefficients in Figure 1(a) legend, PDF p. 10"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "X_pp":    21.70,   # mb — pomeron coupling for pp; Figure 1(a) legend, PDF p. 10
    "epsilon":  0.0808, # pomeron effective power; PDF p. 3 (text after Eq. 2)
    "Y_pp":    56.08,   # mb — Reggeon coupling for pp; Figure 1(a) legend, PDF p. 10
    "eta":      0.4525, # Reggeon effective power; PDF p. 3 (text after Eq. 2)
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}    # Type I — no per-cluster parameters


def predict(X: np.ndarray, X_pp: float, epsilon: float,
            Y_pp: float, eta: float) -> np.ndarray:
    """Predict pp total cross-section in mb using the DL Regge formula.

    Parameters
    ----------
    X : np.ndarray, shape (n, 1)
        Column 0: s — Mandelstam variable in GeV^2.
    X_pp, epsilon, Y_pp, eta : float
        LAW_CONSTANTS (pomeron and Reggeon coupling + powers for pp).

    Returns
    -------
    sigma_tot : np.ndarray, shape (n,)
        Predicted total pp cross-section in mb.
    """
    s = np.asarray(X[:, 0], dtype=float)
    return X_pp * np.power(s, epsilon) + Y_pp * np.power(s, -eta)
