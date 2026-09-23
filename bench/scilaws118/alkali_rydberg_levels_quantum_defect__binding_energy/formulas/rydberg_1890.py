"""Rydberg (1890) spectral series formula -- binding energy of alkali levels.

Rydberg, J. R. (1890). Recherches sur la constitution des spectres d'emission
des elements chimiques. Kongl. Svenska Vetenskaps-Akademiens Handlingar,
Vol. 23, No. 11, pp. 1-155. Public domain (>100 years old).

Equation (9) of the paper (journal p. 40-41, PDF pp. 41-42):

    n_line = n_0 - N_0 / (m + mu)^2

where n_line is the spectral-line wavenumber, n_0 is the series limit,
m is the integer line-order index (= principal quantum number n_QM of the
outer electron for alkalis), and mu is the quantum defect for the series.

Binding energy (term value) relative to the ionization limit is defined as

    E_bind = n_0 - n_line = N_0 / (m + mu)^2

so n_0 drops out of the binding-energy expression; only N_0 and mu enter.
This is the formula directly implemented here.

LAW_CONSTANTS -- empty
----------------------
Form-discovery Type II task: the claim is the inverse-square series FORM plus
the per-series quantum defect. N_0 is the universal Rydberg constant -- a known
universal physical constant, NOT a paper-specific fitted coefficient -- so it is
OTHER, not LAW (cf. the gold exemplar's universal CODATA constant e^2 -> OTHER;
MANUAL Direction B: a universal/CODATA constant in LAW is demoted to OTHER).

OTHER_CONSTANTS -- universal given the formula consumes
-------------------------------------------------------
N_0 : 109721.6 cm^-1 -- Rydberg's 1890 value of the universal Rydberg constant
      N_0, obtained from the Balmer series (journal p. 42, PDF p. 42). Rydberg's
      contribution was showing N_0 is universal across all elements and series;
      the value is a (19th-century-precision) measurement of the universal
      Rydberg constant, slightly below the modern CODATA value. The formula
      consumes it as a fixed scale; it is not the per-cluster discovery target.

LOCAL_FITTABLE -- per-cluster (per (element, series)) quantum defect
--------------------------------------------------------------------
mu : quantum defect (dimensionless). Per-cluster constant. Rydberg's mu.
     Typical range: 0 (hydrogenic) to ~4 (Cs s-series).
     fit() uses a data-derived single-start least-squares.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["n_qm"]
PAPER_REF = "summary_formula_rydberg_1890.md"
EQUATION_LOC = (
    "Rydberg 1890 Eq. (9), journal p. 40-41, PDF pp. 41-42: "
    "n = n_0 - N_0/(m+mu)^2; binding energy = N_0/(m+mu)^2."
)

LAW_CONSTANTS = {}
# Universal given consumed by the form (not fitted; not the SR discovery target).
# Exposed as a prior candidate. See docstring + MANUAL §2 (Direction B).
OTHER_CONSTANTS = {
    "N_0": 109721.6,  # cm^-1; Rydberg's 1890 value of the universal Rydberg constant; PDF p. 42
}
LOCAL_FITTABLE = {
    "mu": {"init": None},
}


def _binding_energy(n_qm, mu):
    """E_bind = N_0 / (n_qm + mu)^2  (cm^-1).  N_0 is a universal given (OTHER)."""
    N_0 = OTHER_CONSTANTS["N_0"]
    n_star = np.asarray(n_qm, dtype=float) + mu
    return N_0 / (n_star * n_star)


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Fit quantum defect mu for one (element, series) cluster.

    Data-derived initialisation:
      From E_bind = N_0 / (n + mu)^2  =>  n + mu = sqrt(N_0 / E_bind)
      =>  mu0 = median(sqrt(N_0 / E_bind) - n_qm).
    Single-start TRF via scipy.optimize.least_squares.
    """
    N_0 = OTHER_CONSTANTS["N_0"]
    n_qm = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    valid = y > 0.0
    if valid.any():
        n_star_est = np.sqrt(N_0 / y[valid])
        mu0 = float(np.median(n_star_est - n_qm[valid]))
    else:
        mu0 = 1.35  # fallback: approximate Na s-series value

    mu0 = float(np.clip(mu0, -5.0, 5.0))

    def residual(p):
        return _binding_energy(n_qm, p[0]) - y

    try:
        sol = least_squares(residual, [mu0],
                            bounds=([-5.0], [5.0]),
                            method="trf", max_nfev=2000)
        mu = float(sol.x[0])
        if not np.isfinite(mu):
            raise RuntimeError("non-finite fit")
        return {"mu": mu}
    except Exception:
        return {"mu": mu0}


def predict(X: np.ndarray, mu: float) -> np.ndarray:
    """Rydberg binding energy: E_bind = N_0 / (n_qm + mu)^2.

    X: (n_rows, 1) -- column [n_qm], the principal quantum number (integer).
    mu: per-cluster quantum defect (LOCAL_FITTABLE).
    N_0 is read from OTHER_CONSTANTS (the universal Rydberg constant given).
    """
    n_qm = np.asarray(X[:, 0], dtype=float)
    return _binding_energy(n_qm, mu)
