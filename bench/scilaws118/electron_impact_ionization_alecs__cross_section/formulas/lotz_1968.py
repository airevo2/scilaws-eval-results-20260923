"""Lotz (1968) semi-empirical electron-impact ionization cross-section.

Lotz, W. (1968). Electron-impact ionization cross-sections and ionization
rate coefficients for atoms and ions from hydrogen to calcium.
Zeitschrift fur Physik 216(3):241-247. DOI:10.1007/BF01392963.

Eq. (1) of Lotz (1968) — total single-ionization cross-section from the
ground state, summed over N subshells:

    sigma(E) = sum_{i=1}^{N}  a_i * q_i * ln(E / P_i) / (E * P_i)
                               * {1 - b_i * exp[-c_i * (E/P_i - 1)]},
               for E >= P_i;   sigma(E) = 0  for E < P_i.

Here:
  E     — kinetic energy of the impact electron (eV)
  P_i   — binding energy of electrons in subshell i, with P_1 = ionization
           potential IP (eV); tabulated per species in Lotz (1968) Tables 1-2
  q_i   — number of equivalent electrons in subshell i (integer); tabulated
  a_i   — amplitude constant (units: 10^{-14} cm^2 eV^2); fit to experiment
           or assumed 4.5e-14 cm^2 eV^2 for four-times-and-higher ionized
           species not in the tables (PDF p. 242)
  b_i   — near-threshold shape constant (dimensionless); assumed 0 for
           four-times-and-higher ionized species
  c_i   — near-threshold decay constant (dimensionless, positive)
  N     — number of subshells: 1 for H/He-like; 2 for Li-through-Ne-like;
           3 for Na-through-Ca-like (PDF p. 241-242)

Remark on the dataset: The benchmark uses experimental cross-sections for
molecules (Dorn & Upendranath, Zenodo 2025), not isolated atoms/ions. The
Lotz formula is strictly derived for single atoms / ions H through Ca. When
applied here, each molecule is treated as a cluster and (P_i, q_i, a_i, b_i,
c_i) for N subshells are LOCAL_FITTABLE (effectively absorbing both the
molecular structure and the per-species constants). This is the natural
extension: the functional FORM is the scientific claim; the constants become
effective per-cluster fits. The number of subshells N is also treated as a
fittable hyperparameter (defaulting to 2 for molecules).

LAW_CONSTANTS — paper-published, frozen numerical values
---------------------------------------------------------
None. The Lotz formula introduces no universal dimensionless constant beyond
the algebraic form. The tables in Lotz (1968) are species-specific (per-atom)
values of (a_i, b_i, c_i, q_i, P_i), not universal law constants; they are
LOCAL_FITTABLE here.

OTHER_CONSTANTS — universal / unit-conversion factor
----------------------------------------------------
- lotz_amplitude_unit : the unit of the Lotz amplitude a_i, = 1e-14 cm^2 (eV)^2
  (Lotz 1968 — "when a_i is given in 10^-14 cm^2 (eV)^2, and P_i and T in eV",
  PDF p. 246; tabulated units PDF p. 242). It is a fixed dimensional / unit
  conversion factor that scales A*ln(E/P)/(E*P) [in 10^-14 cm^2 (eV)^2 / eV^2]
  to cm^2 — a GIVEN, NOT a paper-fitted coefficient → OTHER (MANUAL §1 kind-a,
  a unit conversion; mirrors the open_channel Manning k_n and binary_pulsar
  _DAY_TO_SEC givens). It is numerically folded into the per-cluster fitted
  amplitude A (A and 1e-14 are multiplicatively degenerate), so boxing it into
  OTHER and reading it from the dict is exactly behaviour-preserving; the move
  only makes the bare literal countable and consistent with its candidate prior
  `lotz_amplitude_unit`. Read from the dict in the body (gold style). The
  structure ln(E/P_i)/(E*P_i) — the Bethe high-energy asymptotic shape — stays
  as the algebraic FORM (no declared constant).

LOCAL_FITTABLE — per-cluster (per-molecule), fitted by fit()
------------------------------------------------------------
We use N=2 subshells as the default (one effective outer shell, one inner
shell), covering the Li-through-Ne-like class most relevant to the benchmark
molecules. The fitted parameters are:

  a1, b1, c1, P1, q1 — first (outermost) subshell
  a2, b2, c2, P2, q2 — second (inner) subshell

Because q_i must be a positive integer and the data do not constrain it
independently of a_i, we fold q_i into a_i and fit a single amplitude
  A_i = a_i * q_i  (units: 10^{-14} cm^2 eV^2 * electrons)
per subshell, with P_i as the threshold. b_i and c_i modulate the
near-threshold shape.

Parameterisation actually fitted:
  A1   = a1*q1  (10^{-14} cm^2 eV^2-electrons, > 0)
  P1   — threshold / IP for outer subshell (eV, > 0)
  b1   — near-threshold amplitude (0 <= b1 < 1)
  c1   — near-threshold decay (> 0)
  A2   = a2*q2  (10^{-14} cm^2 eV^2-electrons, > 0)
  P2   — threshold for inner subshell (eV, > P1)
  b2   — near-threshold amplitude (0 <= b2 < 1)
  c2   — near-threshold decay (> 0)

init = None on all: fit() builds data-derived starting values.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["electron_energy_eV"]
# Target unit: 10^-16 cm^2 (same as data/train.csv column cross_section_1e16cm2)
_UNIT_FACTOR_INV = 1e16  # multiply cm^2 by 1e16 to get 10^-16 cm^2 units
PAPER_REF = "summary_formula_lotz_1968.md"
EQUATION_LOC = (
    "Lotz (1968) Eq. (1), PDF p. 241-242 — "
    "sigma = sum_i a_i*q_i*ln(E/P_i)/(E*P_i)*{1 - b_i*exp[-c_i*(E/P_i-1)]}."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {
    # Unit of the Lotz amplitude a_i: 10^-14 cm^2 (eV)^2 (Lotz 1968, PDF p. 246
    # "when a_i is given in 10^-14 cm^2 (eV)^2"; p. 242). A unit-conversion GIVEN
    # (MANUAL §1 kind-a), not a fitted coefficient → OTHER. Numerically folded
    # into the per-cluster fitted amplitude A, so reading it from the dict is
    # behaviour-preserving.
    "lotz_amplitude_unit": 1.0e-14,   # cm^2 eV^2
}
LOCAL_FITTABLE = {
    "A1": {"init": None},  # a1*q1, 10^{-14} cm^2 eV^2-electrons
    "P1": {"init": None},  # outer-shell threshold / IP (eV)
    "b1": {"init": None},  # near-threshold amplitude, subshell 1
    "c1": {"init": None},  # near-threshold decay, subshell 1
    "A2": {"init": None},  # a2*q2, 10^{-14} cm^2 eV^2-electrons
    "P2": {"init": None},  # inner-shell threshold (eV, > P1)
    "b2": {"init": None},  # near-threshold amplitude, subshell 2
    "c2": {"init": None},  # near-threshold decay, subshell 2
}

# Conversion: a_i in paper is in units of 10^{-14} cm^2 (eV)^2.
# Cross-section output is in cm^2.  The factor 1e-14 converts the
# formula result from  [10^{-14} cm^2 eV^2] / [eV * eV] = 10^{-14} cm^2.
# Read from the OTHER_CONSTANTS dict (gold style; cf. open_channel _K_N).
_LOTZ_UNIT = OTHER_CONSTANTS["lotz_amplitude_unit"]  # cm^2 eV^2 (Lotz 1968 p. 246)


def _sigma_subshell(E, A, P, b, c):
    """Single-subshell Lotz cross-section contribution (cm^2).

    A  = a_i * q_i  in units of 10^{-14} cm^2 eV^2
    P  = binding energy / IP for this subshell (eV)
    b  = near-threshold amplitude (dimensionless)
    c  = near-threshold decay (dimensionless)
    """
    E = np.asarray(E, dtype=float)
    sig = np.zeros_like(E)
    above = E >= P
    if not above.any():
        return sig
    Ea = E[above]
    ratio = Ea / P                          # E/P_i (dimensionless)
    log_ratio = np.log(ratio)               # ln(E/P_i)
    near_thresh = b * np.exp(-c * (ratio - 1.0))
    sig[above] = (_LOTZ_UNIT * A * log_ratio / (Ea * P)) * (1.0 - near_thresh)
    return sig


def _sigma_total(E, A1, P1, b1, c1, A2, P2, b2, c2):
    """Two-subshell Lotz cross-section (cm^2)."""
    return _sigma_subshell(E, A1, P1, b1, c1) + _sigma_subshell(E, A2, P2, b2, c2)


def fit(X_fit: np.ndarray, y_fit: np.ndarray, **law_constants) -> dict:
    """Fit two-subshell Lotz parameters to (E, sigma) data.

    X_fit: (n, 1) — electron_energy_eV column.
    y_fit: (n,)  — cross_section in 10^-16 cm^2 (released units).
    law_constants: empty dict for Lotz (LAW_CONSTANTS = {}); accepted for harness compat.

    The least-squares is run in the released 10^-16 cm^2 units (order-1 values)
    with Jacobian column scaling (`x_scale="jac"`), so the optimizer is not
    starved by the ~1e-16 absolute magnitude of cm^2 cross-sections. (Fitting
    the raw cm^2 residual makes the cost ~1e-32, below the default tolerances,
    so the solver terminates after one step and the parameters never leave their
    starting values.) A small deterministic multi-start over (P1, P2/P1,
    amplitude) makes the fit robust to local minima without any RNG, so repeated
    harness seeds give identical results.

    Initialisation strategy:
    - P1: near the lowest energy with positive cross-section (~ threshold/IP).
    - P2: a few times P1 (inner shell deeper).
    - A1: sized so the peak (near E ~ e*P1) matches the data peak.
    - b1, b2: 0.5 (midrange near-threshold correction); c1, c2: 1.0.
    """
    E = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)             # released 10^-16 cm^2 units (order 1)

    pos_mask = y > 0
    P1_base = float(np.min(E[pos_mask])) if pos_mask.any() else float(np.min(E))
    P1_base = max(P1_base, 1.0)
    y_peak = float(np.max(y)) if np.max(y) > 0 else 1.0
    # predict() returns _sigma_total(cm^2) * _UNIT_FACTOR_INV; near E ~ e*P1
    # (ln = 1), sigma_1e16 ~ (_LOTZ_UNIT*_UNIT_FACTOR_INV) * A1 / (e*P1^2),
    # so A1 ~ y_peak * e * P1^2 / (_LOTZ_UNIT*_UNIT_FACTOR_INV).
    _amp = _LOTZ_UNIT * _UNIT_FACTOR_INV           # = 1e-14 * 1e16 = 100.0

    lo = [1e-6, 0.5,  0.0, 0.01, 1e-8, 1.0,  0.0, 0.01]
    hi = [1e8,  500., 0.99, 50.,  1e6,  5000., 0.99, 50.]

    def residual(p):
        A1, P1, b1, c1, A2, P2, b2, c2 = p
        P2 = max(P2, P1 + 1.0)                      # soft-enforce P2 > P1
        return _sigma_total(E, A1, P1, b1, c1, A2, P2, b2, c2) * _UNIT_FACTOR_INV - y

    best, best_cost = None, np.inf
    # Deterministic multi-start (no RNG): (P1 fraction, P2/P1 ratio, amplitude factor).
    for fP1, rP2, fA in ((0.9, 3.0, 1.0), (0.7, 3.0, 2.0),
                         (0.9, 6.0, 0.5), (0.7, 6.0, 1.0)):
        P1_0 = max(P1_base * fP1, 1.0)
        A1_0 = max(y_peak * np.e * P1_0 ** 2 / _amp, 1e-3) * fA
        p0 = [A1_0, P1_0, 0.5, 1.0, 0.1 * A1_0, P1_0 * rP2, 0.5, 1.0]
        p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]
        try:
            sol = least_squares(residual, p0, bounds=(lo, hi), method="trf",
                                x_scale="jac", max_nfev=2000, ftol=1e-10, xtol=1e-10)
        except Exception:                          # noqa: BLE001
            continue
        if sol.cost < best_cost and np.all(np.isfinite(sol.x)):
            best_cost, best = sol.cost, sol.x

    if best is None:
        P1_0 = P1_base
        A1_0 = max(y_peak * np.e * P1_0 ** 2 / _amp, 1e-3)
        best = [A1_0, P1_0, 0.5, 1.0, 0.1 * A1_0, 3.0 * P1_0, 0.5, 1.0]

    A1, P1, b1, c1, A2, P2, b2, c2 = (float(v) for v in best)
    P2 = max(P2, P1 + 1.0)
    return {"A1": A1, "P1": P1, "b1": b1, "c1": c1,
            "A2": A2, "P2": P2, "b2": b2, "c2": c2}


def predict(X: np.ndarray,
            A1: float = 1.0, P1: float = 10.0, b1: float = 0.5, c1: float = 1.0,
            A2: float = 0.1, P2: float = 30.0, b2: float = 0.5, c2: float = 1.0,
            **kwargs) -> np.ndarray:
    """Two-subshell Lotz (1968) electron-impact ionization cross-section.

    X: (n, 1) — column [electron_energy_eV].
    Receives LOCAL params via **params (LAW_CONSTANTS is empty for Lotz).
    Returns sigma in 10^-16 cm^2 (matching cross_section_1e16cm2 column).
    """
    E = np.asarray(X[:, 0], dtype=float)
    P2 = max(P2, P1 + 1.0)
    sigma_cm2 = np.clip(_sigma_total(E, A1, P1, b1, c1, A2, P2, b2, c2), 0.0, None)
    return sigma_cm2 * _UNIT_FACTOR_INV  # -> 10^-16 cm^2
