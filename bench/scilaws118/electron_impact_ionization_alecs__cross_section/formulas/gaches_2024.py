"""Gaches et al. (2024) BEB / dBEB electron-impact ionization cross-section.

Gaches, B. A. L. et al. (2024). The Astrochemistry Low-energy Electron
Cross-Section (ALeCS) database I. Semi-empirical electron-impact ionization
cross-section calculations and ionization rates.
Astronomy & Astrophysics 675, A25. DOI:10.1051/0004-6361/202348293.
arXiv:2310.10739.

This module implements two closely related BEB-family variants from the paper:

Variant 1 — BEB (binary-encounter Bethe; Eqs. 8-9 of Gaches 2024, which
follow Kim & Rudd 1994 Phys. Rev. A 50:3954):

    sigma_BEB = sum_ell  S_ell / (t_ell + u_ell + 1)
                * [ (1/2)(1 - 1/t_ell^2)
                  + (ln t_ell / t_ell)(1 - 1/t_ell - ln(t_ell)/(t_ell+1)) ]

where for each molecular orbital ell:
    t_ell = E_e / B_ell        (reduced electron energy, >= 1 when E_e >= B_ell)
    u_ell = U_ell / B_ell      (kinetic-to-binding energy ratio)
    S_ell = 4*pi*a0^2 * n_ell * R_inf / B_ell    (amplitude, cm^2)
    B_ell — orbital binding energy (eV)
    U_ell — orbital kinetic energy (eV)
    n_ell — orbital occupation number (integer, 1 or 2)
    a0    — Bohr radius = 5.29177210903e-9 cm  (NIST CODATA 2018)
    R_inf — Rydberg constant in eV = 13.605693122994 eV  (NIST CODATA 2018)

Threshold: sigma_BEB = 0 for E_e < B_ell for each orbital (inner sum
  contribution of orbital ell vanishes when t_ell < 1).

Variant 2 — dBEB (damped BEB; Eq. 10 of Gaches 2024):

    B'_ell = B_ell * exp(-(1 - B_ell/IP))

Replace B_ell -> B'_ell (and recompute t_ell, u_ell, S_ell with B'_ell)
before evaluating the BEB formula. This dampens orbitals whose binding
energy exceeds the molecular ionization potential IP, preventing
overestimation at low energies.

APPLICATION TO BENCHMARK:
The benchmark dataset (Dorn & Upendranath 2025, Zenodo) contains total
measured cross-sections for molecules as a function of electron energy E_e.
Each molecule is one cluster. The molecular orbital covariates (B_ell,
U_ell, n_ell, IP) are supplied per cluster and are NOT fitted — they are
physical observables from quantum chemistry.

In the benchmark, molecular orbital data are NOT available as direct
per-row inputs (the dataset rows are (E_e, sigma) pairs). Therefore the
orbital parameters must be fitted collectively as effective per-cluster
constants. We parameterize the orbital sum with a small number of effective
orbitals (n_orb = 2), each with fittable (B_ell, U_ell, n_ell).

LAW_CONSTANTS — paper-published, frozen universal *defining* coefficients
------------------------------------------------------------------------
None. The BEB form (Eqs. 8-9) carries no cross-group invariant fitted
coefficient of its own: its only universal numbers are CODATA constants
(a0, R_inf — see OTHER below) and structural literals (4*pi, 1/2), while the
per-orbital B/U/n are LOCAL (re-fit per cluster). The functional form is the
scientific claim, so LAW_CONSTANTS = {} (the legitimately-empty-LAW Type II
case). (Contrast: the paper's separate SCAR atom-basis Eq. (2) does carry
fitted c_k coefficients — but that model is not implemented here.)

OTHER_CONSTANTS — given CODATA constants the form consumes
---------------------------------------------------------
a0_cm : Bohr radius = 5.29177210903e-9 cm — CODATA universal constant
        (NIST CODATA 2018; Kim & Rudd 1994 Eq. 8; Gaches 2024 Eq. 9).
R_inf_eV : Rydberg constant = 13.605693122994 eV — CODATA universal constant
        (NIST CODATA 2018; Kim & Rudd 1994 Eq. 8; Gaches 2024 Eq. 9).
Both enter only the BEB amplitude S_ell = 4*pi*a0^2 * n_ell * R_inf / B_ell;
they are *given*, not the paper's discovery target, hence OTHER (exposed to
the SR as candidate priors), not LAW.
(Structural literals that stay inline: the 4*pi solid-angle factor and the
1/2 in (1/2)(1 - 1/t^2) are analytic results of the BEB derivation.)

LOCAL_FITTABLE — per-cluster (per-molecule), fitted by fit()
------------------------------------------------------------
We use n_orb=2 effective orbitals per molecule:

  B1 : effective binding energy of orbital 1 (eV, > 0) — lowest B = IP
  U1 : effective kinetic energy of orbital 1 (eV, > 0)
  n1 : effective occupation of orbital 1 (treated as continuous > 0 for fitting)
  B2 : effective binding energy of orbital 2 (eV, > B1)
  U2 : effective kinetic energy of orbital 2 (eV, > 0)
  n2 : effective occupation of orbital 2 (treated as continuous > 0)

init = None on all: fit() builds data-derived starting values.

The dBEB variant additionally uses:
  IP : molecular ionization potential (eV, > 0; constrained to <= B1)

For the BEB variant, IP is not needed in predict() because no damping is
applied; only the BEB formula parameters are used.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["electron_energy_eV"]
# Target unit: 10^-16 cm^2 (same as data/train.csv column cross_section_1e16cm2)
_UNIT_FACTOR_INV = 1e16  # multiply cm^2 by 1e16 to get 10^-16 cm^2 units
PAPER_REF = "summary_formula_dataset_gaches_2024.md"
EQUATION_LOC = (
    "Gaches et al. (2024) Eqs. (8)-(9) for BEB; Eq. (10) for dBEB. "
    "PDF pp. 3 of arXiv:2310.10739. "
    "Original BEB: Kim & Rudd (1994) Phys. Rev. A 50:3954 Eq. (8)."
)

# ── Universal physical constants (NIST CODATA 2018) — GIVEN, not the law ─────
# a0 and R_inf are CODATA universal constants the BEB amplitude S_ell merely
# *consumes*; they are NOT the paper's defining/fitted coefficients. Per the
# four-field rule they are OTHER_CONSTANTS (given quantities), exposed to the SR
# as candidate priors. The BEB functional form carries no cross-group invariant
# fitted coefficient of its own, so LAW_CONSTANTS = {} (the form is the claim;
# the per-orbital B/U/n are LOCAL, re-fit per cluster).
LAW_CONSTANTS = {}

OTHER_CONSTANTS = {
    "a0_cm":    5.29177210903e-9,    # Bohr radius (cm); NIST CODATA 2018
    "R_inf_eV": 13.605693122994,     # Rydberg constant (eV); NIST CODATA 2018
}

LOCAL_FITTABLE = {
    "B1": {"init": None},   # effective binding energy, orbital 1 (eV)
    "U1": {"init": None},   # effective kinetic energy, orbital 1 (eV)
    "n1": {"init": None},   # effective occupation, orbital 1
    "B2": {"init": None},   # effective binding energy, orbital 2 (eV)
    "U2": {"init": None},   # effective kinetic energy, orbital 2 (eV)
    "n2": {"init": None},   # effective occupation, orbital 2
}

# Pre-compute the prefactor 4*pi*a0^2 once (from the OTHER given constants).
_a0 = OTHER_CONSTANTS["a0_cm"]        # cm
_Rinf = OTHER_CONSTANTS["R_inf_eV"]   # eV
_4PI_A0SQ = 4.0 * np.pi * _a0 ** 2   # cm^2


def _S_ell(n, R_inf, B):
    """BEB amplitude S_ell = 4*pi*a0^2 * n * R_inf / B (cm^2).

    Gaches 2024 Eq. (9); Kim & Rudd 1994 Eq. (8).
    """
    return _4PI_A0SQ * n * R_inf / B


def _beb_integrand(t, u):
    """Inner BEB integrand (dimensionless, Gaches 2024 Eqs. 8-9).

    Returns the bracket:
        (1/2)(1 - 1/t^2) + (ln(t)/t) * (1 - 1/t - ln(t)/(t+1))

    Valid for t >= 1 (i.e. E_e >= B_ell).  Returns 0 for t < 1.
    """
    t = np.asarray(t, dtype=float)
    result = np.zeros_like(t)
    ok = t >= 1.0
    if not ok.any():
        return result
    tk = t[ok]
    ln_t = np.log(tk)
    term1 = 0.5 * (1.0 - 1.0 / tk ** 2)
    term2 = (ln_t / tk) * (1.0 - 1.0 / tk - ln_t / (tk + 1.0))
    result[ok] = term1 + term2
    return result


def _sigma_beb_orbitals(E, Bs, Us, ns):
    """BEB cross-section for a list of orbitals (cm^2).

    E  : array of electron energies (eV)
    Bs : list/array of binding energies B_ell (eV)
    Us : list/array of kinetic energies U_ell (eV)
    ns : list/array of occupation numbers n_ell
    """
    E = np.asarray(E, dtype=float)
    sigma = np.zeros_like(E)
    for B, U, n in zip(Bs, Us, ns):
        S = _S_ell(n, _Rinf, B)
        t = E / B
        u = U / B
        denom = t + u + 1.0
        integrand = _beb_integrand(t, u)
        # Contribution vanishes for t < 1 (handled inside _beb_integrand)
        with np.errstate(invalid="ignore", divide="ignore"):
            contrib = np.where(denom > 0, S / denom * integrand, 0.0)
        sigma += contrib
    return np.clip(sigma, 0.0, None)


def _sigma_dbeb_orbitals(E, Bs, Us, ns, IP):
    """dBEB cross-section — replace B_ell -> B'_ell = B_ell * exp(-(1-B_ell/IP)).

    Gaches 2024 Eq. (10).
    E  : array of electron energies (eV)
    Bs, Us, ns : orbital parameters
    IP : molecular ionization potential (eV)
    """
    Bs_prime = [B * np.exp(-(1.0 - B / IP)) for B in Bs]
    return _sigma_beb_orbitals(E, Bs_prime, Us, ns)


# ── Public fit / predict (BEB variant) ──────────────────────────────────────

def fit(X_fit: np.ndarray, y_fit: np.ndarray, **law_constants) -> dict:
    """Fit two effective-orbital BEB parameters to (E, sigma) data.

    X_fit: (n, 1) — electron_energy_eV column.
    y_fit: (n,)  — cross_section in 10^-16 cm^2 (released units).
    law_constants: empty for BEB (LAW_CONSTANTS = {}); accepted for harness compat.
    The Bohr radius / Rydberg constant are OTHER given constants, read from the
    module-level _4PI_A0SQ / _Rinf (derived from OTHER_CONSTANTS).

    The least-squares is run in the released 10^-16 cm^2 units (order-1 values)
    with Jacobian column scaling (`x_scale="jac"`); fitting the raw cm^2 residual
    (~1e-16) starves the optimizer (cost ~1e-32 < tolerance), so it terminates at
    the start point and the parameters never move. A small deterministic
    multi-start over (B1, B2/B1, n) makes the fit robust to local minima without
    any RNG, so repeated harness seeds give identical results.

    Starting values:
    - B1 ~ lowest energy with sigma > 0 (~ threshold/IP); U ~ B (virial-like).
    - B2 a few times B1 (inner-shell estimate).
    - n1 sized to the peak amplitude (S = 4*pi*a0^2*n*R_inf/B); n2 ~ 0.5 n1.
    """
    E = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)             # released 10^-16 cm^2 units (order 1)
    y_cm2_peak = (float(np.max(y)) if np.max(y) > 0 else 1.0) / _UNIT_FACTOR_INV

    pos = y > 0
    B1_base = float(np.min(E[pos])) if pos.any() else float(np.min(E)) * 0.9
    B1_base = max(B1_base, 1.0)

    lo = [0.5,  0.1,   0.05, 1.0,   0.1,   0.05]
    hi = [500., 5000., 200., 5000., 5000., 200.]

    def residual(p):
        B1, U1, n1, B2, U2, n2 = p
        B2 = max(B2, B1 + 0.5)
        return _sigma_beb_orbitals(E, [B1, B2], [U1, U2], [n1, n2]) * _UNIT_FACTOR_INV - y

    best, best_cost = None, np.inf
    # Deterministic multi-start (no RNG): (B1 fraction, B2/B1 ratio, occupation factor).
    for fB1, rB2, fn in ((0.9, 4.0, 1.0), (0.7, 4.0, 2.0),
                         (0.9, 8.0, 0.5), (0.7, 8.0, 1.0)):
        B1_0 = max(B1_base * fB1, 1.0)
        n1_0 = float(np.clip(3.0 * y_cm2_peak * B1_0 / (_4PI_A0SQ * _Rinf), 0.5, 50.0)) * fn
        p0 = [B1_0, B1_0, n1_0, B1_0 * rB2, B1_0 * rB2, 0.5 * n1_0]
        p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]
        try:
            sol = least_squares(residual, p0, bounds=(lo, hi), method="trf",
                                x_scale="jac", max_nfev=2000, ftol=1e-10, xtol=1e-10)
        except Exception:                          # noqa: BLE001
            continue
        if sol.cost < best_cost and np.all(np.isfinite(sol.x)):
            best_cost, best = sol.cost, sol.x

    if best is None:
        B1_0 = B1_base
        best = [B1_0, B1_0, 2.0, 5.0 * B1_0, 5.0 * B1_0, 1.0]

    B1, U1, n1, B2, U2, n2 = (float(v) for v in best)
    B2 = max(B2, B1 + 0.5)
    return {"B1": B1, "U1": U1, "n1": n1, "B2": B2, "U2": U2, "n2": n2}


def predict(X: np.ndarray,
            B1: float = 10.0, U1: float = 10.0, n1: float = 2.0,
            B2: float = 50.0, U2: float = 50.0, n2: float = 1.0,
            **kwargs) -> np.ndarray:
    """BEB cross-section (Gaches 2024 Eqs. 8-9) with two effective orbitals.

    X: (n, 1) — column [electron_energy_eV].
    Receives the LOCAL params (B1..n2) via **params. The Bohr radius / Rydberg
    constant are OTHER given constants (module-level _4PI_A0SQ / _Rinf), not
    arguments. Returns sigma in 10^-16 cm^2 (matching cross_section_1e16cm2).
    """
    E = np.asarray(X[:, 0], dtype=float)
    B2 = max(B2, B1 + 0.5)
    sigma_cm2 = _sigma_beb_orbitals(E, [B1, B2], [U1, U2], [n1, n2])
    return sigma_cm2 * _UNIT_FACTOR_INV  # -> 10^-16 cm^2
