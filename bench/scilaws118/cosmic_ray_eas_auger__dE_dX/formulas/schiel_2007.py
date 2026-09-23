"""Greisen electromagnetic cascade profile — energy-deposit proxy.

Schiel, R. W. and Ralston, J. P. (2007). *Reconciling the Greisen–Zatsepin–
Kuzmin paradox with the data.* Physical Review D 75:016005.
DOI: 10.1103/PhysRevD.75.016005. arXiv:hep-ph/0607248v2 (2007-01-16).

Schiel & Ralston (2007) provide the first complete derivation of the Greisen
(1956) equation for the integral electromagnetic cascade particle number as a
function of shower depth t (in radiation lengths):

    Pi(t; beta0) ≈ (0.31 / sqrt(beta0)) * exp(t * (1 - 1.5 * ln(s)))   [Eq. 6]

where:
    beta0 = ln(E0 / eps_c)                        [log energy parameter]
    s     = 3*t / (t + 2*beta0)                   [shower age, 0 -> 2]

PDF p. 2, Eq. 6; the prefactor 0.31 is traced by Schiel & Ralston to
0.3162/sqrt(pi) in their derivation (PDF p. 2, Eq. 14).

Mapping to the benchmark target (dE/dX vs X in g/cm^2):
  The shower depth in radiation lengths t = X / X_rad where X_rad = 36.7 g/cm^2
  in air (Schiel & Ralston 2007, PDF p. 2; standard air radiation length used
  throughout EAS literature). Since dE/dX is proportional to the local particle
  number in the EM cascade (each particle deposits approximately epsilon_c per
  radiation length), we have:
      dE_dX_approx(X) = A * Pi(X / X_rad; beta0)
  where A absorbs the per-shower normalization (proportional to E0 * eps_c /
  X_rad). This is an energy-deposit proxy via the Greisen cascade profile.

Physics caveat: Greisen's Pi(t) is the integral particle count (electrons +
positrons above eps_c), not the differential energy deposit per unit atmospheric
depth. The shape of Pi(t) versus t closely approximates the shape of dE/dX
versus X under the approximation that energy deposit is dominated by the EM
cascade and each particle deposits ~eps_c per radiation length. This is a
physics-motivated alternative baseline for the benchmark, not an exact model of
dE/dX. The direct model is the Gaisser-Hillas (R,L) form in aab_2019.py.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
- GREISEN_PREFACTOR = 0.31 : the numerical prefactor in Pi(t; beta0). Published in
  Greisen (1956) and given explicitly in Schiel & Ralston (2007) Eq. 6 (PDF p. 2)
  as the "0.31" multiplying 1/sqrt(beta0); confirmed verbatim in the extracted text
  ("replacing the coefficient 0.31 ...", PDF p. ~5). This is the genuine cross-shower
  invariant defining coefficient of the Greisen form — held fixed at eval, used in
  _greisen_pi(). (The 0.3162/sqrt(pi) that also appears in the paper is an
  intermediate derivation step, not the published prefactor; we use 0.31.)

OTHER_CONSTANTS — universal / structural / given physical constants
-------------------------------------------------------------------
- X_RAD_AIR = 36.7 : radiation length in air (g/cm^2). Standard physics constant
  for EM cascades; Schiel & Ralston use t = X/X_rad (PDF p. 2) but do not state 36.7
  numerically — it is the conventional air radiation length (PDG Cosmic Rays chapter).
  C10: universal physics constant -> OTHER. Consumed by _dEdX_model().
- EPS_C_AIR_MEV = 81.0 : critical energy in air (MeV); Schiel & Ralston 2007 (PDF
  p. 7: "the critical energy is eps_c = 81 MeV"). A material/physical constant, NOT a
  defining coefficient -> OTHER (Direction B). It is also INERT in this implementation:
  beta0 = ln(E0/eps_c) is fitted DIRECTLY as a per-shower LOCAL parameter, so eps_c is
  never consumed by predict()/fit(). Declared for completeness (documents the physics
  meaning of beta0); not in priors, no leak.
- The 3 and 2 in the shower age s = 3t/(t + 2*beta0) are algebraic coefficients
  from the cascade transport equations, not tunable constants.
- The 1.5 = 3/2 in the exponent (1 - 1.5*ln(s)) is likewise a structural
  coefficient from the Greisen approximation (Eq. 11, PDF p. 2).

LOCAL_FITTABLE — per-shower (per-cluster), fitted via nonlinear LS
-------------------------------------------------------------------
- A     : normalization constant (PeV/(g/cm^2) / [particle count]).
          Absorbs (dE/dX)_max and the eps_c / X_rad conversion factor,
          and accounts for per-shower energy.
- beta0 : log-energy parameter ln(E0 / eps_c). Per-shower/cluster, because
          primary energy E0 varies across showers. Typical Auger range ~25-50
          for E0 = 10^17.8 to 10^{19.5} eV at eps_c = 81 MeV.
- t0    : depth offset (radiation lengths); shifts the profile onset.
          Accounts for the fact that the Greisen form starts from t=0 at the
          point of first interaction, which varies per shower. Equivalent to
          a free X-axis shift for the cascade onset.
init = None on all: fit() builds its own data-derived start.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["X"]
PAPER_REF = "summary_formula_schiel_2007.md"
EQUATION_LOC = (
    "Schiel & Ralston (2007) Eq. 6, PDF p. 2 — "
    "Pi(t; beta0) = (0.31/sqrt(beta0)) * exp(t*(1 - 1.5*ln(s))), "
    "s = 3*t/(t + 2*beta0); "
    "mapped to dE/dX via dE_dX_approx = A * Pi((X - X0) / 36.7; beta0)."
)

# Paper-published numerical constant (Schiel & Ralston 2007)
LAW_CONSTANTS = {
    "GREISEN_PREFACTOR": 0.31,    # Eq. 6, PDF p. 2 — the published Greisen prefactor
                                  # (used in _greisen_pi; the genuine defining coefficient)
}

OTHER_CONSTANTS = {
    "X_RAD_AIR": 36.7,    # radiation length in air (g/cm^2); standard EAS value
                          # not stated numerically in Schiel & Ralston (2007);
                          # C10 classification: universal physics constant -> OTHER
    "EPS_C_AIR_MEV": 81.0,  # critical energy in air (MeV); PHYSICAL constant
                            # (Schiel & Ralston 2007 PDF p. 7: "eps_c = 81 MeV").
                            # Demoted LAW->OTHER (Direction B): a material constant,
                            # NOT a defining coefficient — and INERT here, since beta0
                            # is fitted directly per shower (never derived via eps_c).
}

LOCAL_FITTABLE = {
    "A":     {"init": None},
    "beta0": {"init": None},
    "t0":    {"init": None},
}


def _greisen_pi(t, beta0):
    """Greisen integral particle number Pi(t; beta0).

    Schiel & Ralston (2007) Eq. 6, PDF p. 2.
    Returns 0 for t <= 0 (shower not yet started) or where s <= 0.
    """
    GREISEN_PREFACTOR = LAW_CONSTANTS["GREISEN_PREFACTOR"]
    t = np.asarray(t, dtype=float)
    # Shower age s = 3t / (t + 2*beta0); defined for t > 0
    denom = t + 2.0 * beta0
    # Avoid divide-by-zero and log(0)
    safe_denom = np.where(denom > 1e-12, denom, 1e-12)
    s = 3.0 * t / safe_denom
    # Exponent: t*(1 - 1.5*ln(s)); for s <= 0 set result to 0
    # For s=0 (t=0), ln(s)->-inf so exp->0; profile is 0 at shower start.
    safe_s = np.where(s > 1e-15, s, 1e-15)
    exponent = t * (1.0 - 1.5 * np.log(safe_s))
    prefactor = GREISEN_PREFACTOR / np.sqrt(np.maximum(beta0, 1e-6))
    pi_val = prefactor * np.exp(exponent)
    # Zero-out unphysical region (t <= 0 or negative depth)
    return np.where(t > 0.0, pi_val, 0.0)


def _dEdX_model(X, A, beta0, t0):
    """Energy-deposit proxy from Greisen cascade profile.

    t = (X - X0_phys) / X_rad where X0_phys = t0 * X_rad is the shower-start
    depth in g/cm^2. Equivalently, t0 is the onset offset in radiation lengths.
    dE_dX_approx = A * Pi(t; beta0).
    """
    X_RAD = OTHER_CONSTANTS["X_RAD_AIR"]
    t = (X - t0 * X_RAD) / X_RAD     # t = (X - X_start) / X_rad
    return A * _greisen_pi(t, beta0)


def fit(X_fit: np.ndarray, y_fit: np.ndarray,
        GREISEN_PREFACTOR: float = None) -> dict:
    """Fit the Greisen energy-deposit proxy to per-shower profile data.

    Data-derived initialisation:
      A0     : set so that A * Pi(t_peak; beta0_0) matches the observed max.
      beta0_0: use beta0 = 30 as a starting value (corresponds to E0 ~ 10^18.5 eV
               at eps_c=81 MeV, midpoint of the Auger energy range).
      t0_0   : set so that the Greisen peak (s=1, i.e. t_max = beta0) aligns
               with the observed peak X value.

    The Greisen profile peaks at shower age s=1, i.e. t_max = beta0.
    So X_peak_obs ≈ (t_max + t0) * X_rad = (beta0_0 + t0) * X_rad
    -> t0 = X_peak_obs / X_rad - beta0_0.
    """
    # Use passed-in LAW values (harness passes them as kwargs) or fall back to module dict
    _PREF = GREISEN_PREFACTOR if GREISEN_PREFACTOR is not None else LAW_CONSTANTS["GREISEN_PREFACTOR"]
    X_RAD = OTHER_CONSTANTS["X_RAD_AIR"]    # always use module-level value (not in LAW)
    X = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # Initial estimates
    idx_peak = int(np.argmax(y))
    X_peak = float(X[idx_peak])
    y_peak = float(np.maximum(y[idx_peak], 1e-12))

    beta0_0 = 30.0                          # mid-range starting value
    t0_0 = X_peak / X_RAD - beta0_0        # align Greisen peak to observed peak
    # A0: scale so Pi at peak matches y_peak
    GREISEN_PREFACTOR = _PREF
    # At t = beta0, s = 1, so Pi = 0.31/sqrt(beta0) * exp(beta0 * (1 - 0)) = 0.31/sqrt(beta0) * exp(beta0)
    pi_at_peak = GREISEN_PREFACTOR / np.sqrt(beta0_0) * np.exp(beta0_0)
    A0 = y_peak / max(pi_at_peak, 1e-30)
    if not np.isfinite(A0) or A0 <= 0:
        A0 = 1e-30

    p0 = [A0, beta0_0, t0_0]

    # Bounds: A > 0 (no upper physically meaningful limit within LS);
    # beta0 in [10, 60] (E0 from ~10^15 to ~10^20 eV at eps_c=81 MeV);
    # t0 in [-30, 30] r.l. (onset depth offset).
    lo = [1e-40, 10.0, -30.0]
    hi = [1e10,  60.0,  30.0]
    p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]

    def residual(p):
        return _dEdX_model(X, p[0], p[1], p[2]) - y

    try:
        sol = least_squares(residual, p0, bounds=(lo, hi),
                            method="trf", max_nfev=5000)
        A, beta0, t0 = sol.x
        if not np.all(np.isfinite([A, beta0, t0])):
            raise RuntimeError("non-finite fit")
        return {"A": float(A), "beta0": float(beta0), "t0": float(t0)}
    except Exception:                                   # noqa: BLE001
        return {"A": float(p0[0]), "beta0": float(p0[1]), "t0": float(p0[2])}


def predict(X: np.ndarray, A: float, beta0: float, t0: float,
            GREISEN_PREFACTOR: float = None) -> np.ndarray:
    """Greisen electromagnetic cascade energy-deposit proxy.

    X: (n, 1) — column [X] with slant atmospheric depth (g/cm^2).
    Returns dE/dX proxy in PeV/(g/cm^2), via
        dE_dX ≈ A * Pi((X - t0*X_rad) / X_rad; beta0).
    The harness passes the single LAW constant GREISEN_PREFACTOR as a kwarg;
    the internal functions read GREISEN_PREFACTOR (LAW) and X_RAD_AIR (OTHER)
    from the module-level dicts. EPS_C_AIR_MEV is an inert OTHER constant
    (not consumed by the implemented form).
    """
    X_arr = np.asarray(X[:, 0], dtype=float)
    return _dEdX_model(X_arr, A, beta0, t0)
