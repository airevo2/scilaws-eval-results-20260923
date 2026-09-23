"""Brutsaert-Nieber (1977) power-law baseflow recession — Q(t), integrated form.

Brutsaert, W. and Nieber, J. L. (1977). Regionalized drought flow hydrographs
from a mature glaciated plateau. *Water Resources Research*, 13(3), 637–643.
DOI: 10.1029/WR013i003p00637.

The BN77 ODE (Eq. 1 of Brutsaert & Nieber 1977; reproduced verbatim on PDF p. 3
of the open-access proxy Chor & Dias 2014, doi:10.5194/hessd-11-12519-2014):

    -dQ/dt = a * Q^b                                                       (BN77 Eq. 1)

where Q(t) is baseflow discharge (m³/s or ft³/s), a > 0 is the recession
amplitude coefficient, and b >= 1 is the nonlinearity exponent.

Closed-form integration for b != 1 (standard separation of variables):

    Q^(-b) dQ = -a dt
    => Q(t)^(1-b)/(1-b) - Q_0^(1-b)/(1-b) = -a * t
    => Q(t)^(1-b) - Q_0^(1-b) = -a * t * (1-b) = a * t * (b-1)
    => Q(t)^(1-b) = Q_0^(1-b) + a*(b-1)*t
    => Q(t) = [ Q_0^(1-b) + a*(b-1)*t ]^(1/(1-b))

This formula generalises the Maillet (b=1) exponential and covers the
classical Boussinesq (1904) late-time solution (b=3/2) and the early-time
Polubarinova-Kochina (b=3) solution, both cited in Roques et al. (2022)
(PDF p. 4, Eq. 4) and Chor & Dias (2014) (PDF pp. 3–5).

The b=1 limit recovers the Maillet exponential Q(t) = Q_0 * exp(-a*t) =
Q_0 * exp(-t/k) with k=1/a (L'Hopital applied to the power-law form;
stated in Roques et al. 2022, PDF p. 2: "When b=1, the aquifer drainage
is exponential with characteristic timescale tau = 1/a").

Equation location
-----------------
BN77 Eq. 1 (-dQ/dt = aQ^b) is not directly accessible (AGU paywall).
It is reproduced verbatim by Chor & Dias (2014) at PDF p. 3 (Eq. 1 of that
paper). The integrated form is derived by standard calculus and is given
implicitly in Chor & Dias (2014) at PDF p. 3–4 (the state-space Q × dQ/dt
analysis) and in the context of Roques et al. (2022) at PDF pp. 1–2, 4
(Eq. 1 and Eq. 4, b=3/2 Boussinesq late-time coefficient).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The functional FORM of BN77 (ODE -dQ/dt = aQ^b) is the scientific
claim; a and b are per-event (per-cluster) aquifer properties with no
universal values fixed by Brutsaert & Nieber 1977. In the benchmark context
the exponent b is a per-cluster fit parameter: shallow thin aquifers tend
to b ~ 1.5 (Boussinesq), thick/confined aquifers tend to b ~ 1, and
heterogeneous or compartmentalized hillslopes produce b > 1.5 (Roques 2022
PDF pp. 2–5).

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The integrated BN77 power-law form has no universal prefactors
beyond those in the formula structure (the exponents 1-b and 1/(1-b) are
algebraic consequences of the ODE integration).

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- a : recession amplitude coefficient (> 0, units: [Q]^(1-b) / day). Scales
      the discharge decay rate. Fitted per recession event.
- b : recession nonlinearity exponent (>= 1.0). Classic Boussinesq range
      b in [1, 1.5]; extended range b in [1, 4] accommodates compartmentalized
      hillslopes (Roques 2022 PDF pp. 2, 5). b = 1 recovers the exponential
      (but fit() clips to b > 1.001 to avoid the degenerate limit).

init = None on both: fit() builds a data-derived start using log-space
linear regression on log(-dQ/dt) vs log(Q) (the standard BN77 state-space
approach; Roques 2022 PDF pp. 1–2, Eq. 1 linearises to
log(-dQ/dt) = log(a) + b*log(Q), a simple linear regression for log(a),b).
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["t", "Q_0"]
PAPER_REF = "summary_supporting_chor_dias_2014.md"
EQUATION_LOC = (
    "Chor & Dias (2014) Eq. 1, PDF p. 3: -dQ/dt = alpha*Q^beta "
    "(reproducing BN77 Eq. 1); integrated by separation of variables: "
    "Q(t) = [Q_0^(1-b) + a*(b-1)*t]^(1/(1-b)) for b>1. "
    "Roques et al. (2022) Eq. 1 (PDF p. 1) and Eq. 4 (PDF p. 4, b=3/2 Boussinesq); "
    "b=1 limit cited on PDF p. 2: tau = 1/a."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "a": {"init": None},
    "b": {"init": None},
}

_B_LO, _B_HI = 1.001, 5.0   # b >= 1 physically; 1.001 avoids degenerate b=1 limit
_A_LO, _A_HI = 1e-8, 1e4    # a > 0


def _Q_t(t: np.ndarray, Q_0: np.ndarray, a: float, b: float) -> np.ndarray:
    """Q(t) = [Q_0^(1-b) + a*(b-1)*t]^(1/(1-b)) for b > 1.

    Derivation: integrate -dQ/dt = a*Q^b:
      Q^(1-b)/(1-b)|_Q0^Q(t) = -a*t
      Q(t)^(1-b) = Q_0^(1-b) + a*(b-1)*t
      Q(t) = [Q_0^(1-b) + a*(b-1)*t]^(1/(1-b))
    Since b>1: 1-b<0, so as inner grows with t, Q(t) decreases (correct).
    """
    one_minus_b = 1.0 - b
    # Q(t)^(1-b) = Q_0^(1-b) + a*(b-1)*t   (from integration of -dQ/dt = a*Q^b)
    # Since b > 1: 1-b < 0, so Q_0^(1-b) > 0 (decays), and a*(b-1)*t > 0 adds to it.
    # As t increases, inner increases (1-b < 0), meaning Q(t) = inner^(1/(1-b)) decreases. ✓
    inner = np.asarray(Q_0, dtype=float) ** one_minus_b + a * (b - 1.0) * np.asarray(t, dtype=float)
    inner = np.maximum(inner, 1e-20)   # physical floor (should never be hit for finite t)
    return inner ** (1.0 / one_minus_b)


def _log_deriv_init(t: np.ndarray, Q_0: np.ndarray, y: np.ndarray):
    """Estimate log(a) and b from finite-difference approximation of -dQ/dt.

    Uses adjacent-row finite differences to get (Q_mid, -dQ/dt_approx) pairs,
    then fits log(-dQ/dt) = log(a) + b*log(Q) (BN77 state-space linearisation).

    Returns (a0, b0) data-derived starting estimates.
    """
    # Sort by t within each Q_0 group (in practice each cluster is one event,
    # sorted by t).
    sort_idx = np.argsort(t)
    t_s, y_s = t[sort_idx], y[sort_idx]

    # Finite difference -dQ/dt (forward) at each pair of adjacent rows
    dt_diff = np.diff(t_s)
    dQ_diff = np.diff(y_s)
    neg_dQdt = -dQ_diff / np.where(dt_diff > 0, dt_diff, 1.0)  # -dQ/dt
    Q_mid   = (y_s[:-1] + y_s[1:]) / 2.0

    # Keep only positive derivatives and positive Q (physically meaningful)
    valid = (neg_dQdt > 0) & (Q_mid > 0)
    if valid.sum() < 3:
        # Too few valid pairs: use Maillet defaults (b~1.3, a~0.05)
        return 0.05, 1.3

    log_neg_dQdt = np.log(neg_dQdt[valid])
    log_Q        = np.log(Q_mid[valid])

    # OLS: log(-dQ/dt) = log(a) + b * log(Q)
    A_mat = np.column_stack([np.ones(valid.sum()), log_Q])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_neg_dQdt, rcond=None)
        log_a0, b0 = float(coeffs[0]), float(coeffs[1])
        a0 = float(np.exp(log_a0))
    except Exception:   # noqa: BLE001
        return 0.05, 1.3

    # Clamp to physically plausible range
    a0 = float(np.clip(a0, _A_LO * 10, _A_HI / 10))
    b0 = float(np.clip(b0, _B_LO + 0.01, _B_HI - 0.1))
    return a0, b0


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear LS fit of the BN77 integrated power-law recession.

    Uses the log-space BN77 state-space linearisation for a deterministic
    starting estimate (a0, b0), then polishes with bounded NLS.

    Parameters
    ----------
    X_fit : (n, 2) array — columns [t, Q_0].
    y_fit : (n,) array — observed Q_t (discharge, > 0).

    Returns
    -------
    dict with keys "a" and "b".
    """
    t   = np.asarray(X_fit[:, 0], dtype=float)
    Q_0 = np.asarray(X_fit[:, 1], dtype=float)
    y   = np.asarray(y_fit, dtype=float)

    # Data-derived initial estimates
    valid = (y > 0) & np.isfinite(t) & np.isfinite(Q_0) & np.isfinite(y)
    if valid.sum() >= 3:
        a0, b0 = _log_deriv_init(t[valid], Q_0[valid], y[valid])
    else:
        a0, b0 = 0.05, 1.3

    # Clamp to bounds
    a0 = float(np.clip(a0, _A_LO, _A_HI))
    b0 = float(np.clip(b0, _B_LO, _B_HI))

    def residual(p):
        a_, b_ = p
        y_pred = _Q_t(t, Q_0, a_, b_)
        return y_pred - y

    try:
        sol = least_squares(
            residual,
            [a0, b0],
            bounds=([_A_LO, _B_LO], [_A_HI, _B_HI]),
            method="trf",
            max_nfev=8000,
        )
        a_fit, b_fit = float(sol.x[0]), float(sol.x[1])
        if not (np.isfinite(a_fit) and np.isfinite(b_fit) and a_fit > 0 and b_fit > 0):
            raise RuntimeError("non-finite or non-positive fit")
        return {"a": a_fit, "b": b_fit}
    except Exception:   # noqa: BLE001
        return {"a": a0, "b": b0}


def predict(X: np.ndarray, a: float, b: float) -> np.ndarray:
    """BN77 integrated power-law: Q(t) = [Q_0^(1-b) + a*(b-1)*t]^(1/(1-b)).

    Parameters
    ----------
    X : (n, 2) array — columns [t, Q_0].
    a : recession amplitude coefficient (> 0).
    b : recession nonlinearity exponent (> 1).

    Returns
    -------
    (n,) array of predicted discharge Q_t.
    """
    t   = np.asarray(X[:, 0], dtype=float)
    Q_0 = np.asarray(X[:, 1], dtype=float)
    return _Q_t(t, Q_0, a, b)
