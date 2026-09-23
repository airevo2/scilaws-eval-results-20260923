"""Non-rectangular hyperbola (NRH) light-response curve — net assimilation.

Lobo, F. de A., de Barros, M. P., Dalmagro, H. J., Dalmolin, A. C.,
Pereira, W. E., de Souza, E. C., Vourlitis, G. L., & Rodriguez Ortiz,
C. E. (2013). Fitting net photosynthesis models to PSII electron-transport
curves. Photosynthetica 51(3): 445-456.
DOI:10.1007/s11099-013-0045-y

The non-rectangular hyperbola (NRH) for the net photosynthetic
light-response curve is given by Lobo 2013 Eq. 6 (PDF p. 3), itself
attributed to Prioul & Chartier (1977) and Marshall & Biscoe (1980):

    PN = [phi*I + Pgmax - sqrt((phi*I + Pgmax)^2 - 4*theta*phi*I*Pgmax)]
          / (2*theta) - RD

where
  I      = incident PPFD (Qin in Schmiege dataset), umol m-2 s-1
  PN     = net CO2 assimilation rate (A in Schmiege dataset), umol CO2 m-2 s-1
  phi    = apparent quantum yield at I=0, umol(CO2)/umol(photon)
  Pgmax  = asymptotic maximum gross rate (I->inf), umol(CO2) m-2 s-1
  theta  = convexity of the PN/I curve, dimensionless (0-1)
  RD     = dark respiration rate, umol(CO2) m-2 s-1

Boundary check (Lobo 2013 §Introduction, Eq. 7 exclusion criterion):
  At I = 0 -> PN = [Pgmax - sqrt(Pgmax^2)]/(2*theta) - RD = 0 - RD = -RD.
The NRH satisfies PN(I=0) = -RD exactly.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. The NRH form is the scientific claim. phi, Pgmax, theta, and RD
are per-plant fit parameters. Lobo 2013 documents typical ranges
(phi: 0.0266-0.0800; theta: 0.70-0.99; Pgmax: 42-75 umol m-2 s-1;
RD ~10% of PNmax) but publishes no universal numerical values — all are
LOCAL_FITTABLE.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
The factor 4 in 4*theta*phi*I*Pgmax and the factor 2 in the denominator
2*theta are structural algebraic constants of the NRH quadratic form
(roots of theta*PN^2 - (phi*I + Pgmax)*PN + phi*I*Pgmax = 0), not
tunable empirical constants. They appear as integer literals below.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- phi   : apparent quantum yield at zero irradiance (umol(CO2)/umol(photon)).
          Lobo 2013 reports 0.0266-0.0800; theoretical maximum 0.125.
- Pgmax : asymptotic gross photosynthetic rate (umol CO2 m-2 s-1). > 0.
- theta : convexity parameter, in (0, 1]. Typical: 0.70-0.99 for C3 leaves.
- RD    : dark respiration rate (umol CO2 m-2 s-1). > 0.

init = None on all: fit() builds data-derived starts.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["Qin"]
PAPER_REF = "summary_formula_lobo_2013.md"
EQUATION_LOC = (
    "Lobo 2013 Eq. 6, PDF p. 3 — non-rectangular hyperbola (NRH) "
    "PN/I light-response model (Prioul & Chartier 1977; Marshall & Biscoe 1980)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "phi":   {"init": None},
    "Pgmax": {"init": None},
    "theta": {"init": None},
    "RD":    {"init": None},
}


def _nrh(I, phi, Pgmax, theta, RD):
    """Non-rectangular hyperbola net assimilation (Lobo 2013 Eq. 6)."""
    inside = (phi * I + Pgmax) ** 2 - 4.0 * theta * phi * I * Pgmax
    # Clip to avoid sqrt of tiny negatives from floating-point at I=0
    inside = np.maximum(inside, 0.0)
    return (phi * I + Pgmax - np.sqrt(inside)) / (2.0 * theta) - RD


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the NRH model.

    Data-derived initial estimates:
      RD0    = -min(A, 0) or 1.0 if all positive (dark respiration ~|A(I=0)|)
      Pgmax0 = max(A) + RD0 (rough asymptote from observed peak)
      phi0   = 0.04 (mid-range quantum yield from Lobo 2013)
      theta0 = 0.85 (mid-range convexity from Lobo 2013 Table 1)
    """
    I = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    # Estimate dark respiration from near-zero irradiance points if available
    low_light_mask = I < 10.0
    if low_light_mask.any():
        RD0 = float(np.maximum(-np.mean(y[low_light_mask]), 0.5))
    else:
        RD0 = float(max(-np.min(y), 0.5))

    Pgmax0 = float(max(np.max(y) + RD0, 1.0))
    phi0   = 0.04    # mid-range, Lobo 2013 reports 0.0266-0.0800
    theta0 = 0.85    # mid-range, Lobo 2013 reports 0.70-0.99

    # Fit bounds: biologically motivated
    #             phi          Pgmax       theta         RD
    lo = [1e-5,        0.1,    1e-3,      1e-3]
    hi = [0.125,     300.0,    1.0 - 1e-6, 50.0]

    p0 = [
        min(max(phi0,   lo[0]), hi[0]),
        min(max(Pgmax0, lo[1]), hi[1]),
        min(max(theta0, lo[2]), hi[2]),
        min(max(RD0,    lo[3]), hi[3]),
    ]

    def residual(p):
        return _nrh(I, p[0], p[1], p[2], p[3]) - y

    try:
        sol = least_squares(residual, p0, bounds=(lo, hi),
                            method="trf", max_nfev=6000)
        phi, Pgmax, theta, RD = sol.x
        if not np.all(np.isfinite([phi, Pgmax, theta, RD])):
            raise RuntimeError("non-finite")
        return {"phi": float(phi), "Pgmax": float(Pgmax),
                "theta": float(theta), "RD": float(RD)}
    except Exception:                                # noqa: BLE001
        return {"phi": float(p0[0]), "Pgmax": float(p0[1]),
                "theta": float(p0[2]), "RD": float(p0[3])}


def predict(X: np.ndarray, phi: float, Pgmax: float,
            theta: float, RD: float) -> np.ndarray:
    """Non-rectangular hyperbola net assimilation.

    X: (n, 1) — column [Qin] in umol m-2 s-1.
    Returns PN (= A_n) in umol(CO2) m-2 s-1.
    """
    I = np.asarray(X[:, 0], dtype=float)
    return _nrh(I, phi, Pgmax, theta, RD)
