"""Mitscherlich exponential-saturation light-response model — net assimilation.

Peek, M. S., Russek-Cohen, E., Wait, D. A., & Forseth, I. N. (2002).
Physiological response curve analysis using nonlinear mixed models.
Oecologia 132(2): 175-180.
DOI:10.1007/s00442-002-0954-0

The Mitscherlich exponential-saturation model for the photosynthetic
light-response curve is given by Peek 2002 Eq. 2 (PDF p. 2), following
Potvin et al. (1990):

    A = Amax * [1 - exp(-Aqe * (PPF - LCP))]

where
  PPF  = incident photosynthetic photon flux (= Qin in Schmiege dataset),
         umol m-2 s-1
  A    = net photosynthesis (= A_n in Schmiege dataset), umol(CO2) m-2 s-1
  Amax = light-saturated asymptote of photosynthesis, umol(CO2) m-2 s-1
  Aqe  = apparent quantum yield (initial slope at low light),
         umol(CO2)/umol(photon)
  LCP  = light compensation point (A = 0 x-intercept), umol m-2 s-1

Boundary check: at PPF = LCP -> A = Amax * (1 - e^0) = 0. The model
places the zero-crossing exactly at PPF = LCP by construction, satisfying
the biological constraint PN(I = LCP) = 0.

For PPF < LCP, the exponent (-Aqe*(PPF-LCP)) > 0, so exp > 1, and A < 0,
correctly representing net CO2 loss under sub-compensation irradiance.

Note on SAS rescaling (Peek 2002 Appendix, PDF p. 6): the SAS code uses
Aqe scaled by 0.0001 as an implementation artefact to avoid numerical
convergence issues in PROC NLMIXED. Here Aqe is stored in its natural units
(umol(CO2)/umol(photon)); no rescaling is applied.

Equivalence: this model is equivalent to Lobo 2013 Eq. 8 (exponential)
when Icomp = LCP; Peek 2002 Eq. 2 parameterises the x-intercept explicitly
via LCP, which is ecophysiologically interpretable and avoids the
compensation-point mislocation problem flagged by Lobo 2013.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. Amax, Aqe, and LCP are per-plant fit parameters. Peek 2002 Table 2
reports per-treatment means (Amax: 10-42, Aqe: 0.0011-0.0059,
LCP: 10-41 umol m-2 s-1) but no universal numerical value — all are
LOCAL_FITTABLE.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
The exponential base e is a universal mathematical constant. The literal
-1 in (1 - exp(...)) is the structural coefficient of the saturation
approach, not a tunable empirical constant.

LOCAL_FITTABLE — per-cluster, fitted by fit() via bounded nonlinear LS
-----------------------------------------------------------------------
- Amax : light-saturated net photosynthesis (umol CO2 m-2 s-1). > 0.
         Peek 2002 Table 2: range 10-42 across treatment groups.
- Aqe  : apparent quantum yield (umol(CO2)/umol(photon)). > 0.
         Peek 2002 Table 2: 0.0011-0.0059 across treatment groups.
- LCP  : light compensation point (umol m-2 s-1). >= 0.
         Peek 2002 Table 2: 10-41 across treatment groups.

init = None on all: fit() builds data-derived starts.
"""

import numpy as np
from scipy.optimize import least_squares

USED_INPUTS = ["Qin"]
PAPER_REF = "summary_formula_peek_2002.md"
EQUATION_LOC = (
    "Peek 2002 Eq. 2, PDF p. 2 — Mitscherlich exponential-saturation "
    "A = Amax * [1 - exp(-Aqe * (PPF - LCP))]; after Potvin et al. (1990)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "Amax": {"init": None},
    "Aqe":  {"init": None},
    "LCP":  {"init": None},
}


def _mitscherlich(PPF, Amax, Aqe, LCP):
    """Mitscherlich exponential-saturation net photosynthesis (Peek 2002 Eq. 2)."""
    return Amax * (1.0 - np.exp(-Aqe * (PPF - LCP)))


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Bounded nonlinear least-squares fit of the Mitscherlich model.

    Data-derived initial estimates:
      Amax0 = max(A, 1.0)  (observed peak approximates saturated rate)
      Aqe0  = 0.03         (mid-range; Peek 2002 range 0.001-0.006;
                            Schmiege conifer data may have higher phi)
      LCP0  = I at zero-crossing, estimated by linear interpolation near A=0,
              or 30 umol m-2 s-1 if data do not span the compensation point.
    """
    PPF = np.asarray(X_fit[:, 0], dtype=float)
    y   = np.asarray(y_fit, dtype=float)

    Amax0 = float(max(np.max(y), 1.0))

    # Estimate LCP: zero-crossing of A vs PPF
    sorted_idx = np.argsort(PPF)
    PPF_s, y_s = PPF[sorted_idx], y[sorted_idx]
    LCP0 = 30.0  # default fallback (umol m-2 s-1)
    for i in range(len(y_s) - 1):
        if y_s[i] <= 0.0 <= y_s[i + 1] and y_s[i + 1] > y_s[i]:
            # linear interpolation between bracketing points
            frac = -y_s[i] / (y_s[i + 1] - y_s[i])
            LCP0 = float(PPF_s[i] + frac * (PPF_s[i + 1] - PPF_s[i]))
            break

    Aqe0 = 0.03  # broad mid-point; Schmiege conifer data

    # Fit bounds
    #              Amax       Aqe          LCP
    lo = [0.1,    1e-5,       0.0]
    hi = [500.0,  0.5,        500.0]

    p0 = [
        min(max(Amax0, lo[0]), hi[0]),
        min(max(Aqe0,  lo[1]), hi[1]),
        min(max(LCP0,  lo[2]), hi[2]),
    ]

    def residual(p):
        return _mitscherlich(PPF, p[0], p[1], p[2]) - y

    try:
        sol = least_squares(residual, p0, bounds=(lo, hi),
                            method="trf", max_nfev=4000)
        Amax, Aqe, LCP = sol.x
        if not np.all(np.isfinite([Amax, Aqe, LCP])):
            raise RuntimeError("non-finite")
        return {"Amax": float(Amax), "Aqe": float(Aqe), "LCP": float(LCP)}
    except Exception:                               # noqa: BLE001
        return {"Amax": float(p0[0]), "Aqe": float(p0[1]), "LCP": float(p0[2])}


def predict(X: np.ndarray, Amax: float, Aqe: float, LCP: float) -> np.ndarray:
    """Mitscherlich exponential-saturation net photosynthesis.

    X: (n, 1) — column [Qin] in umol m-2 s-1.
    Returns A_n (net CO2 assimilation) in umol(CO2) m-2 s-1.
    """
    PPF = np.asarray(X[:, 0], dtype=float)
    return _mitscherlich(PPF, Amax, Aqe, LCP)
