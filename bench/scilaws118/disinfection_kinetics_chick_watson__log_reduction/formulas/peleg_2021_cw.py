"""Peleg (2021) static Chick-Watson model parameterised on CT dose — LRV = k * CT.

Peleg, M. (2021). Modeling the dynamic kinetics of microbial disinfection
with dissipating chemical agents—a theoretical investigation. Applied
Microbiology and Biotechnology 105(2):539-549.
DOI:10.1007/s00253-020-11042-8. PMC7780086.

The static Chick-Watson (CW) model, Eq. (1) of Peleg (2021) (PDF p. 2,
journal p. 540):

    Log_10[S(t)] = -k * C^n * t

and thus the log reduction value (LRV = -Log_10[S]):

    LRV = k * C^n * t

For the special case n = 1 (the original Chick (1908) first-order law):

    LRV = k * C * t = k * CT

where CT = C * t is the disinfection dose (mg/L * min).

The Mofidi (2025) database reports CT as a single measured quantity; most
rows lack C and t individually. This formulation uses CT as the sole
input, making the benchmark compatible with all 4,004 qualifying rows.
The n = 1 (linear-in-C) case is the most common engineering assumption
(Gyurek and Finch 1998; Haas and Kara 1984, cited by Peleg 2021 PDF p. 2)
and is the original Chick law (Chick 1908 PDF p. 5, journal p. 95).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
None. Both the formula structure (Chick-Watson, Eq. 1 Peleg 2021) and the
dose-linear special case (n = 1) are well-established conventions; k is
entirely specimen/organism/disinfectant specific (Peleg 2021 PDF p. 2:
"k and n are characteristic to the targeted microbe and the particular
disinfectant type"). No universal numeric value of k is published.

OTHER_CONSTANTS — universal / structural factors
------------------------------------------------
(empty.) The CT parameterisation and log base 10 are structural; the
literal 1 is an inline numeral in the body (n = 1 absorbed into CT).

LOCAL_FITTABLE — per-cluster, fitted by fit() via OLS through origin
--------------------------------------------------------------------
- k : inactivation rate constant per CT unit ((mg/L*min)^-1, > 0).
      Refit per (organism x disinfectant) cluster from (CT, LRV) pairs.
      Closed-form OLS through origin: k = dot(CT, LRV) / dot(CT, CT).
      init = None: fit() uses closed-form estimate.

Type II: one (k) per (disinfectant x organism) cluster. No global fit.
Column mapping: CT (mg/L*min) -> CT column; LRV -> log_reduction target.
"""

import numpy as np

USED_INPUTS = ["CT"]
PAPER_REF = "summary_formula_peleg_2021.md"
EQUATION_LOC = (
    "Peleg (2021) Eq. (1), PDF p. 2, journal p. 540 — "
    "LRV = k * C^n * t; n=1 special case gives LRV = k * CT. "
    "Chick (1908) PDF p. 5, journal p. 95 (original form)."
)

LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {
    "k": {"init": None},
}


def _lrv(CT, k):
    return k * CT


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """OLS through the origin: k = dot(CT, LRV) / dot(CT, CT).

    X_fit: (n, 1) — column [CT].
    y_fit: (n,)  — LRV.
    Closed-form (no optimiser). Positive k is enforced by clamping.
    """
    CT = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)
    denom = float(np.dot(CT, CT))
    k0 = float(np.dot(CT, y)) / denom if denom > 0 else 1e-4
    k0 = max(k0, 1e-9)
    return {"k": k0}


def predict(X: np.ndarray, k: float) -> np.ndarray:
    """LRV = k * CT.

    X: (n, 1) — column [CT].
    """
    CT = np.asarray(X[:, 0], dtype=float)
    return _lrv(CT, k)
