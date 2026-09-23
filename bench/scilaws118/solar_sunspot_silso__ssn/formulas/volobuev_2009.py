"""Volobuev (2009) cycle-shape formula — per-cycle fit.

Primary citation:
    Volobuev, D. M. (2009). "The Shape of the Sunspot Activity Cycle."
    Solar Physics 258:319-330.
    https://doi.org/10.1007/s11207-009-9429-3
    (paywalled; formula form verified via secondary source below)

Verbatim-form source (secondary citation):
    Penza, V. et al. (2024). "Reconstruction of the Total Solar Irradiance
    during the last Millennium." arXiv:2409.12648v1, Section 4, Equation 3.
    The Penza et al. paper states: "We utilize the functional form presented
    in Volobuev (2009)" and gives Eq. 3 verbatim.

Citation-chain transparency note:
    The Volobuev (2009) paper itself is paywalled. The formula implemented
    here is taken verbatim from Penza et al. (2024) arXiv:2409.12648v1 Eq. 3,
    which explicitly attributes it to Volobuev (2009). The W121 triage
    (2026-05-26_volobuev_2009_oa_hunt_W121.md) confirms this sourcing. Both
    references are cited to document the chain transparently.

Formula (Penza 2024 Eq. 3, attributed to Volobuev 2009):

    x_k(t) = ((t - T0_k) / Ts_k)^2 * exp(-((t - T0_k) / Td_k)^2)

    valid for: T0_k < t < T0_k + tau_k

where:
    T0_k  = start year of cycle k (decimal year)
    Ts_k  = rise timescale (yr) — controls slope of the rising flank
    Td_k  = decay/shape timescale (yr) — controls peak position and width
    tau_k = cycle duration (yr; not fitted here — validity condition only)

Peak properties:
    Peak occurs at t = T0_k + Td_k.
    Peak value = (Td_k / Ts_k)^2 * exp(-1) ≈ 0.3679 * (Td_k / Ts_k)^2.

LAW_CONSTANTS = {} note (data_spec §5.2 form-as-claim):
    The Volobuev formula has NO frozen scalar constants in the shape equation.
    There is no asymmetry constant analogous to HWR's c = 0.8. The structural
    exponent 2 is an inline numeral. The form itself (with Ts and Td as free
    per-cycle parameters) is the scientific claim. LAW_CONSTANTS = {} is
    explicitly permitted by data_spec §5.2 for form-as-claim baselines
    (citing Atkinson Pareto, Peleg Chick-Watson, Benzekry Gompertz as
    precedent examples).

    NOTE: Penza et al. (2024) also fit the formula with Waldmeier-effect
    coefficients s1=0.02, s2=3.14 to link Ts/Td to cycle amplitude. These
    are Penza's empirical fit values, NOT Volobuev's primary-paper-frozen
    constants, so they are NOT included as LAW_CONSTANTS here.

Unit convention:
    Decimal years throughout. T0, Ts, Td are all in units of decimal years.
    X[:, 0] = t_year (decimal year).

Validity range:
    Volobuev's formula is defined for T0_k < t < T0_k + tau_k. The
    released CSV rows are post-minimum (t > T0 for any reasonable T0 fit),
    so this condition is satisfied in practice. At t = T0, the formula
    gives 0 (valid boundary). For t < T0, the formula gives positive values
    (since even power), but the physical domain is t >= T0.

Type II — per-cycle fit:
    Each solar cycle (cluster) has its own (T0, Ts, Td). The harness calls
    fit(X_fit, y_fit) on test_fit rows, then predict(X_test, T0, Ts, Td)
    on test_test rows.

Fit notes:
    Multi-start Nelder-Mead minimises MSE on (y_fit, predict(X_fit)).
    Init grid (3 values per param, respecting max_init_size_per_param=3):
        T0: [t_min - 0.5, t_min - 0.333, t_min - 0.083] (yr before min obs)
        Ts: [0.10, 0.20, 0.30]  (rise timescales in yr)
        Td: [3.0, 4.0, 5.0]    (decay timescales in yr)
    Total = 27 multi-start combinations.
    Peak value at (Ts=0.15, Td=3.0) = (3.0/0.15)^2 * exp(-1) ≈ 147.15 SSN,
    consistent with typical modern cycle peak on ISN v2 scale.
"""

import numpy as np
from scipy.optimize import minimize

USED_INPUTS = ["t_year"]
PAPER_REF = "volobuev_2009"
EQUATION_LOC = "Volobuev (2009) Sol. Phys. 258:319 Eq. (form); verbatim from Penza et al. (2024) arXiv:2409.12648v1 Eq. 3"

# --- Law constants (form-as-claim per data_spec §5.2) ---
# Volobuev's formula has no frozen scalar constants in the shape equation.
# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {}

# === OTHER_CONSTANTS — none (empty) ===
OTHER_CONSTANTS = {}   # structural exponent 2 is inline numeral

# Per-cycle fit parameters — TYPE II
# Init grid: 3 values per param (max_init_size_per_param = 3)
LOCAL_FITTABLE = {
    "T0": {"init": [None, None, None]},     # cycle start (yr); set in fit()
    "Ts": {"init": [0.10, 0.20, 0.30]},     # rise timescale (yr)
    "Td": {"init": [3.0, 4.0, 5.0]},        # decay/shape timescale (yr)
}


def _volobuev(t, T0, Ts, Td):
    """Core Volobuev scalar formula; numerically safe throughout.

    x(t) = ((t - T0) / Ts)^2 * exp(-((t - T0) / Td)^2)
    """
    u_s = (t - T0) / Ts
    u_d = (t - T0) / Td
    with np.errstate(over="ignore", invalid="ignore"):
        val = (u_s ** 2) * np.exp(-(u_d ** 2))
    # Replace NaN/Inf (e.g. Ts or Td very small) with 0
    val = np.where(np.isfinite(val), val, 0.0)
    return val


def fit(X_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Fit (T0, Ts, Td) to one cycle's fit-window rows.

    Uses multi-start Nelder-Mead to avoid local minima. T0 init values are
    built from the data: slightly before the earliest observed point in the
    cluster. Ts and Td are initialized from the precomputed grid.
    Returns dict with keys T0, Ts, Td (matching LOCAL_FITTABLE).
    """
    t = np.asarray(X_fit[:, 0], dtype=float)
    y = np.asarray(y_fit, dtype=float)

    t_min = float(t.min())
    # Three T0 inits: 6 months, 4 months, 1 month before earliest observed pt
    T0_inits = [t_min - 0.5, t_min - 0.333, t_min - 0.083]
    Ts_inits = [0.10, 0.20, 0.30]
    Td_inits = [3.0, 4.0, 5.0]

    def objective(params):
        T0_, Ts_, Td_ = params
        if Ts_ <= 0.0 or Td_ <= 0.0:
            return 1e12
        pred = _volobuev(t, T0_, Ts_, Td_)
        return float(np.mean((pred - y) ** 2))

    best_mse = np.inf
    best_params = (t_min - 0.333, 0.20, 4.0)

    for T0_0 in T0_inits:
        for Ts0 in Ts_inits:
            for Td0 in Td_inits:
                try:
                    res = minimize(
                        objective,
                        x0=[T0_0, Ts0, Td0],
                        method="Nelder-Mead",
                        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 5000},
                    )
                    if res.fun < best_mse:
                        best_mse = res.fun
                        best_params = tuple(res.x)
                except Exception:
                    pass

    T0_fit, Ts_fit, Td_fit = best_params
    # Guard against degenerate fits
    if Ts_fit <= 0.0:
        Ts_fit = 0.20
    if Td_fit <= 0.0:
        Td_fit = 4.0
    return {"T0": float(T0_fit), "Ts": float(Ts_fit), "Td": float(Td_fit)}


def predict(X: np.ndarray, T0: float, Ts: float, Td: float) -> np.ndarray:
    """Volobuev (2009) cycle-shape formula (Penza 2024 arXiv:2409.12648v1 Eq. 3).

    x(t) = ((t - T0) / Ts)^2 * exp(-((t - T0) / Td)^2)

    X[:, 0] = t_year (decimal year). Returns predicted SSN array.
    params: T0 (cycle start year, yr), Ts (rise timescale, yr),
            Td (decay/shape timescale, yr).
    No LAW_CONSTANTS (form-as-claim per data_spec §5.2).
    """
    t = np.asarray(X[:, 0], dtype=float)
    return _volobuev(t, T0, Ts, Td)
