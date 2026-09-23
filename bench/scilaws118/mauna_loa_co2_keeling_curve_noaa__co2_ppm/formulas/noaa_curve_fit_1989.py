"""NOAA GML curve fit (Thoning-Tans-Komhyr 1989) — strongest rung.

Thoning, Tans & Komhyr (1989), "Atmospheric carbon dioxide at Mauna Loa
Observatory: 2. Analysis of the NOAA GMCC data, 1974-1985", JGR 94 (D6)
8549-8565 (paywalled at AGU; functional form replicated from the open
NOAA GML curve-fit page https://gml.noaa.gov/ccgg/mbl/crvfit/crvfit.html
and Keeling et al. 2001 §3, PDF on disk).  The canonical NOAA GML
analysis pipeline (CCGCRV) fits the Mauna Loa monthly mean series with
a low-order polynomial trend (k=3 → quadratic) plus a finite sum of
yearly harmonics (nh=4 → 8 sine/cosine coefficients):

    co2(t) = A + B*(t - T0) + C*(t - T0)^2
           + Σ_{m=1..4} [α_m sin(2π m t) + β_m cos(2π m t)].

The 11 free coefficients (A, B, C, α₁, β₁, α₂, β₂, α₃, β₃, α₄, β₄) are
pre-fit on the v2 train calibration window (1958-2019, 742 rows) and
frozen as LAW_CONSTANTS.  The first-harmonic amplitude
√(α₁² + β₁²) ≈ 2.83 ppm is consistent with the published Mauna Loa
seasonal cycle of ~3 ppm peak-to-peak (Keeling 2001 Fig. 5).

NOAA's full pipeline additionally applies a low-pass FFT residual
filter (80-day / 667-day cutoffs) to extract synoptic and inter-annual
components, but the FFT filter is not part of the analytic closed
form and is omitted here by design.

LAW_CONSTANTS — frozen, pre-fit on v2 train calibration window
--------------------------------------------------------------
- A      = 337.5798  ppm at t = T0
- B      =   1.3407  ppm / yr at t = T0
- C      =   0.012808 ppm / yr^2
- ALPHA1 =   2.6433  ppm (annual sin)
- BETA1  =  -0.9973  ppm (annual cos)
- ALPHA2 =  -0.4446  ppm (semi-annual sin)
- BETA2  =   0.6562  ppm (semi-annual cos)
- ALPHA3 =  -0.0751  ppm (4-monthly sin)
- BETA3  =   0.0430  ppm (4-monthly cos)
- ALPHA4 =   0.0480  ppm (3-monthly sin)
- BETA4  =  -0.0472  ppm (3-monthly cos)

OTHER_CONSTANTS — structural epoch
----------------------------------
- T0 = 1980.0   yr (fixed reference epoch)

LOCAL_FITTABLE
--------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["year_decimal"]
PAPER_REF = "summary_keeling_curve.md"
EQUATION_LOC = (
    "Thoning-Tans-Komhyr 1989 NOAA GML CCGCRV pipeline (k=3 polynomial + "
    "nh=4 harmonics); functional form documented in Keeling 2001 §3 (PDF on "
    "disk) and the NOAA GML curve-fit page. 11 coefficients pre-fit on v2 "
    "train calibration window."
)

LAW_CONSTANTS = {
    "A":      337.5798,
    "B":        1.3407,
    "C":        0.012808,
    "ALPHA1":   2.6433,
    "BETA1":   -0.9973,
    "ALPHA2":  -0.4446,
    "BETA2":    0.6562,
    "ALPHA3":  -0.0751,
    "BETA3":    0.0430,
    "ALPHA4":   0.0480,
    "BETA4":   -0.0472,
}
OTHER_CONSTANTS = {
    "T0": 1980.0,
}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, A: float = 337.5798, B: float = 1.3407,
            C: float = 0.012808,
            ALPHA1: float = 2.6433, BETA1: float = -0.9973,
            ALPHA2: float = -0.4446, BETA2: float = 0.6562,
            ALPHA3: float = -0.0751, BETA3: float = 0.0430,
            ALPHA4: float = 0.0480, BETA4: float = -0.0472) -> np.ndarray:
    """co2(t) = quadratic trend + 4 yearly harmonics (NOAA GML CCGCRV form)."""
    t = np.asarray(X[:, 0], dtype=float)
    dt = t - OTHER_CONSTANTS["T0"]
    y = A + B * dt + C * dt * dt
    two_pi_t = 2.0 * np.pi * t
    y = y + ALPHA1 * np.sin(    two_pi_t) + BETA1 * np.cos(    two_pi_t)
    y = y + ALPHA2 * np.sin(2 * two_pi_t) + BETA2 * np.cos(2 * two_pi_t)
    y = y + ALPHA3 * np.sin(3 * two_pi_t) + BETA3 * np.cos(3 * two_pi_t)
    y = y + ALPHA4 * np.sin(4 * two_pi_t) + BETA4 * np.cos(4 * two_pi_t)
    return y
