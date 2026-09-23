"""Xu-Randall (1996) cloud cover scheme with DYAMOND-fit coefficients.

The Xu-Randall functional form was introduced in:
  Xu, K.-M., & Randall, D. A. (1996). "A semiempirical cloudiness
  parameterization for use in climate models." J. Atmos. Sci., 53(21),
  3084-3102.

The specific coefficients used here (α, β tuned to DYAMOND data) are
from:
  Grundner et al. (2024). "Data-Driven Equation Discovery of a Cloud
  Cover Parameterization." JAMES. DOI 10.1029/2023MS003763.
  Eq. 4, PDF p. 7; fitted values at PDF p. 13 (Sec 5, Pareto frontier).

Formula (Eq. 4, Grundner 2024 PDF p. 7):

    C_XR = min{ RH^alpha * (1 - exp(-beta * (q_c + q_i))), 1 }

    C = 0      if q_c + q_i == 0   [PC2 constraint, by design]
        C_XR   otherwise

LAW_CONSTANTS — best DYAMOND-fit values (PDF p. 13):
  alpha = 0.9     — exponent on RH; PDF p. 13 "{α, β} = {0.9, 9·10^5}"
  beta  = 9.0e5   — condensate factor [kg/kg]^-1; same citation

Note on coefficient interpretation: with large beta (9e5), the condensate
term saturates to 1 for any positive condensate value, yielding C ≈ RH^0.9
(confirmed by the text at PDF p. 13: "cloud cover is always approximately
equal to relative humidity (i.e., C ≈ RH^0.9) when cloud condensates are
present"). Thus alpha = 0.9 is the RH exponent and beta = 9e5 is the
condensate sensitivity.

Note on alpha/beta naming: Grundner 2024 prints Eq. 4 as
RH^beta * (1 - exp(-alpha*(q_c + q_i))) with {alpha, beta} = {0.9, 9e5}, yet
the prose (PDF p. 13) states the scheme reduces to C ≈ RH^0.9 when condensate
is present. The only physically-consistent reading — matching the prose — is
RH exponent = 0.9 and condensate factor = 9e5. This module names the 0.9
exponent `alpha` and the 9e5 factor `beta`, so predict() yields exactly
C ≈ RH^0.9, matching the paper's stated behaviour.

OTHER_CONSTANTS — none (the formula is dimensionless).

Type designation: Type I — two global parameters (α, β) from a single DYAMOND
fit. No per-cluster refit. LOCAL_FITTABLE = {}.

Column mapping:
  RH    -> col RH   (dimensionless)
  q_c   -> col q_c  (kg/kg)
  q_i   -> col q_i  (kg/kg)

Note: The original Xu & Randall (1996) paper (J. Atmos. Sci. 53, 3084-3102)
IS shipped at reference/xu_randall_1996.pdf for scheme provenance, but the AMS
PDF has no extractable text layer (xu_randall_1996.txt holds only the access
watermark). The equation form and the {alpha, beta} = {0.9, 9e5} coefficients
used here are therefore grounded in the Grundner 2024 PDF (Eq. 4, p. 7; fitted
values p. 13), where they were verified. PAPER_REF points to the Grundner 2024
summary. This baseline is a classical-scheme comparison point.

Caveats: Coefficients were calibrated on DYAMOND global SRM data, not
NARVAL. OOD bias is expected on the test set, consistent with the other
baselines in this bank.
"""

import numpy as np

USED_INPUTS = ["q_c", "q_i", "RH"]
PAPER_REF = "summary_formula_grundner_2024.md"
EQUATION_LOC = "Xu-Randall Eq. 4, Grundner 2024 PDF p. 7; fitted values PDF p. 13"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "alpha": 0.9,     # RH exponent; Grundner 2024 PDF p. 13
    "beta":  9.0e5,   # condensate factor [1/(kg/kg)]; Grundner 2024 PDF p. 13
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}   # formula is dimensionless; no structural constants

LOCAL_FITTABLE = {}    # Type I — no per-cluster parameters


def predict(X: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Cloud cover fraction per Xu-Randall scheme (DYAMOND-fit coefficients).

    X shape: (n, 3) — columns in USED_INPUTS order: q_c, q_i, RH.
    Returns predicted cloud area fraction in [0, 1].
    """
    q_c = np.asarray(X[:, 0], dtype=float)
    q_i = np.asarray(X[:, 1], dtype=float)
    RH  = np.asarray(X[:, 2], dtype=float)

    # RH clamped to [0,1] for stability with large alpha exponent
    RH_safe = np.clip(RH, 0.0, 1.0)

    condensate = q_c + q_i
    C_xr = RH_safe ** alpha * (1.0 - np.exp(-beta * condensate))
    C_xr = np.clip(C_xr, 0.0, 1.0)
    C = np.where(condensate == 0.0, 0.0, C_xr)
    return C
