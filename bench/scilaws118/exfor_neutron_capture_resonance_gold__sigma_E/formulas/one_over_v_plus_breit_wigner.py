"""1/v background + single-level Breit-Wigner resonance — best rung.

The full low-energy Au-197(n,gamma) cross section in the E < 50 eV window
is the 1/v s-wave background plus the single-level Breit-Wigner
resonance at the dominant 4.9 eV s-wave level (Mughabghab 'Atlas of
Neutron Resonances' 6th ed. 2018; the next resonances at 60.3/78.7/107
eV are above this window):

    sigma(E) = SV / sqrt(E)
               + SR * (G/2)^2 / [ (E - E_R)^2 + (G/2)^2 ],

a 1/v term (SV the scale) plus a Lorentzian resonance pole (peak SR at
E = E_R, full width at half maximum G).  The model is evaluated in
linear barns and log10-transformed to the target.

This captures both regimes — the thermal 1/v decline AND the 4.9 eV
resonance peak — lifting the test r^2 to ~0.90 (rmse ~ 0.33), far above
the pure-1/v rung (r^2 ~ 0.37).  The fitted resonance energy
E_R = 4.86 eV reproduces the documented 4.906 eV s-wave resonance, the
width G ~ 0.14 eV matches the documented Gamma_total ~ 0.137 eV, and the
peak amplitude SR ~ 30,000 b matches the documented ~33,000 b peak.

The Lorentzian resonance pole is the non-trivial structure an SR system
must discover beyond the smooth 1/v background — the canonical
compound-nucleus resonance form.

LAW_CONSTANTS — frozen, pre-fit on v2 train
-------------------------------------------
- SV  = 3.117     (1/v background scale, barn*sqrt(eV))
- SR  = 29668     (resonance peak cross section, barn; documented ~33,000)
- E_R = 4.8634    (resonance energy, eV; documented 4.906)
- G   = 0.1437    (resonance full width, eV; documented Gamma_total ~0.137)

OTHER_CONSTANTS / LOCAL_FITTABLE
--------------------------------
None (Type I).
"""

import numpy as np

USED_INPUTS = ["E_eV"]
PAPER_REF = "summary_exfor_au197.md"
EQUATION_LOC = (
    "1/v + single-level Breit-Wigner: sigma = SV/sqrt(E) + "
    "SR*(G/2)^2/((E-E_R)^2+(G/2)^2) (Bohr-Wheeler 1/v + Mughabghab 2018 "
    "4.906 eV s-wave resonance; summary_exfor_au197.md (i),(ii)).  "
    "Constants pre-fit on v2 train; target is log10(sigma)."
)

LAW_CONSTANTS = {
    "SV":  3.117,
    "SR":  29668.4,
    "E_R": 4.8634,
    "G":   0.1437,
}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}


def predict(X: np.ndarray, SV: float = 3.117, SR: float = 29668.4,
            E_R: float = 4.8634, G: float = 0.1437) -> np.ndarray:
    """log10 of [ SV/sqrt(E) + SR*(G/2)^2/((E-E_R)^2+(G/2)^2) ]; 1/v + Breit-Wigner."""
    E = np.asarray(X[:, 0], dtype=float)
    sigma = SV / np.sqrt(E) + SR * (G / 2.0) ** 2 / ((E - E_R) ** 2 + (G / 2.0) ** 2)
    return np.log10(np.clip(sigma, 1e-9, None))
