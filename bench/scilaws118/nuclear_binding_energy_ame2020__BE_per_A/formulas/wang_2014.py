"""Wang-Liu-Zhao (2014) WS4 macroscopic / liquid-drop part.

Wang, Liu, Zhao, Phys. Lett. B 734 (2014) 215-219 — the WS4
Weizsäcker-Skyrme mass formula. The full WS4 model adds a Strutinsky
shell correction computed from a Woods-Saxon potential (the WSBETA code,
not closed-form). This module implements the analytically tractable
liquid-drop component E_LD (Eq. 1, PDF p. 5):

    E_LD(A, Z) = a_v * A
               + a_s * A**(2/3)
               + a_c * Z**2 / A**(1/3) * (1 - 0.76 * Z**(-2/3))
               + a_sym(A, I) * I**2 * A * f_s
               + a_pair * A**(-1/3) * delta_np
               + c_w * |I|                       (Wigner term)

with I = (N-Z)/A and

    a_sym(A,I) = c_sym * (1 - kappa/A**(1/3) + xi*(2-|I|)/(2+|I|*A))
    f_s        = 1 + kappa_s * eps * A**(1/3),   eps = (I-I0)**2 - I**4
    I0         = 0.4 * A / (A + 200)

and the four-case pairing staggering delta_np (PDF pp. 5-6). The
benchmark target is BE/A = -E_LD / A (WS4 has E_LD < 0 for bound nuclei).

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
The nine LD coefficients, Wang 2014 Table I, PDF p. 7:

    a_v = -15.5181  a_s = 17.4090  a_c = 0.7092
    c_sym = 30.1594  kappa = 1.5189  xi = 1.2230
    a_pair = -5.8166  c_w = 0.8705  kappa_s = 0.1536

OTHER_CONSTANTS — structural operator-definition constants
----------------------------------------------------------
Fixed constants from the Eq. 1 prose — not among the nine fitted
coefficients, declared so every constant the formula consumes is
countable:
- coul_fs   = 0.76      Coulomb finite-size coefficient
- i0_slope  = 0.4       slope in I0 = i0_slope * A / (A + i0_offset)
- i0_offset = 200.0     offset in the same I0 expression
- ee_pair   = 17/16     even-even pairing-case factor

Type designation: Type I — LOCAL_FITTABLE empty, no fit().

Mapping: Z, N, A, NZ_diff, pair are direct CSV columns; `pair` encoded
-1/0/+1 selects the pairing cases.
"""

import numpy as np

USED_INPUTS = ["Z", "N", "A", "NZ_diff", "pair"]
PAPER_REF = "summary_formula_wang_2014.md"
EQUATION_LOC = "Wang 2014 Eq. 1 (E_LD), PDF p. 5; coefficients Table I, PDF p. 7"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a_v":    -15.5181,
    "a_s":     17.4090,
    "a_c":      0.7092,
    "c_sym":   30.1594,
    "kappa":    1.5189,
    "xi":       1.2230,
    "a_pair":  -5.8166,
    "c_w":      0.8705,
    "kappa_s":  0.1536,
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "coul_fs":   0.76,        # Coulomb finite-size coefficient (Eq. 1 prose)
    "i0_slope":  0.4,         # slope in I0 = i0_slope * A / (A + i0_offset)
    "i0_offset": 200.0,       # offset in the same I0 expression
    "ee_pair":   17.0 / 16.0, # even-even pairing-case factor
}
LOCAL_FITTABLE = {}    # Type I


def predict(X: np.ndarray, a_v: float, a_s: float, a_c: float, c_sym: float,
            kappa: float, xi: float, a_pair: float, c_w: float,
            kappa_s: float) -> np.ndarray:
    """BE/A (MeV) under the WS4 liquid-drop component.

    X: (n, 5) — columns Z, N, A, NZ_diff, pair.
    """
    coul_fs   = OTHER_CONSTANTS["coul_fs"]
    i0_slope  = OTHER_CONSTANTS["i0_slope"]
    i0_offset = OTHER_CONSTANTS["i0_offset"]
    ee_pair   = OTHER_CONSTANTS["ee_pair"]

    Z = np.asarray(X[:, 0], dtype=float)
    A = np.asarray(X[:, 2], dtype=float)
    NZ_diff = np.asarray(X[:, 3], dtype=float)
    pair = np.asarray(X[:, 4], dtype=float)

    I = NZ_diff / A
    abs_I = np.abs(I)
    A13 = np.power(A, 1.0 / 3.0)
    A23 = A13 * A13

    coul = a_c * Z * Z / A13 * (1.0 - coul_fs * np.power(Z, -2.0 / 3.0))
    a_sym = c_sym * (1.0 - kappa / A13 + xi * (2.0 - abs_I) / (2.0 + abs_I * A))
    I0 = i0_slope * A / (A + i0_offset)
    eps = (I - I0) ** 2 - I ** 4
    f_s = 1.0 + kappa_s * eps * A13
    sym = a_sym * I * I * A * f_s

    odd_odd_mask   = (pair < -0.5).astype(float)
    even_even_mask = (pair > 0.5).astype(float)
    delta_np = (odd_odd_mask * (abs_I - I * I)
                + even_even_mask * (2.0 - abs_I - I * I) * ee_pair)
    pair_term = a_pair / A13 * delta_np
    wigner = c_w * abs_I

    E_LD = a_v * A + a_s * A23 + coul + sym + pair_term + wigner
    return -E_LD / A
