"""Möller-Nix-Myers-Swiatecki (1995) FRDM macroscopic energy, spherical limit.

Möller, Nix, Myers & Swiatecki, At. Data Nucl. Data Tables 59 (1995)
185-381. The full FRDM combines a macroscopic droplet-model term with a
microscopic folded-Yukawa Strutinsky shell + pairing correction whose
ground-state shape parameters are minimised numerically per nucleus (no
closed form). This module implements the analytically tractable
macroscopic part of Eq. 40 in the spherical / undistorted limit
(B_i = 1, epsilon_gs = 0, tau_bar = 0, delta_bar = I):

    B(Z, N) = a1 * A - J * I**2 * A
            - a2 * A**(2/3)
            - (9/4) * J**2 / Q * I**2 * A**(2/3)
            - c1 * Z**2 / A**(1/3)
            + c4 * Z**(4/3) / A**(1/3)
            + c_a * (N - Z)
            - W * |I|
            - pairing(Z, N, A)

with I = (N-Z)/A and the Eq. 40 four-case pairing rule, Delta = r_pair/A^(1/3),
delta_np = h_np/A^(2/3). The benchmark target is BE/A = B/A.

LAW_CONSTANTS — paper-published, frozen
---------------------------------------
The eight primary macroscopic coefficients from Möller 1995 calibration
lists pp. 18-19:

    a1 = 16.247   a2 = 23.92    J  = 32.73    Q  = 29.21
    c1 = 0.7448153
    c_a = 0.436   W  = 30.0     r_pair = 4.80

OTHER_CONSTANTS — derived or universal physics factors
------------------------------------------------------
- c4 = 0.5687475 MeV : Coulomb-exchange correction coefficient, fully
  determined by c1 through Möller 1995 Eq. 66 (PDF p. 21):
      c4 = (5/4) * (3/(2π))^(2/3) * c1
  With the paper's c1 = (3/5)·e²/r0 (e² = 1.4399764 MeV·fm, r0 = 1.16 fm)
  giving c1 = 0.7448153 MeV, this evaluates to c4 = 0.5687475 MeV. c4 is a
  derived constant (not independently calibrated); the paper's FRDM
  constants table on pp. 18-19 lists a1, a2, J, Q, ca, W, rmac, h, but
  NOT c4 — c4 is left implicit via the Eq. 66 relation. It is stored in
  OTHER_CONSTANTS per the C10 protocol for non-free-parameter constants
  that the predict() function consumes.
- h_np = 6.6 MeV : the universal neutron-proton pairing interaction
  constant (Möller 1995 p. 18). Fixed, not a discovery target.

Type designation: Type I — LOCAL_FITTABLE empty, no fit().

Mapping: Z, N, A, NZ_diff, pair are direct CSV columns; `pair` encoded
-1 (odd-odd), 0 (odd-A), +1 (even-even) selects the pairing cases.
"""

import numpy as np

USED_INPUTS = ["Z", "N", "A", "NZ_diff", "pair"]
PAPER_REF = "summary_formula+dataset_moller_1995.md"
EQUATION_LOC = "Möller 1995 Eq. 40 macroscopic, PDF p. 14 (constants pp. 18-19)"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a1":     16.247,
    "a2":     23.92,
    "J":      32.73,
    "Q":      29.21,
    "c1":      0.7448153,
    "c_a":     0.436,
    "W":      30.0,
    "r_pair":  4.80,
}
# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    # c4 is DERIVED from c1 per Möller 1995 Eq. 66 (PDF p. 21):
    #   c4 = (5/4)(3/(2π))^(2/3) * c1
    # With c1 = 0.7448153 (= (3/5)·e²/r0; e²=1.4399764 MeV·fm, r0=1.16 fm),
    # this gives c4 = 0.5687475 MeV. The paper's constants table on pp. 18-19
    # does NOT publish c4 explicitly — c4 is implicit via Eq. 66.
    "c4":   0.5687475,
    "h_np": 6.6,    # MeV — universal np-pairing interaction constant (p. 18)
}
LOCAL_FITTABLE = {}    # Type I


def predict(X: np.ndarray, a1: float, a2: float, J: float, Q: float,
            c1: float, c_a: float, W: float, r_pair: float) -> np.ndarray:
    """BE/A (MeV) under the FRDM macroscopic spherical-limit model.

    X: (n, 5) — columns Z, N, A, NZ_diff, pair.
    c4 is a derived constant (Eq. 66: c4 = (5/4)(3/(2π))^(2/3) * c1) and
    is read from OTHER_CONSTANTS rather than passed as a free parameter.
    """
    h_np = OTHER_CONSTANTS["h_np"]
    c4   = OTHER_CONSTANTS["c4"]
    Z = np.asarray(X[:, 0], dtype=float)
    N = np.asarray(X[:, 1], dtype=float)
    A = np.asarray(X[:, 2], dtype=float)
    NZ_diff = np.asarray(X[:, 3], dtype=float)
    pair = np.asarray(X[:, 4], dtype=float)

    I = NZ_diff / A
    A13 = np.power(A, 1.0 / 3.0)
    A23 = A13 * A13
    Z43 = np.power(Z, 4.0 / 3.0)

    vol  = a1 * A - J * I * I * A
    surf = -a2 * A23 - (9.0 / 4.0) * (J * J / Q) * I * I * A23
    coul = -c1 * Z * Z / A13 + c4 * Z43 / A13
    chg_asym = c_a * (N - Z)
    wigner = -W * np.abs(I)

    Delta = r_pair / A13
    delta_np = h_np / A23
    n_odd = 1.0 - pair                             # +1→0, 0→1, -1→2
    odd_odd_mask = (pair < -0.5).astype(float)
    pairing_excess = n_odd * Delta - odd_odd_mask * delta_np

    B = vol + surf + coul + chg_asym + wigner - pairing_excess
    return B / A
