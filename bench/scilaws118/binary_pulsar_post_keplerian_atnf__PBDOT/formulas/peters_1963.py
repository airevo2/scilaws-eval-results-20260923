"""GR orbital-period decay of a compact binary -- Peters & Mathews 1963.

Peters & Mathews (1963), "Gravitational Radiation from Point Masses in a
Keplerian Orbit", Phys. Rev. 131:435 (DOI 10.1103/PhysRev.131.435).
PDF: reference/peters_1963.pdf.

The paper's principal result is Eq. (16), PDF p. 437, for the orbit-averaged
gravitational-wave power radiated by a Keplerian binary of arbitrary
eccentricity:

    <P> = (32/5) * (G^4 / c^5) * m1^2 m2^2 (m1+m2) / [a^5 (1-e^2)^(7/2)]
          * (1 + (73/24) e^2 + (37/96) e^4)

The corresponding eccentricity enhancement factor (Eq. 17, PDF p. 437) is

    f(e) = (1 + (73/24) e^2 + (37/96) e^4) / (1 - e^2)^(7/2).

The orbital-period derivative PBDOT (the benchmark target) follows from
energy balance dE/dt = -<P> combined with the Keplerian energy
E = -G m1 m2 / (2a) and Kepler's third law P_b^2 = 4 pi^2 a^3 / [G (m1+m2)]:

    PBDOT = -(192 pi / 5) * (G / c^3)^(5/3)
            * (P_b / 2pi)^(-5/3) * m1 m2 / (m1 + m2)^(1/3) * f(e).

In solar / geometric units (T_sun = G M_sun / c^3 = 4.925490947 us,
Kramer 2006 PDF p. 9) this rearranges to the chirp-mass form (Will 2014
Eq. 108, PDF p. 69):

    PBDOT = -(192 pi / 5) * (2 pi * M_c * T_sun * f_b)^(5/3) * F(e)

    where  M_c = eta^(3/5) * (m1+m2) = (m1 m2)^(3/5) / (m1+m2)^(1/5)
    is the chirp mass [M_sun],  f_b = 1 / P_b [Hz],  F(e) = f(e).

The benchmark task ships only (Pb, e) as inputs. Individual component masses
are not measured for most ATNF binaries (Mp_Msun is NaN for 91/92 rows in
psrcat v2.8.0), so the chirp mass is supplied here as a single canonical
value M_c = 1.20 M_sun -- the population mean for galactic
double-neutron-star (DNS) binaries reported by Antoniadis et al. (2013)
ApJ 778 and Bagchi (2013) MNRAS 428 (~1.20 M_sun, sigma ~0.07 M_sun).
This value is the **canonical DNS chirp mass** used throughout the literature
for population-level GR-decay estimates when per-system masses are not
available.

Symbol map (paper -> released CSV columns):
    P_b  <-  Pb  (days; converted to seconds inside predict)
    e    <-  e

LAW_CONSTANTS / OTHER_CONSTANTS classification (v2 contract):

  LAW_CONSTANTS  (paper-published, frozen, NOT refit):
      None -- every numerical coefficient (192 pi / 5, 73/24, 37/96, 7/2,
              5/3) is a PN structural constant of GR (Peters 1963 Eq. 16-17;
              Will 2014 Eq. 88, 108).

  OTHER_CONSTANTS  (universal physics constants / unit conversions):
      T_sun       = 4.925490947e-6  s    G M_sun / c^3     (Kramer 2006 PDF p. 9)
      M_c         = 1.20            M_sun   canonical DNS chirp mass
                  (Antoniadis 2013 ApJ 778; Bagchi 2013 MNRAS 428)
      DAY_TO_SEC  = 86400.0         s/d    days -> seconds unit conversion
                  (Pb is catalogued in days; 1 d = 86400 s exactly, SI)

  LOCAL_FITTABLE: {} -- Type I, no per-system parameters.

Runnability
-----------
The formula is defined for every test row (Pb > 0, 0 <= e < 1 satisfied by
data construction). It uses **only** the inputs that are NaN-free in both
train and test partitions (Wave-17 column drop). The canonical M_c = 1.20
M_sun is a population-level approximation;
individual systems deviate by a factor of <2 (J0737-3039A/B M_c ~ 1.13;
B1913+16 M_c ~ 1.23). At the per-pulsar level the prediction can be off
by a factor ~2 -- correct to the GR-power scaling exponent (5/3) but with
a constant offset reflecting the canonical-vs-actual chirp-mass mismatch.
"""

import numpy as np

USED_INPUTS = ["Pb", "e"]
PAPER_REF = "summary_formula_peters_1963.md"
EQUATION_LOC = ("Peters & Mathews 1963 Eq. 16 (PDF p. 437) + Eq. 17 (PDF p. 437); "
                "PBDOT derived form -- Will 2014 Eq. 108 (PDF p. 69) "
                "and Kramer 2006 PDF p. 9 (T_sun definition); "
                "canonical M_c from Antoniadis 2013 ApJ 778 / Bagchi 2013 MNRAS 428")

# === LAW_CONSTANTS -- paper-published, frozen ===
LAW_CONSTANTS = {}     # all numerical coefficients are PN structural constants

# === OTHER_CONSTANTS -- universal physics factors / unit conversions / canonical means ===
OTHER_CONSTANTS = {
    "T_sun":      4.925490947e-6,   # s; solar mass in geometric units G M_sun / c^3
    "M_c":        1.20,             # M_sun; canonical DNS chirp mass
                                    # (Antoniadis+2013, Bagchi 2013)
    "DAY_TO_SEC": 86400.0,          # s/d; days -> seconds (1 d = 86400 s, SI exact)
}

LOCAL_FITTABLE = {}    # Type I -- no per-cluster parameters


def predict(X: np.ndarray) -> np.ndarray:
    """Predict GR PBDOT via Peters 1963 / Will 2014 Eq. 108, canonical M_c.

    Type I, LAW_CONSTANTS = {} -- the harness calls predict(X, **{}) with no
    kwargs. All given constants (T_sun, the canonical chirp mass M_c, and the
    days->seconds conversion) are read from OTHER_CONSTANTS (gold style);
    every other coefficient (192 pi / 5, 2 pi, 73/24, 37/96, 7/2, 5/3) is a
    structural rational of the published GR law, kept inline.

    Parameters
    ----------
    X : np.ndarray, shape (n, 2)
        Columns in USED_INPUTS order: Pb [d], e [dimensionless].

    Returns
    -------
    np.ndarray, shape (n,)
        Predicted PBDOT [dimensionless, s/s]. Always negative for bound
        orbits (energy is radiated away).
    """
    T_sun = OTHER_CONSTANTS["T_sun"]
    M_c = OTHER_CONSTANTS["M_c"]
    day_to_sec = OTHER_CONSTANTS["DAY_TO_SEC"]

    Pb_d = np.asarray(X[:, 0], dtype=float)
    e = np.asarray(X[:, 1], dtype=float)

    Pb_s = Pb_d * day_to_sec
    f_b = 1.0 / Pb_s            # orbital frequency [Hz]

    e2 = e * e
    e4 = e2 * e2
    # f(e) = (1 + 73/24 e^2 + 37/96 e^4) / (1 - e^2)^(7/2)   (Eq. 17)
    F_e = (1.0 + (73.0 / 24.0) * e2 + (37.0 / 96.0) * e4) / (1.0 - e2) ** 3.5

    # PBDOT = -(192 pi / 5) * (2 pi M_c T_sun f_b)^(5/3) * F(e)   (Will 2014 Eq. 108)
    return -(192.0 * np.pi / 5.0) \
           * (2.0 * np.pi * float(M_c) * float(T_sun) * f_b) ** (5.0 / 3.0) \
           * F_e
