"""Maximal-basin Hack's Law (Dodds & Rothman 2000, Annu. Rev. Earth Planet.
Sci. 28, Eq. 36 / Fig. 8 caption, PDF p. 23).

The published equation, fitted to 37 of the world's largest river basins
(data tabulated in Leopold 1994):

    l_mi = c~ * a_sqmi^h~                          (mi, sq mi)

with reported constants ``c~ = 3.0`` (1-sigma range 1.3-6.6) and
``h~ = 0.50 +/- 0.06`` (1-sigma range 0.44-0.56).

Converting to SI/metric units (1 mi = 1.609344 km, 1 sq mi = 2.589988
km^2) yields:

    l_km = 1.609344 * 3.0 * (a_km2 / 2.589988)^0.5
         = (1.609344 * 3.0 / 2.589988^0.5) * a_km2^0.5
         = 3.000000 * a_km2^0.5

(the metric-unit prefactor is numerically equal to 3.0 because the
unit-conversion factors happen to cancel for h = 0.5).

The constants ``c~`` and ``h~`` are presented as universal maximal-basin
constants (single fit to 37 basins, no per-basin refit), so they are
GLOBAL_FITTABLE with the paper's published values as INIT defaults.

References
----------
- summary_formula_dodds_2000.md (PDF p. 22-23, Eq. 36 + Fig. 8 caption)
- Dodds & Rothman 2000 PDF p. 23: "l~ = c~ a~^{h~}, h~ = 0.50 +/- 0.06,
  c~ = 3.0 (1-sigma 1.3-6.6)"
- Dodds & Rothman 2000 PDF p. 7 Eq. 13: general form l_bar ~ a^h.
"""

import numpy as np

USED_INPUTS = ["upland_area_skm"]
PAPER_REF = "summary_formula_dodds_2000.md"
EQUATION_LOC = "Eq. 36 / Fig. 8 caption, PDF p. 23 (unit-converted: km from mi, km^2 from sq mi)"

# Both constants are paper-published values (Dodds & Rothman 2000 Eq. 36
# unit-converted from imperial to metric). They are frozen as LAW_CONSTANTS.
# The unit-conversion factors (1.609344, 2.589988) cancel numerically for
# h=0.5, leaving c_tilde=3.0 in both unit systems; OTHER_CONSTANTS is empty.
LAW_CONSTANTS = {
    # c~ = 1.609344 * 3.0 / 2.589988^0.5 = 3.0 (km, km^2; factors cancel)
    "c_tilde": 3.0,
    # h~ = 0.50 (paper Eq. 36)
    "h_tilde": 0.5,
}
OTHER_CONSTANTS = {}   # no separate runtime unit-conversion factors needed
LOCAL_FITTABLE = {}    # Type I


def predict(X: np.ndarray, c_tilde: float, h_tilde: float) -> np.ndarray:
    """Maximal-basin Hack's law: L_km = c~ * A_km2^h~.

    Parameters
    ----------
    X : np.ndarray, shape (n_rows, 1)
        Column 0 = ``upland_area_skm`` (km^2).
    c_tilde, h_tilde : float
        Prefactor and exponent (Dodds & Rothman 2000 Eq. 36
        maximal-basin fit, unit-converted to km / km^2).

    Returns
    -------
    np.ndarray, shape (n_rows,)
        Predicted main-stem length in km.
    """
    A = np.asarray(X[:, 0], dtype=float)
    return c_tilde * np.power(A, h_tilde)
