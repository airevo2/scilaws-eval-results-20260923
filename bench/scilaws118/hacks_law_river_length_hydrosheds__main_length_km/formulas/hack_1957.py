"""Hack's Law (Hack 1957, USGS Professional Paper 294-B, Eq. 3, p. 23).

The published equation in imperial units:

    L_mi = 1.4 * A_sqmi^0.6                      (mi, sq mi)

Converting both axes to SI/metric (1 mi = 1.609344 km, 1 sq mi =
2.589988 km^2) yields the metric-unit form used by the benchmark:

    L_km = 1.609344 * 1.4 * (A_km2 / 2.589988)^0.6
         = (1.609344 * 1.4 / 2.589988^0.6) * A_km2^0.6
         ~= 1.27291 * A_km2^0.6

Both the coefficient ``c = 1.4`` (mi-units) and the exponent ``h = 0.6``
are presented in the paper as fixed regional constants for the
northeastern US calibration set (>100 Virginia/Maryland localities; cross-
checked against ~400 Langbein 1947 gaging-station records). They are NOT
refit per basin in the source. The benchmark adopts the same convention:
both constants are taken at the published values and only treated as
fittable so the harness can quote a fitted-vs-published comparison.

References
----------
- summary_formula+dataset_hack_1957.md (PDF p. 23-25, Eq. 3 + Fig. 25)
- Hack 1957 PDF p. 23: "L = 1.4 A^{0.6}"
- Hack 1957 PDF p. 25: lithologic scatter in coefficient (1-2.5; ~2.0 for
  sandstone) noted but 1.4 is the representative regional value.
"""

import numpy as np

USED_INPUTS = ["upland_area_skm"]
PAPER_REF = "summary_formula+dataset_hack_1957.md"
EQUATION_LOC = "Eq. 3, PDF p. 23 (unit-converted: km from mi, km^2 from sq mi)"

# Both constants are paper-published values (Hack 1957 Eq. 3 unit-converted
# from imperial to metric). They are frozen as LAW_CONSTANTS.
# The unit-conversion factors (1.609344 mi->km, 2.589988 sq mi->km^2) are
# absorbed into the derived numeric value stored in LAW_CONSTANTS; there are
# no separate runtime unit-conversion constants, so OTHER_CONSTANTS is empty.
LAW_CONSTANTS = {
    # c = 1.609344 * 1.4 / 2.589988^0.6  (mi -> km, sq mi -> km^2)
    "c": 1.27291182,
    # exponent is dimensionless and unchanged by unit conversion
    "h": 0.6,
}
OTHER_CONSTANTS = {}   # no separate runtime unit-conversion factors needed
LOCAL_FITTABLE = {}    # Type I


def predict(X: np.ndarray, c: float, h: float) -> np.ndarray:
    """Hack's Law in metric units: L_km = c * A_km2^h.

    Parameters
    ----------
    X : np.ndarray, shape (n_rows, 1)
        Column 0 = ``upland_area_skm`` (km^2).
    c, h : float
        Power-law prefactor and exponent (Hack 1957 Eq. 3
        unit-converted to km / km^2).

    Returns
    -------
    np.ndarray, shape (n_rows,)
        Predicted main-stem length in km.
    """
    A = np.asarray(X[:, 0], dtype=float)
    return c * np.power(A, h)
