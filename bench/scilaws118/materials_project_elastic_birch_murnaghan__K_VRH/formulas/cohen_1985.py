"""Cohen (1985) bulk modulus scaling law — paper-frozen form, no per-class fit.

Cohen, M. L. (1985). Calculation of bulk moduli of diamond and zinc-blende
solids. Phys. Rev. B 32, 7988. DOI: 10.1103/PhysRevB.32.7988.

Paper-frozen closed form (Cohen 1985 main result, confirmed via abstract
+ Cohen's 1985 widely-cited form quoted in subsequent literature, e.g.
Neumann 1987 Crystal Res. Technol. 22:122; Verma & Bhardwaj 2007 J. Phys.
Chem. Solids 68:1799; Xu, Wang & Tian 2013 Sci. Rep. 3:3068):

    K (GPa) = (1972 - 220 * lambda) / d^3.5            (Cohen 1985 eq.)

with `d` the nearest-neighbour bond length in Angstroms and `lambda` the
ionicity index that takes the integer values:

    lambda = 0  for group-IV  (C, Si, Ge, Sn — pure covalent)
    lambda = 1  for III-V     (e.g. GaAs, InP)
    lambda = 2  for II-VI     (e.g. ZnS, CdSe)

Cohen's center-of-table compact form (lambda ~ 1 numerically) is the
widely-quoted

    K (GPa) = 1761 * d^(-3.5)                          (Cohen 1985 abstract)

For this paper-frozen baseline we adopt the **lambda = 0 (group-IV pure
covalent) prefactor 1972**, which is the upper end of Cohen's family
and the form most often listed under "Cohen 1985" in
textbook tables (e.g. Anderson 1995 *Equations of State of Solids for
Geophysics*, Table 1-1; Kittel *Introduction to Solid State Physics* §3).
We do NOT fit lambda per crystal-system because (i) the Materials
Project elastic dataset is dominated by oxides and ionic solids that
fall outside Cohen's diamond/zincblende domain of validity — fitting
lambda per crystal-system would absorb that out-of-domain error into a
nominally "paper-frozen" parameter and constitute FM-C10d; (ii) Cohen's
own paper specifies lambda only for the three covalent-bond-character
classes listed above, none of which align with the seven Bravais
crystal-system classes that prep_data.py exposes.

Mapping V_atomic to bond length d
---------------------------------
Cohen's `d` is the nearest-neighbour bond length. The dataset exposes
only `V_atomic = V_cell / N_sites` (volume per atomic site). With the
canonical packing approximation `V_atomic ~ d^3` (i.e. one bond length
cubed per site, a defensible zero-th-order estimate that absorbs the
N_packing structural factor into the choice of effective bond length):

    d  =  V_atomic^(1/3)        (Angstroms)
    K  =  1972 / d^3.5
       =  1972 * V_atomic^(-3.5/3)

This is the paper-frozen Cohen baseline as expressed in V_atomic.

LAW_CONSTANTS — paper-frozen, from Cohen 1985
---------------------------------------------
- cohen_prefactor: 1972.0  (Cohen 1985 eq. with lambda=0, group-IV pure
  covalent. Confirmed via Cohen 1985 abstract on APS, Neumann 1987
  Crystal Res. Technol., Verma & Bhardwaj 2007 J. Phys. Chem. Solids,
  and Xu et al. 2013 Sci. Rep. 3:3068 review of polar-covalent forms.
  See reference/summary_birch_murnaghan_kvrh.md §3.)
- cohen_exponent: -3.5     (Cohen 1985 — K proportional to d^(-3.5);
  the exponent is the Cohen-Birch signature, distinct from the
  ionic-crystal scaling K proportional to d^(-3).)

OTHER_CONSTANTS — none
----------------------
This is a fully paper-frozen baseline with zero fit-on-data parameters.
Prior task A1 run baked per-crystal-system A prefactors fit on
data/train.csv into OTHER_CONSTANTS; those have been removed in the F1
restructure because (a) they were dataset-specific (FM-C10d-flavor when
shipped with an FM-A1/A2 closed-form interpretation), and (b) Cohen's
own paper does not partition by Bravais crystal system. The dataset-fit
per-csid prefactor remains available in the v1 source but is not a
paper-frozen Cohen baseline.

Expected performance
--------------------
On the Materials Project elastic dataset (mostly oxides and complex
inorganic compounds), Cohen's tetrahedral-covalent formula systematically
under-predicts K_VRH by ~50% (mean of K predicted at lambda=0 ~ 78 GPa
vs observed mean ~ 137 GPa), giving R^2 below the naive baseline.
This is expected and faithful to the paper: the formula's domain of
validity is diamond/zincblende, not the broad MP elastic-tensor set.
Linear_ols and anderson_nafe baselines are more appropriate for this
data; cohen_1985 is retained as a literature-anchored reference point
to demonstrate the V^(-3.5/3) functional form, not as a fit-quality
baseline.

Type designation: Type I — LOCAL_FITTABLE = {}, no fit() method.

Mapping: V_atomic_A3 (col 1) from USED_INPUTS.
"""

import numpy as np

USED_INPUTS = ["V_atomic_A3"]
PAPER_REF = "summary_birch_murnaghan_kvrh.md"
EQUATION_LOC = "Cohen 1985 Phys. Rev. B 32:7988 — K = (1972 - 220*lambda)/d^3.5; this baseline uses lambda=0 (group-IV) per-paper prefactor 1972 with d = V_atomic^(1/3); documented in summary_birch_murnaghan_kvrh.md §3"

LAW_CONSTANTS = {
    "cohen_prefactor": 1972.0,    # Cohen 1985 — group-IV (lambda=0) prefactor
    "cohen_exponent":  -3.5,      # Cohen 1985 — K proportional to d^(-3.5)
}

OTHER_CONSTANTS = {}

LOCAL_FITTABLE = {}   # Type I


def predict(X: np.ndarray, cohen_prefactor: float,
            cohen_exponent: float) -> np.ndarray:
    """K_VRH (GPa) from Cohen 1985 paper-frozen bulk modulus formula.

    K = cohen_prefactor / d^|cohen_exponent|    with d = V_atomic^(1/3)

    X: (n, 1) — column [V_atomic_A3] per USED_INPUTS.
    Coefficients are the LAW_CONSTANTS; the harness passes predict(X, **LAW_CONSTANTS).
    Returns K_VRH in GPa (Cohen 1985 lambda=0 prediction).
    """
    V = np.asarray(X[:, 0], dtype=float)
    V_safe = np.where(V > 0, V, 1e-9)
    d = np.power(V_safe, 1.0 / 3.0)
    return cohen_prefactor * np.power(d, cohen_exponent)
