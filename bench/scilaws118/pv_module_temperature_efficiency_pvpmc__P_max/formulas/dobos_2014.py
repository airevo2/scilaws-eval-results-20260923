"""PVWatts V5 linear temperature-corrected DC power model (Type I).

Dobos, A.P. (2014). *PVWatts Version 5 Manual*. NREL/TP-6A20-62641,
National Renewable Energy Laboratory.
DOI (OSTI): https://www.osti.gov/biblio/1158421

Core equation (Eq. 8, PDF p. 12):

    P_dc = (I_tr / G_STC) * P_dc0 * (1 + gamma * (T_cell - T_ref))

where G_STC = 1000 W/m² and T_ref = 25 °C are the Standard Rating Conditions
(SRC) defined in the paper, and the factor normalises measured irradiance to a
fraction of the SRC value.

Benchmark mapping
-----------------
In the PVPMC benchmark (NREL PVDAQ system 4902, NIST_Ground_1, Sharp NU-U235F2
c-Si modules), the raw POA irradiance column irradiance_poa_o_2204 maps to I_tr
and the back-of-cell module temperature temperature_module_o_2206 maps to T_cell.
T_cell is a direct measurement, so the Fuentes thermal sub-model is not needed.

Type designation: Type I — all 7 shunt groups instrument the SAME physical module
type (Sharp NU-U235F2, 235 W STC); no per-cluster variation in formula structure.
LOCAL_FITTABLE is empty.

LAW_CONSTANTS — paper-published defining coefficients fitted FOR this law
------------------------------------------------------------------------
Empty. The PVWatts V5 DC-power model has no coefficient that is *fitted on this
task's data* as the law's discovery target. The scientific claim of this baseline
is the functional FORM itself: power is linear in transmitted irradiance with a
linear temperature derating, P = (G/G_ref)·P_dc0·(1 + γ·(T − T_ref)). Every
numeric quantity it consumes is a standard reference condition, a system spec, or
a module-type datasheet value supplied to the model (see OTHER_CONSTANTS) — not a
coefficient this task calibrates. (Field-classification re-audit 2026-06-02 moved
all four numbers OTHER per MANUAL §1/§2; cf. the gravity_wgs84 precedent where an
all-givens Type I form has LAW = {}.)

OTHER_CONSTANTS — givens the form consumes (reference conditions / specs)
------------------------------------------------------------------------
- G_STC : 1000 W/m² — Standard Rating Condition reference irradiance. The paper
          states it directly: Dobos 2014 Eq. 8 narrative, PDF p. 12 — "reference
          irradiance is 1000 W/m²". A standard reference condition / structural
          normalization, not a discovery target. (OTHER, kind a/c)
- T_ref : 25 °C — SRC reference cell temperature. Dobos 2014 Eq. 8 narrative,
          PDF p. 12 — "The reference cell temperature Tref is 25◦ C". A standard
          reference condition, not a discovery target. (OTHER, kind a)
- P_dc0 : 235 W — nameplate DC power rating at SRC for the Sharp NU-U235F2 array.
          The paper treats Pdc0 as a user-supplied system spec: Dobos 2014 PDF
          p. 12 — "a speciﬁed nameplate DC rating of Pdc0". A given module/system
          spec the formula merely scales by, NOT fitted on this task's data.
          (OTHER, kind a) — NOTE the docstring's earlier "Sandia module database
          entry for Sharp NU-U235F2" provenance is FALSE (NU-U235F2 is absent from
          pvlib SandiaMod; that wrong DB claim is why the sibling king_2004 baseline
          was dropped — see VERDICT.md). The traceable provenance for 235 W is the
          PVDAQ system-4902 nameplate / the module's IEC STC rating, not a SandiaMod
          entry; it is a given spec regardless.
- gamma : -0.0047 /°C — temperature coefficient of DC power for the Standard
          (multicrystalline-Si) module class. The paper FITS this on an EXTERNAL
          database, not on this task's data: Dobos 2014 Table 3, PDF p. 7 lists the
          three technology-class values (Standard -0.47 %/°C, Premium -0.35 %/°C,
          Thin film -0.20 %/°C), and PDF p. 12 states "The values used were
          determined from a statistical analysis of over 11000 modules in the CEC
          module database". PVWatts thus consumes γ as a per-module-type datasheet/
          database spec selected by technology class — a given, NOT a coefficient
          this task calibrates. (OTHER, kind a)
"""

import numpy as np

USED_INPUTS = ["G_W_m2", "T_module_C"]
PAPER_REF   = "summary_formula_dobos_2014.md"
EQUATION_LOC = (
    "Dobos (2014) Eq. 8, PDF p. 12 — "
    "P_dc = (I_tr / G_STC) * P_dc0 * (1 + gamma * (T_cell - T_ref)); "
    "Table 3, PDF p. 7 — gamma = -0.0047 /°C for standard silicon."
)

# === LAW_CONSTANTS — paper-published defining coefficients fitted FOR this law ===
LAW_CONSTANTS = {}      # the claim is the FORM; all numbers are consumed givens (see OTHER_CONSTANTS)
# === OTHER_CONSTANTS — givens the form consumes (reference conditions / specs) ===
OTHER_CONSTANTS = {
    "G_STC":  1000.0,   # W/m² — SRC reference irradiance (Dobos 2014 PDF p. 12, "reference irradiance is 1000 W/m²")
    "T_ref":  25.0,     # °C — SRC reference cell temperature (Dobos 2014 PDF p. 12, "Tref is 25◦ C")
    "P_dc0":  235.0,    # W — Sharp NU-U235F2 nameplate DC rating, a user-supplied system spec
                        # (Dobos 2014 PDF p. 12, "specified nameplate DC rating of Pdc0")
    "gamma":  -0.0047,  # 1/°C — Standard-Si module-type temp coeff from CEC-database analysis
                        # (Dobos 2014 Table 3, PDF p. 7; "determined from ... over 11000 modules in the CEC module database")
}
LOCAL_FITTABLE = {}    # Type I — no per-cluster fitting


def predict(X: np.ndarray) -> np.ndarray:
    """PVWatts V5 DC power at maximum power point (Dobos 2014 Eq. 8).

    X: (n, 2) — columns [G_W_m2 (W/m²), T_module_C (°C)].
    Returns P_max (W per module), clipped non-negative.

    LAW_CONSTANTS is empty (the claim is the functional form); the consumed
    givens are read from OTHER_CONSTANTS, matching the gold OTHER-read style.
    """
    G_STC = OTHER_CONSTANTS["G_STC"]
    T_ref = OTHER_CONSTANTS["T_ref"]
    P_dc0 = OTHER_CONSTANTS["P_dc0"]
    gamma = OTHER_CONSTANTS["gamma"]
    G = np.asarray(X[:, 0], dtype=float)
    T = np.asarray(X[:, 1], dtype=float)
    P = (G / G_STC) * P_dc0 * (1.0 + gamma * (T - T_ref))
    return np.clip(P, 0.0, None)
