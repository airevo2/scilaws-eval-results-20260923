"""Heyrovsky-Volmer (HV) kinetic model — current_density for HOR/HER on Pt (Type I).

Kucernak, A. R. & Zalitis, C. (2016). General models for the electrochemical
hydrogen oxidation and hydrogen evolution reactions. J. Phys. Chem. C
120(20):10721-10745. DOI:10.1021/acs.jpcc.6b00011.

The paper derives a closed-form steady-state current density for the two-step
Heyrovsky-Volmer mechanism:

    H2 + M  <-->  M-H_ads + H+ + e-   (Heyrovsky, rate const k1)
    M-H_ads <-->  M + H+ + e-          (Volmer,    rate const k2)

Steady-state H-coverage (Eq 15, PDF p. 7) gives the HV current density (Eq 19,
PDF p. 9, dimensionless-scaled form used for the fit):

    j_HV(eta) = F * k2_eq * 2*G*(exp((1+beta)*f*eta) - exp(-(1-beta)*f*eta))
                              / [(G*exp(f*eta)+1) + K*(exp(f*eta)+G)]

with f = F/(R*T), and the per-condition dimensionless parameters built from the
intrinsic constants via the explicit activity / Arrhenius terms:
    a_H2 = P_H2 / 1 bar    (Henry's law, Eq 69, PDF p. 27)
    a_H+ = c_acid          (proton activity)
    K_eq = K_std / a_H+
    G_eq = G_std * a_H2 / a_H+
    k2_eq(T) = k2_eq_ref * exp(-Ea/R * (1/T - 1/T_ref))   (Arrhenius, Eq 71, PDF p. 27)

=== TYPE I (corrected 2026-06-02 audit — was mis-built as Type II) ===
Kucernak & Zalitis fit ALL 13 polarisation curves SIMULTANEOUSLY with ONE shared
set of five parameters (PDF p. 27, Sec. 4.1, verbatim: "the five parameters are
used to simultaneously fit all 13 datasets"; Table 4 = "the simultaneous fit of
all data sets"; Fig. 8 = "a common set of parameters"). The five intrinsic
constants are therefore CROSS-CONDITION INVARIANTS — the paper's discovery
target — and belong in LAW_CONSTANTS. The condition-dependence (T, P_H2, c_acid)
is captured EXPLICITLY by the form above; there are no per-condition free
parameters, so this is a Type I global formula (LOCAL_FITTABLE = {}, no fit()).
(A per-cluster Type II treatment is also degenerate: within one condition T is
constant, so the Arrhenius factor collapses into the current scale and Ea is
unidentifiable from a single condition.)

LAW_CONSTANTS — the five intrinsic kinetic constants (the discovery target)
--------------------------------------------------------------------------
- k2_eq_ref : reference Volmer rate constant at T_ref (sets the current scale)
- K_std     : dimensionless equilibrium ratio (controls equil. H-coverage and
              HOR/HER asymmetry; paper-reported ~3.09, PDF p. 28)
- G_std     : dimensionless Heyrovsky/Volmer balance (per-condition G_eq =
              G_std*a_H2/a_H+)
- beta      : symmetry / transfer coefficient (paper-reported 1/3 for HV, PDF p. 28)
- Ea        : apparent activation energy, J/mol (single value across all
              elementary steps, paper-reported ~18 kJ/mol, PDF p. 28)

VALUES: globally least-squares fit to data/train.csv across all 13 conditions
(the SR's discovery target; "fit on our train" per the field-classification rule).
The released floating-electrode currents (jspecific, A cm-2 per real Pt area, up
to ~500 A cm-2) sit at a different absolute scale from the paper's own example
parameterisation (the paper's k2_eq~5.1e-6 reproduces neither the magnitude nor
sign of this data), so k2_eq_ref is fit to the released data; the recovered shape
constants (beta=0.35, Ea=17.8 kJ/mol, K_std=3.30) reproduce the paper's Table 4
values (1/3, 18 kJ/mol, 3.09), confirming the form and the fit. G_std~0.29 places
this data in the small-G regime; note k2_eq_ref and G_std are partially degenerate
there (only the product k2_eq_ref*G_std sets the scale) — the HV advantage over BV
comes from the K_std denominator term (K_std=3.30, not ~1), exactly the paper's
point that the Butler-Volmer reduction (K~1) does not hold on Pt.

OTHER_CONSTANTS — given universal / structural constants the form consumes
-------------------------------------------------------------------------
- F     : Faraday constant 96485 C/mol      — CODATA universal (given, not LAW)
- R     : molar gas constant 8.314 J/(mol K) — CODATA universal (given, not LAW)
- T_ref : 298.15 K reference temperature     — fixed structural pivot in Eq 71
The factor 2 in 2*G*(...) is the two-electron-transfer stoichiometry (structural
literal, stays inline).

USED_INPUTS: eta (V vs RHE), T (K), P_H2 (bar), c_acid (mol/dm3) — all per-row.
"""

import numpy as np

USED_INPUTS = ["eta", "T", "P_H2", "c_acid"]
PAPER_REF = "summary_formula_dataset_kucernak_2016.md"
EQUATION_LOC = (
    "Kucernak & Zalitis (2016) Eq 15 (theta, PDF p. 7), "
    "Eq 19 (j_HV dimensionless-scaled form, PDF p. 9), "
    "Eq 71 (temperature scaling, PDF p. 27), "
    "Eq 69 (Henry's law a_H2, PDF p. 27). "
    "Five parameters fit simultaneously to all 13 datasets (PDF p. 27, Sec. 4.1)."
)

# The five intrinsic kinetic constants — invariant across all conditions, the
# paper's simultaneous-fit discovery target (globally fit to data/train.csv).
LAW_CONSTANTS = {
    "k2_eq_ref": 0.009599691056554147,   # reference Volmer rate (current scale)
    "K_std":     3.2974833423959504,     # equilibrium ratio (paper ~3.09)
    "G_std":     0.2928146998115704,     # Heyrovsky/Volmer balance
    "beta":      0.3521851894691138,     # symmetry factor (paper ~1/3)
    "Ea":        17797.591475410172,     # apparent activation energy, J/mol (paper ~18 kJ/mol)
}

# Given universal / structural constants the form consumes (exposed as priors).
OTHER_CONSTANTS = {
    "F":     96485.0,    # C/mol — Faraday constant (CODATA universal)
    "R":     8.314,      # J/(mol*K) — molar gas constant (CODATA universal)
    "T_ref": 298.15,     # K — reference temperature (structural pivot, Eq 71)
}

LOCAL_FITTABLE = {}      # Type I — no per-cluster parameters, no fit()

_F = OTHER_CONSTANTS["F"]
_R = OTHER_CONSTANTS["R"]
_T_REF = OTHER_CONSTANTS["T_ref"]


def _j_hv(eta, T, P_H2, c_acid, k2_eq_ref, K_std, G_std, beta, Ea):
    """Heyrovsky-Volmer current density (A cm-2) per Eq 18/19 + Eq 71."""
    f = _F / (_R * T)
    a_H2 = P_H2 / 1.0          # Henry's law, P_H2_std = 1 bar (literal)
    a_Hp = c_acid               # a_H+ ~ c_HClO4 (mol/dm3)
    K_eq = K_std / a_Hp
    G_eq = G_std * a_H2 / a_Hp

    T_arr = np.clip(T, 200.0, 500.0)
    k2_T = k2_eq_ref * np.exp(-Ea / _R * (1.0 / T_arr - 1.0 / _T_REF))

    exp_f_eta = np.exp(np.clip(f * eta, -80.0, 80.0))
    exp_ap_f  = np.exp(np.clip((1.0 + beta) * f * eta, -80.0, 80.0))
    exp_am_f  = np.exp(np.clip(-(1.0 - beta) * f * eta, -80.0, 80.0))

    numer = 2.0 * G_eq * (exp_ap_f - exp_am_f)
    denom = (G_eq * exp_f_eta + 1.0) + K_eq * (exp_f_eta + G_eq)
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    return _F * k2_T * numer / denom


def predict(
    X: np.ndarray,
    k2_eq_ref: float = LAW_CONSTANTS["k2_eq_ref"],
    K_std: float = LAW_CONSTANTS["K_std"],
    G_std: float = LAW_CONSTANTS["G_std"],
    beta: float = LAW_CONSTANTS["beta"],
    Ea: float = LAW_CONSTANTS["Ea"],
    **kwargs,
) -> np.ndarray:
    """Heyrovsky-Volmer current density (A cm-2) for HOR/HER on Pt.

    X: (n, 4) — columns [eta (V), T (K), P_H2 (bar), c_acid (mol/dm3)].
    The harness (Type I) calls predict(X, **LAW_CONSTANTS); the five intrinsic
    constants arrive as kwargs. F, R, T_ref are OTHER given constants read from
    module level. Returns current_density (A cm-2 per real Pt area).
    """
    eta    = np.asarray(X[:, 0], dtype=float)
    T      = np.asarray(X[:, 1], dtype=float)
    P_H2   = np.asarray(X[:, 2], dtype=float)
    c_acid = np.asarray(X[:, 3], dtype=float)
    return _j_hv(eta, T, P_H2, c_acid, k2_eq_ref, K_std, G_std, beta, Ea)
