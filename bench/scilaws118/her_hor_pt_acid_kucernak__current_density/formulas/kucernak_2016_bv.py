"""Butler-Volmer limiting form of the Heyrovsky-Volmer model — current_density (Type I).

Kucernak, A. R. & Zalitis, C. (2016). General models for the electrochemical
hydrogen oxidation and hydrogen evolution reactions. J. Phys. Chem. C
120(20):10721-10745. DOI:10.1021/acs.jpcc.6b00011.

The paper shows (Eqs 20-21, PDF p. 9) that the full HV expression (Eq 19) reduces
to the classic Butler-Volmer (BV) form in the limits K ~ 1 with G << 1 (or G >> 1):

    j_BV(eta) = i0 * (exp(beta*f*eta) - exp(-(1-beta)*f*eta)),   f = F/(R*T)

with i0 a single exchange-current scale and temperature dependence via Eq 71
(PDF p. 27):  i0(T) = i0_ref * exp(-Ea/R * (1/T - 1/T_ref)).

The BV form is a LIMITING case. The paper states explicitly (PDF p. 9, below
Eq 21; p. 32) that neither K ~ 1 nor G << 1 holds on Pt/HClO4 (K_std ~ 3.09), so
the BV form is a deliberately-inadequate approximation baseline — it lacks the
K_std denominator term that gives the full HV its HOR/HER asymmetry, and (in this
single-i0 form) carries no explicit P_H2 / c_acid dependence. The full HV model
(kucernak_2016_hv.py) is the paper's recommended formula and the stronger baseline.

=== TYPE I (corrected 2026-06-02 audit — was mis-built as Type II) ===
Like the HV model, the BV limiting form is fit globally: one shared {i0_ref, beta,
Ea} across all conditions, with T captured explicitly by the Arrhenius term
(LOCAL_FITTABLE = {}, no fit()). See kucernak_2016_hv.py for the full rationale
(paper fits all 13 datasets simultaneously; Ea is unidentifiable per single
condition because T is constant within a condition).

LAW_CONSTANTS — the BV intrinsic constants (the discovery target)
----------------------------------------------------------------
- i0_ref : exchange current density at T_ref (A cm-2; absorbs 2*F*k_eq and the
           per-condition activities into one scale)
- beta   : symmetry / transfer coefficient (paper-reported ~1/3, PDF p. 28)
- Ea     : apparent activation energy, J/mol (paper-reported ~18 kJ/mol, PDF p. 28)
VALUES: globally least-squares fit to data/train.csv across all 13 conditions
("fit on our train"). The single-i0 BV form cannot capture the P_H2 dependence of
the 0.5 M series, so it fits the released data markedly worse than the full HV
model — the intended "BV is inadequate on Pt" message of the paper.

OTHER_CONSTANTS — given universal / structural constants the form consumes
-------------------------------------------------------------------------
- F     : Faraday constant 96485 C/mol      — CODATA universal (given, not LAW)
- R     : molar gas constant 8.314 J/(mol K) — CODATA universal (given, not LAW)
- T_ref : 298.15 K reference temperature     — fixed structural pivot in Eq 71
The factor 2 in the BV prefactor (two-electron transfer) is absorbed into i0_ref.

USED_INPUTS: eta (V vs RHE), T (K), P_H2 (bar), c_acid (mol/dm3). P_H2 and c_acid
are present for benchmark API consistency but do not enter this single-i0 BV form
(their effect is absorbed into i0_ref) — part of why BV under-fits.
"""

import numpy as np

USED_INPUTS = ["eta", "T", "P_H2", "c_acid"]
PAPER_REF = "summary_formula_dataset_kucernak_2016.md"
EQUATION_LOC = (
    "Kucernak & Zalitis (2016) Eqs 20-21 (BV limiting cases of HV, PDF p. 9); "
    "Eq 71 (temperature scaling, PDF p. 27). "
    "Fit globally to all 13 datasets (Type I)."
)

# The BV intrinsic constants — invariant across conditions (globally fit to train).
LAW_CONSTANTS = {
    "i0_ref": 62.16814883075403,    # A cm-2 — exchange current density at T_ref
    "beta":   0.20585757090144077,  # symmetry factor (paper ~1/3 for HV)
    "Ea":     27158.147495547546,   # apparent activation energy, J/mol
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


def _j_bv(eta, T, i0_ref, beta, Ea):
    """Butler-Volmer current density (A cm-2) per Eqs 20-21 + Eq 71."""
    T_arr = np.clip(T, 200.0, 500.0)
    f = _F / (_R * T_arr)
    i0_T = i0_ref * np.exp(-Ea / _R * (1.0 / T_arr - 1.0 / _T_REF))
    anodic   = np.exp(np.clip(beta * f * eta,           -80.0, 80.0))
    cathodic = np.exp(np.clip(-(1.0 - beta) * f * eta,  -80.0, 80.0))
    return i0_T * (anodic - cathodic)


def predict(
    X: np.ndarray,
    i0_ref: float = LAW_CONSTANTS["i0_ref"],
    beta: float = LAW_CONSTANTS["beta"],
    Ea: float = LAW_CONSTANTS["Ea"],
    **kwargs,
) -> np.ndarray:
    """Butler-Volmer limiting-case current density (A cm-2) for HOR/HER on Pt.

    X: (n, 4) — columns [eta (V), T (K), P_H2 (bar), c_acid (mol/dm3)].
    The harness (Type I) calls predict(X, **LAW_CONSTANTS); {i0_ref, beta, Ea}
    arrive as kwargs. F, R, T_ref are OTHER given constants read from module level.
    P_H2 / c_acid are not used by this single-i0 BV form. Returns A cm-2.
    """
    eta = np.asarray(X[:, 0], dtype=float)
    T   = np.asarray(X[:, 1], dtype=float)
    return _j_bv(eta, T, i0_ref, beta, Ea)
