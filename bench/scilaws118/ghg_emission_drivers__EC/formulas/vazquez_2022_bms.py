"""Bayesian Machine Scientist (BMS) symbolic baseline for energy consumption — Vázquez et al. 2022.

Citation: Vázquez D., Guimerà R., Sales-Pardo M., Guillén-Gosálbez G. (2022).
Automatic modeling of socioeconomic drivers of energy consumption and pollution
using Bayesian symbolic regression. Sustainable Production and Consumption 30,
596–607. DOI: 10.1016/j.spc.2021.12.025.

Formula (§4.2, PDF p. 8; parameters from Table 3 EC column, PDF p. 9):

    log(EC) = a₆·a₇ · ( a₄  +  AP·(a₇²·GDP·UR·(a₅^AP / TP)^a₂)^((a₁+a₃^DP)/AT)  +  AP )

Denominator in the outer exponent is AT (confirmed directly from the
rendered LaTeX image on PDF p. 8); the leading constant inside the
inner parens is a7² (not a2²). The expression was read from the PDF
image because the flat-text extraction garbles it across the page
break at pp. 8–9. AT enters the formula in Kelvin (= AT_Celsius +
273.15), matching the sister STIRPAT baseline's AT_K_OFFSET convention
and avoiding division-near-zero at the freezing point; the paper's
glossary (PDF p. 2) defines AT as "Average temperature (K)".

Audit trail: this docstring + the predict() body were repaired on
2026-05-25 after an opus PDF audit (audit/triages/vazquez_2022_bms_EC_PDF_AUDIT_2026-05-25.md)
found that the previous implementation (a) had the outer exponent
denominator wrong (a3 vs AT), (b) had the inner-base leading
constant wrong (a2² vs a7²), and (c) was missing AT from USED_INPUTS.
The previous docstring's "paper-internal inconsistency" narrative was
a post-hoc rationalisation of a mis-read; the paper IS self-consistent
(Table 4 EC/AT = X|X AND printed formula contains AT in the exponent
denominator, corroborated by Fig. 6's populated E_AT elasticity panel).

LAW_CONSTANTS — all from Table 3, EC column, PDF p. 9:
    a1 =  5.804
    a2 = -2.033
    a3 =  0.594
    a4 = -100.810
    a5 =  1.934
    a6 =  1.429e-13
    a7 =  1.834e12

    Note: a6 × a7 ≈ 0.262 (the large/small magnitudes are an artifact of
    the unconstrained BMS search and largely cancel each other).

OTHER_CONSTANTS = {"AT_K_OFFSET": 273.15}   (Celsius-to-Kelvin offset
                        for the AT input; consistent with sister STIRPAT
                        baseline which uses the same convention)

Type I: single global fit; no per-cluster parameters. LOCAL_FITTABLE = {}.

Column mapping (CSV → formula):
    GDP → GDP (per-capita GDP, 2015 USD)
    TP  → TP  (total population, persons)
    AP  → AP  (active population %, 15–64)
    DP  → DP  (population density, persons/km²)
    UR  → UR  (urbanisation rate %)
    AT  → AT  (mean annual surface air temperature, °C → Kelvin in predict)

Caveats:
  - The inner base a₇²·GDP·UR·(a₅^AP/TP)^a₂ is clipped to a small
    positive floor (1e-300) to prevent complex-number results when
    combinations are far outside the calibration domain; in practice
    the floor is never hit over the panel.
  - The product a₆·a₇ ≈ 0.262 keeps log_EC in a finite range. Note: this
    BMS implementation reproduces the paper's published functional form;
    Vazquez 2022 Table 1 reports R²=0.870 on the original EORA26 panel.
    Performance on the OOD top-GDP-quartile test split (released here)
    may differ; reference_metrics.json must be re-run after this 2026-05-25
    structural-fix commit (the prior measured rmse=2.15e7 / r2=-0.158 is
    STALE — it corresponded to the buggy a3-denominator + a2² formula).
"""

import numpy as np

USED_INPUTS = ["GDP", "TP", "AP", "DP", "UR", "AT"]
PAPER_REF = "summary_formula+dataset_vazquez_2022.md"
EQUATION_LOC = "§4.2, PDF p. 8 (formula image); Table 3 EC column, PDF p. 9"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "a1":  5.804,
    "a2": -2.033,
    "a3":  0.594,
    "a4": -100.810,
    "a5":  1.934,
    "a6":  1.429e-13,
    "a7":  1.834e12,
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {
    "AT_K_OFFSET": 273.15,  # AT Celsius → Kelvin conversion (paper convention; matches sister STIRPAT)
}

LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters


def predict(
    X: np.ndarray,
    a1: float,
    a2: float,
    a3: float,
    a4: float,
    a5: float,
    a6: float,
    a7: float,
) -> np.ndarray:
    """Predict EC (TJ) from X = [GDP, TP, AP, DP, UR, AT] columns.

    LAW_CONSTANTS (a1..a7) arrive as named params via predict(X, **LAW_CONSTANTS);
    the OTHER constant AT_K_OFFSET is read from the OTHER_CONSTANTS dict.
    """
    AT_K_OFFSET = OTHER_CONSTANTS["AT_K_OFFSET"]
    GDP = np.asarray(X[:, 0], dtype=float)
    TP  = np.asarray(X[:, 1], dtype=float)
    AP  = np.asarray(X[:, 2], dtype=float)
    DP  = np.asarray(X[:, 3], dtype=float)
    UR  = np.asarray(X[:, 4], dtype=float)
    AT_C = np.asarray(X[:, 5], dtype=float)
    AT_K = AT_C + AT_K_OFFSET  # Celsius → Kelvin per paper p. 2 glossary

    # Inner base: a7² · GDP · UR · (a5^AP / TP)^a2
    inner_base = (a7 ** 2) * GDP * UR * (a5 ** AP / TP) ** a2
    inner_base = np.maximum(inner_base, 1e-300)  # floor against complex power

    # Outer fractional exponent: (a1 + a3^DP) / AT  — denominator AT (Kelvin)
    # (confirmed from rendered LaTeX image on PDF p. 8; see audit triage 2026-05-25)
    exponent = (a1 + a3 ** DP) / AT_K

    inner = inner_base ** exponent

    log_EC = a6 * a7 * (a4 + AP * inner + AP)
    return np.exp(log_EC)
