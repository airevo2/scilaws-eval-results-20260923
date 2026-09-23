"""Khandelwal & Bhyravabhotla (2010) phenomenological DNA melting temperature model.

Khandelwal G, Bhyravabhotla J (2010). "A Phenomenological Model for Predicting
Melting Temperatures of DNA Sequences." PLoS ONE 5(8): e12433.
doi:10.1371/journal.pone.0012433. Equation (1), PDF page 4.

Formula (Equation 1, PDF p. 4, after the sentence "The final equation derived
after the multiple regression is:"):

    Tm(°C) = (7.35 × E) + [17.34 × ln(Len)] + [4.96 × ln(Conc)]
             + [0.89 × ln(DNA)] - 25.42

Variable definitions (PDF p. 4):
    Tm   = Predicted melting temperature (°C)
    E    = DNA strength parameter per base (dimensionless)
    Len  = Length of nucleotide sequence (number of base pairs)
    Conc = [Na+] concentration of the solution (Molar)
    DNA  = Total nucleotide strand concentration (Molar)

LAW_CONSTANTS — globally fitted regression coefficients, Equation (1), PDF p. 4
-----------------------------------------------------------------
    E_coeff      =  7.35   — coefficient for E (DNA strength parameter)
    ln_N_coeff   = 17.34   — coefficient for ln(sequence length)
    ln_salt_coeff =  4.96  — coefficient for ln([Na+])
    ln_dna_coeff  =  0.89  — coefficient for ln(strand concentration)
    intercept    = -25.42  — regression intercept

All five values are from Eq. (1) as printed on PDF page 4. The text extract
(reference/khandelwal_2010.txt, lines 316–317) confirms:
"Tmð0 CÞ~ð7:35|EÞz½17:34|lnðLenÞz½4:96|lnðConcÞ z½0:89|lnðDNAÞ{25:42"
(encoding artefact from PDF ligature; numerics confirmed).

OTHER_CONSTANTS — none
-----------------------------------------------------------------
The natural logarithm is a structural operator, not a constant. No unit
conversion factors needed (inputs are already in Molar and bp; output is °C).

Type designation: Type I
-----------------------------------------------------------------
The formula has NO per-cluster fitted parameters. All five regression
coefficients were globally fit once on the training dataset (123 oligomers,
Table S1) and frozen. Every row (individual oligonucleotide measurement) is
independent. LOCAL_FITTABLE is empty.

Column mapping (paper → CSV)
-----------------------------------------------------------------
    E (paper)   → E         (column 1)
    Len (paper) → N         (column 2)
    Conc        → salt_M    (column 3)
    DNA         → dna_conc_M (column 4)

Caveats
-----------------------------------------------------------------
- The formula was fit on 123 oligomers (15–30 bp; Tables S1). The test set
  (Table S3, 100 fifteen-mers) is fully held out (N=15 not in training).
- The E parameter is derived from the raw sequence via the dinucleotide
  strength table (Table 1, PDF p. 2); the pre-computed E column is released
  in the CSV rather than the raw sequences.
"""

import numpy as np

USED_INPUTS   = ["E", "N", "salt_M", "dna_conc_M"]
PAPER_REF     = "summary_formula+dataset_khandelwal_2010.md"
EQUATION_LOC  = "Eq. (1), PDF p. 4"

# === LAW_CONSTANTS — paper-published, frozen ===
LAW_CONSTANTS = {
    "E_coeff":       7.35,   # coefficient for DNA strength parameter E; Eq.(1) PDF p.4
    "ln_N_coeff":   17.34,   # coefficient for ln(sequence length); Eq.(1) PDF p.4
    "ln_salt_coeff": 4.96,   # coefficient for ln([Na+]); Eq.(1) PDF p.4
    "ln_dna_coeff":  0.89,   # coefficient for ln(strand concentration); Eq.(1) PDF p.4
    "intercept":   -25.42,   # regression intercept; Eq.(1) PDF p.4
}

# === OTHER_CONSTANTS — universal physics factors ===
OTHER_CONSTANTS = {}   # no unit factors or universal constants needed

LOCAL_FITTABLE  = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray,
            E_coeff: float,
            ln_N_coeff: float,
            ln_salt_coeff: float,
            ln_dna_coeff: float,
            intercept: float) -> np.ndarray:
    """Predict DNA melting temperature (°C) from Khandelwal 2010 Eq. (1).

    X: (n, 4) — columns in USED_INPUTS order: E, N, salt_M, dna_conc_M.
    Returns: (n,) array of predicted Tm in degrees Celsius.
    """
    E          = X[:, 0]
    N          = X[:, 1]
    salt_M     = X[:, 2]
    dna_conc_M = X[:, 3]

    return (E_coeff * E
            + ln_N_coeff    * np.log(N)
            + ln_salt_coeff * np.log(salt_M)
            + ln_dna_coeff  * np.log(dna_conc_M)
            + intercept)
