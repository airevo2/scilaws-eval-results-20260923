"""Block & Halzen (2005) Froissart-saturation ln²(s) parametrization for σ_tot(pp).

Reference paper:
  M. M. Block and F. Halzen (2005).
  "New evidence for the saturation of the Froissart bound."
  Physical Review D 72, 036006.
  arXiv: hep-ph/0506031.

This paper is cited as [2] in Block & Halzen (2011), Phys. Rev. D 83, 077901
(arXiv:1102.3163), which is the "Block & Halzen 2011" reference in the task.
The 2011 paper uses the same model parameters and only updates LHC predictions.

=== FORMULA ===
Equation (5), hep-ph/0506031 p. 5 (high-energy limit of the analytic amplitude):

    σ_± = c0 + c1·ln(ν/mp) + c2·ln²(ν/mp)
                + βP'·(ν/mp)^(µ-1)  ±  δ·(ν/mp)^(α-1)

where ν is the laboratory nucleon energy and mp is the proton mass.

In the high-energy kinematic limit s → 2·mp·ν (ν ≫ mp), this becomes:

    ln(ν/mp) = ln(s / (2·mp²)) = ln(s / s_0)

with s_0 = 2·mp² ≈ 1.7607 GeV² (mp = 0.93827 GeV).

This module implements the crossing-even component only — the three-parameter
ln²(s) form that is the main scientific claim of the paper:

    σ_tot(pp) ≈ c0 + c1·ln(s/s_0) + c2·ln²(s/s_0)

The full formula also includes:
  - βP'·(s/s_0)^((µ-1)/2) — Reggeon P' term (power-law falloff, vanishes at high s)
  - δ·(s/s_0)^((α-1)/2)   — odd amplitude (pp vs ppbar difference; small at high s)

At the LHC energies (s ≥ 10^7 GeV²) both correction terms are < 1 mb and
the 3-parameter form is the published approximation. For completeness the
Reggeon term is included as OTHER_CONSTANTS (not fitted).

=== LAW_CONSTANTS — paper's scientific claim, frozen ===
The implemented (graded) formula is the crossing-even THREE-parameter ln²(s)
form — the paper's headline Froissart-saturation claim. Its defining
coefficients, from Table 3 of hep-ph/0506031 (pp. 9-10 of the arXiv PDF),
ln²(ν/mp) fit with Δχ²_i,max = 6 (preferred cut, renormalized χ²/d.f. = 1.095):

    c0   = 37.32  mb   — constant offset
    c1   = −1.440 mb   — ln(s/s0) coefficient  (± 0.070 mb fit error)
    c2   = 0.2817 mb   — ln²(s/s0) coefficient (± 0.0064 mb fit error)

These are the directly-graded discovery target. The paper (PDF p. 1, just
after Eq. 8) calls the model's amplitudes "the real coefficients c0 , c1 ,
c2 ,βP ′ and δ".

=== OTHER_CONSTANTS — given / derived / fixed quantities ===
    s_0  = 2·mp² = 2·(0.93827)² ≈ 1.76070 GeV²  — reference scale.
        DERIVED from the universal proton mass via the high-energy kinematic
        substitution ν/m → s/(2·mp²) (s → 2·mp·ν, ν ≫ mp). Not a fitted
        coefficient — a kinematic conversion constant (cf. the gold's e2). It
        is consumed by predict() but read from OTHER_CONSTANTS, not graded as
        a discovery target. [Field-classification re-audit 2026-05-30:
        moved LAW → OTHER; see VERDICT.md.]

The full Eq. (5) adds two power-law correction terms that VANISH at high s
(< 1 mb at the LHC) and are NOT part of the implemented even form; their
Table-3 parameters are kept here for completeness but are NOT consumed by
predict():
    betaP = 37.10 mb, mu = 0.5            — Reggeon P' term  βP'·(s/s0)^((µ−1)/2)
    delta = −28.56 mb, alpha = 0.415      — odd amplitude    δ·(s/s0)^((α−1)/2); pp uses the + sign
  - mu  = 0.5 is held FIXED throughout (PDF p. 2: "µ = 0.5 throughout, which
    is appropriate for a Regge-descending trajectory") → fixed structural given.
  - alpha is the odd-amplitude exponent, completely determined by the
    constraint Eq. (15) from the experimental Δm, Δσ at the transition energy
    (PDF p. 2) → derived/given exponent.
  - betaP, delta ARE two of the model's "real coefficients" (PDF p. 2, with
    c0,c1,c2) — i.e. defining-coefficient-nature, NOT given quantities. They
    are filed in OTHER only because the implemented baseline is the even
    3-parameter form, which does not contain the βP' or δ terms, so they are
    inert here. Whether to (a) implement the full Eq. (5) and promote
    betaP,delta → LAW or (b) drop them as un-implemented is a structural /
    metric-changing decision deferred out of the classification-only re-audit
    (recorded AMBIGUOUS in VERDICT.md, 2026-05-30).

=== Type designation ===
Type I: global fit constants from paper; no per-cluster refit. LOCAL_FITTABLE
is empty.

=== Column mapping ===
Paper symbol  | Units  | CSV column | Position
σ_tot         | mb     | sigma_tot  | output
s             | GeV²   | s          | input column 0
"""

import numpy as np

USED_INPUTS = ["s"]
PAPER_REF = "block_halzen_2005.pdf"
EQUATION_LOC = "Eq. (5), arXiv:hep-ph/0506031 p. 5; Table 3 (pp/ppbar), PDF pp. 9-10"

#: Proton mass in GeV (PDG value used implicitly in the Block-Halzen fit).
_MP_GEV = 0.93827

# === LAW_CONSTANTS — paper-published defining coefficients, frozen ===
# The implemented even 3-parameter ln²(s) form; coefficients = Table 3,
# hep-ph/0506031 p. 9 (Δχ²_i,max = 6 column).
LAW_CONSTANTS = {
    "c0":  37.32,           # mb — constant offset; Table 3, hep-ph/0506031 p. 9
    "c1":  -1.440,          # mb — ln(s/s0) coefficient; Table 3, hep-ph/0506031 p. 9
    "c2":   0.2817,         # mb — ln²(s/s0) coefficient; Table 3, hep-ph/0506031 p. 9
}

# === OTHER_CONSTANTS — given / derived / fixed quantities ===
OTHER_CONSTANTS = {
    # s_0 = 2·mp²: reference scale DERIVED from the universal proton mass via
    # the high-energy kinematic substitution ν/m → s/(2·mp²). Not a fitted
    # coefficient. Consumed by predict() (read from this dict). [re-audit
    # 2026-05-30: moved LAW → OTHER — derived/universal, not a discovery target.]
    "s_0":  2 * _MP_GEV**2, # GeV² — = 2·mp² ≈ 1.76070; derived from Eq.(5) kinematics
    # --- Below: parameters of the FULL Eq. (5)'s correction terms (Reggeon P'
    #     + odd amplitude), which the implemented even form does NOT contain.
    #     NOT consumed by predict(); kept for completeness. All from Table 3,
    #     hep-ph/0506031 p. 9 (Δχ²_i,max = 6 column). ---
    "betaP": 37.10,   # mb — Reggeon P' amplitude. A "real coefficient" of Eq.(5) (PDF p.1) → defining-coefficient-nature; inert here. See VERDICT (AMBIGUOUS, 2026-05-30).
    "mu":     0.5,    # Reggeon exponent, held FIXED throughout (PDF p.2) → fixed structural given.
    "delta":  -28.56, # mb — odd-amplitude amplitude. A "real coefficient" of Eq.(5) (PDF p.1) → defining-coefficient-nature; inert here. See VERDICT (AMBIGUOUS, 2026-05-30).
    "alpha":   0.415, # odd-amplitude exponent, determined by constraint Eq.(15) from data (PDF p.2) → derived/given exponent.
}

LOCAL_FITTABLE = {}   # Type I — no per-cluster parameters


def predict(X: np.ndarray, c0: float, c1: float, c2: float) -> np.ndarray:
    """Predict pp total cross-section in mb using the Block-Halzen ln²(s) formula.

    Implements Eq. (5) of Block & Halzen (2005) hep-ph/0506031 in the
    high-energy limit (even amplitude only):

        σ_tot(pp) = c0 + c1·ln(s/s_0) + c2·ln²(s/s_0)

    Parameters
    ----------
    X : np.ndarray, shape (n, 1)
        Column 0: s — Mandelstam variable in GeV².
    c0, c1, c2 : float
        LAW_CONSTANTS (ln² parametrization coefficients in mb).

    Notes
    -----
    s_0 (the reference scale ≈ 1.7607 GeV² = 2·mp²) is a derived/universal
    quantity read from OTHER_CONSTANTS — it is not a graded discovery target.

    Returns
    -------
    sigma_tot : np.ndarray, shape (n,)
        Predicted total pp cross-section in mb.
    """
    s_0 = OTHER_CONSTANTS["s_0"]
    s = np.asarray(X[:, 0], dtype=float)
    log_s = np.log(s / s_0)
    return c0 + c1 * log_s + c2 * log_s**2
