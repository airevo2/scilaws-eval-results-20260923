# Memorization probe — frozen protocol (kit edition, 2026-09)

This is the instrument. Every leg, on every model, must use exactly these settings; a leg run
with different settings cannot enter the moat computation ("clean under **all** panel models").

## 1. What is measured
For each `(task, model)`: can the model write the task's published closed-form law from memory,
given only the target and input descriptions (no data)?

| probe | what the model sees | samples | temperature | used in the paper |
|---|---|---|---|---|
| **D1 cold recall** | target + inputs (names, symbols, units, descriptions, typical ranges); nothing else | **n=5** | **0.8** (chat models; reasoning models run at the provider default) | **yes — the only probe the paper uses** |
| D2 cued recall | D1 + `Surname (year)` of the source paper | n=3 | 0.3 | no (dropped 2026-06-27); `--probes d1,d2` keeps it available |

The D1 prompt template is in `probe.py` (`D1_PROMPT`) and is byte-identical to the paper runs.
`prompts_frozen.json` holds a hash of every rendered prompt; `run_mem.py --dry-run` verifies them.

## 2. Judging
- Judge = **`gpt-4.1`**, temperature 0, `max_tokens=400`, n=1. **Frozen** (chosen by alignment with
  a human-verified gold set; human–judge Cohen's κ = 0.77, paper Appendix D).
- The judge compares the candidate `predict` with each reference formula file (docstrings and `fit()`
  stripped) for the **same functional form**: coefficients fold; exponents, term count, variables used
  and function family are structural.
- The judge is never the subject; a `gpt-4.1` subject is judged by `gpt-4.1-mini`.
- Empty completions are scored as misses without a judge call.

## 3. Verdict — frozen
`cold-recalled(task, model) = 1[ d1_hits >= 3 ]` on the task's **best-baseline** formula (the
reference the main benchmark anchors at S_N = 0.5). `result.json` also carries the legacy
`memorized/suspect/clean` label (D1 ≥3/5 → memorized; ≥1/5 → suspect) — not used by the paper.
Task tiers over a panel: **Moat** = no model cold-recalls, **Canon** = every model does, **Mid** = rest.

## 4. Inference configuration
| setting | value | note |
|---|---|---|
| reasoning effort | medium | OpenAI native path sends nothing (= provider default medium); OpenRouter legs send `reasoning.effort=medium` explicitly (other vendors' defaults are not medium); custom backends should do the same |
| temperature | 0.8 on chat models; not sent to reasoning models | APIs reject a temperature on reasoning endpoints |
| output ceiling | 28,800 tokens on reasoning models (`max_completion_tokens`), 900 on chat models | a cap, not a spend; binds on ~3% of samples, which can only push recall down |
| system prompt / extra text | none | one user message, verbatim |
| provider (OpenRouter) | pinned to the first-party host, `allow_fallbacks=false` | OpenRouter otherwise mixes resellers and quantizations |

## 5. Task sets and prompt versions
- `bench/scilaws118`: the 118 benchmark tasks (66 single-group, 52 multi-group), **metadata frozen 2026-07-28**.
  This is the version the six later legs of the paper's panel (gpt-5.4-mini, Claude Opus 4.8, Gemini 3.5 Flash,
  DeepSeek-V4 Pro, GLM-5.2, Qwen3.7-Max) ran on. The three June legs (gpt-4o-mini, gpt-5-mini, gpt-5.5) ran on an
  earlier metadata version: 85/118 prompts differ (50 only in the "typical range" formatting, 34 in description
  wording, 1 in the input list); the measured effect on recall is ~1 percentage point. Re-running those three on
  this frozen version (tier-2 experiment) makes the panel single-version.
- `bench/feynman100`: the 100 core AI-Feynman equations in the same schema; prompts unchanged since the paper runs.
- The probe reads only `metadata.yaml` and `formulas/*.py`; no data CSVs are shipped or needed.

## 6. What every run persists
```
runs/<bench>/<task>/<model>/
  result.json      per formula: d1_hits / d1_rate (/ d2_*), legacy verdict; judge model; prompt hash; parse-failure count
  d1_samples.txt   the five D1 completions, verbatim
  d1_raw/ judge_raw/ (d2_raw/)   every call: prompt, completions, usage, provider, billed cost when returned
```
`check_run.py` verifies coverage, empties, judge parse failures and prompt hashes before anything is analysed.
