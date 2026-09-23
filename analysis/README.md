# analysis/ — our side (no API calls)

```bash
bash analysis/run_all.sh          # recompute -> fig5 data -> figures -> PAPER_DELTA.md   (all under analysis/out/)
```

Inputs it uses:
- `data/openai8_scilaws_rows.json`, `data/openai8_feynman_rows.json` — the June 8-OpenAI rows (paper controls).
- `data/runs_2026-07/**/result.json` — the six later legs of the paper's 9-model panel.
- `runs/scilaws/**/result.json`, `runs/feynman/**/result.json` — **new legs** (unzip the collaborator's `mem_runs.zip` at the kit root).
- `data/model_task_scores.json` — the paper's main-experiment per-task S_N / S_S (Fig 5). For a **new model**, drop the
  collaborators' scores for it under `runs/scores/<anything>.json` in the same schema
  (`{"tasks": {<task>: {"type":..., "real_numeric": {<main_exp_id>: ...}, "parallel_structure": {<main_exp_id>: ...}}}}`).
- `analysis/models_meta.json` — add one entry per new model (vendor, short label, `main_exp_id`, and `openai_ladder_pos`
  for OpenAI models so they appear on the AI-Feynman ladder).

What `recompute.py` does: gates itself by reproducing the committed paper numbers (8-OpenAI: moat 63 / canon 12 /
26.8%; 9-model panel: moat 56 / canon 14 / 30.7%), then builds the **current** panel = the 9 main-table models + every
complete new model; a new leg of a model already in the panel (the tier-2 re-runs) **replaces** its June rows.
`PAPER_DELTA.md` lists every paper location (PDF page/line + tex line) with committed vs recomputed value, the new
Table 3 rows, the tier-membership diff, the AI-Feynman ladder incl. new OpenAI models, and sanity checks.

Known 0.1–0.2 pp difference: the committed Fig 5(c) was built from a 1,059-row file that dropped three (task, model)
cells with a Parallel score but no Real score; `build_fig5_data.py` keeps them (qwen 10.5→10.3, glm 12.8→12.7 full-law %).
All numbers quoted in the paper text reproduce exactly.
