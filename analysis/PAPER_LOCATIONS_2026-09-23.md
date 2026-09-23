# Memorization section — what changes when new models are added (pinpoint map)

Written 2026-09-23. Paper = Overleaf `6a16057c826cab34f286cca6`, `master` @ `3798564` (2026-09-23),
compiled locally to 41 pages; page/line numbers below are the PDF's own margin line numbers
(continuous, 001–~2100). Only **rendered** text is covered — commented-out and `\iffalse` blocks
are ignored on purpose (they are leftovers, not to be maintained).

> 中文摘要：加新模型后，记忆一节里会变的东西分两类。**A 类（只依赖我们自己的冷回忆审计）**：
> 47.5% moat / 11.9% canon / 34.7% 前沿 / "nine models" / "six vendors" / 附录 E 的 Table 3、Fig 11–13
> 和 E.1/E.4/E.5 的全部数字。**B 类（还依赖主实验 per-task 分数 S_N、S_S）**：Fig 5 三个 panel、
> 21%→54%、0.85→0.59、~0.08、0%→21%、r=0.97。**C 类（8-OpenAI 对照，冻结不动）**：Fig 2、E.6 relax、
> E.7 AI-Feynman、E.8 integrity、Fig 14/15 —— 除非你决定把 GPT-6 也接进 OpenAI 阶梯（需要多跑 100 题 Feynman）。
> κ 人评（§3.1 一句 + 附录 D）与模型数无关，不动。moat 和 canon 加模型只会变小或不变（单调性）。

---

## 0. Facts verified this session

| Fact | Evidence |
|---|---|
| Instrument = D1 cold recall only: 118 tasks × 5 samples (T=0.8 where supported), judge `gpt-4.1` T=0, cell memorized iff ≥3/5 match the **best-baseline** formula | `mem_pack/PROTOCOL.md`, `runner/pilot_v3.py`, `xvendor/recompute.py` (`MEMHIT=3`) |
| Paper panel = the 9 main-table models (`maintable9`): moat **56/118 = 47.5%**, canon **14 = 11.9%**, Mid 48, cells 1062, memorized 30.7%, Type I 23.2 / II 40.2 | `mem_pack/xvendor/xvendor_recompute.json` |
| Fig 5 (`figures/mem_figure5.pdf`) is byte-identical to `xvendor/fig6_variants/FIG6_final_V2.pdf`; Fig 11/Fig 2 identical to `xvendor/appendix_9model/*.pdf` | md5 match |
| Fig 5 inputs: `xvendor/fig5_data.json` == collaborator `hf_realsr_benchmark_v3/baseline_agent/model_task_scores.json` (real 1059/1059, struct 1055/1055 identical). Same file is in the public repo `SCILAWS-BENCH/baseline_agent/model_task_scores.json` (md5 identical) | recomputed |
| Fig 5 text numbers reproduce from those two files: GPT-5.5 >Ref% Canon 21 → Moat 54 (near-ref 57 → 14); pooled published-form 0.85 / 0.79 / 0.59; full-law 0.071 / 0.109 / 0.070; per-model full-law 0 (4o-mini) … 21.2 (5.5) | recomputed |
| **r = 0.97** (§3.2) is the correlation of per-model **means** of S_N and S_S (9 models: 0.968; the 5-model source gave 0.967). The sentence's own wording (">Ref% and full-law recovery … r=0.97") gives **0.905** on the same data. **r = 0.04** within GPT-5.5 is corr(S_N,S_S) over 118 tasks on an older data version; current data gives **0.075** | recomputed from `fig5_data.json`; origin `memorization_v4/new_findings_mining/NEW_FINDINGS.md` §N2 |
| moat and canon are **monotone non-increasing** in the model set (a task is moat iff *no* model recalls it; canon iff *all* do). Adding models can only shrink or keep both; Mid grows | definition |
| The public GitHub/HF release has **no memorization code** and **no reference-formula code** (`formulas/*.py` were not exported); the probe's judge needs those files. They live only in `mem_pack/runner/bench/` (3.6 MB, 118 tasks, 279 formula files) | `grep -ril memoriz SCILAWS-BENCH` → none; `build/hf_eval/tasks/*/` has no `formulas/` |
| `runner/bench` renders **exactly** the D1 prompts used by the 6 non-June legs (gpt-5.4-mini, opus, gemini, deepseek, glm, qwen): 118/118 identical each | compared against `d1_raw/*.json` prompts |
| The 3 OpenAI legs in the panel (gpt-4o-mini, gpt-5-mini, gpt-5.5) were run in **June** on an earlier metadata: 85/118 prompts differ from `runner/bench` (49 = range formatting only, 34 = description wording, 2 = input count). The current release tree (Sep) differs from `runner/bench` in the same way. Inputs annotated with "PRIMARY / DISTRACTOR / not used by any reference baseline" exist in only **7** runner tasks (6 of them Moat; recall on them ≈ 0 for every model) → immaterial | compared raw June prompts vs runner vs `hf_realsr_benchmark_v3/tasks` |
| Cost per new model: 590 subject completions + ≈1,395 judge calls (279 formulas × 5) ≈ **$4.7** of `gpt-4.1`; 2–4 h at 12 workers for reasoning models | `run_mem.py` `JUDGE_EST`, `PROTOCOL.md` |
| Compile note: after `3798564` the pipeline figure is commented out but §2.2's first sentence still says `Figure~\ref{fig:pipeline}` → renders "Figure ??" (p3, L138). Not mem-related. | `iclr2026_conference.log` |

---

## 1. Main text — every rendered spot that carries memorization information

Priority: **P1** = must change when any model is added · **P2** = changes or needs a check · **P3** = qualitative, only if the conclusion flips · **—** = frozen/unaffected.

| # | Where (PDF) | tex | What it says now | Depends on | Pri |
|---|---|---|---|---|---|
| 1 | p1 abstract, L028–029 | `iclr2026_conference.tex:171` | "memorization shapes whether models reproduce or move beyond published formulas" | conclusion | P3 |
| 2 | p2 intro, L089–090 | `1_introduction.tex:15` | "A cold-recall audit suggests that these problems are less memorized than canonical textbook equations (Section 2.2)" | Fig 2 (8-OpenAI) | P3 |
| 3 | p2 intro, L104–105 | `1_introduction.tex:27` | "we evaluate **nine** frontier LLMs … including GPT-5.5, Claude Opus 4.8, Gemini 3.5 Flash, and DeepSeek-V4 Pro" | main-table roster (collaborator) | P2 |
| 4 | p2–3 intro finding 2, L110–112 | `1_introduction.tex:33` | "Models use memory to reproduce known laws, but struggle to discover novel structure…" | conclusion | P3 |
| 5 | p3 §2.2, L151–161 + **Figure 2** (wrapfigure "Memorization comparison") | `3_problem_formulation.tex:43–50, 66` | "matched **eight-model** cold-recall test … eight OpenAI models common to both corpora, whereas … Section 3.3 and Appendix E reports the full **nine-model** results, same panel as the main Table 1" | Fig 2 = frozen 8-OpenAI vs AI-Feynman; the word "nine-model" must track the panel size | P1 (word) / — (figure, unless extending the OpenAI ladder) |
| 6 | p6 §3.1 "Scoring and Validation", L271–275 | `5_experiments.tex:30–33` | judge names (gpt-5.4-mini / gpt-4.1), κ 0.77/0.82/0.84 | κ study = frozen sample | — |
| 7 | p6 §3.2 **Table 1**, L305–316 | `6_discussion.tex:12–46` | main results, 9 rows | collaborator | P1 (entangled: row order = the weak→strong order used by Fig 5(a,c), Fig 12, Table 3) |
| 8 | p6 §3.2 "High fit does not reliably imply…", L288–297 | `6_discussion.tex:9` | "Across models, >Ref% and full-law recovery rise together, Pearson **r=0.97** (Figure 5(a,c)). Within GPT-5.5 … **r=0.04**"; "all **nine** models submit physically valid forms" (proton) | main-exp S_N/S_S per task; **definition mismatch** (see §0) | P1 |
| 9 | p7 §3.3 opening, L343–347 | `6_discussion.tex:66–68` | qualitative | — | P3 |
| 10 | p7 §3.3 "Memorization audit.", L348–355 | `6_discussion.tex:77–83` | "only **11.9%** recalled by all **nine** models while **47.5%** by none"; "Even **GPT-5.5** can only cold-recall **34.7%**"; "stable across the **six** audited vendors" | audit only | **P1 ⭐** |
| 11 | p8 **Figure 5** (a)(b)(c), L410–431 | `6_discussion.tex:86–91`, `figures/mem_figure5.pdf` | (a) per tier × model below/comparable/above baseline; (b) published-form vs full-law by tier; (c) full-law per model, all vs Moat. Caption has no numbers | audit (tiers) **+** main-exp S_N, S_S of every model incl. the new one | **P1 ⭐** |
| 12 | p7 §3.3 "On Real, models beat…", L356–366 | `6_discussion.tex:94–95` | "Canon (**14**), Mid (**48**), Moat (**56**)"; "GPT-5.5's >Ref% rises from **21%** to **54%**"; examples gravity (Canon) / DNA-melting (Moat) | tiers + main-exp S_N; frontier identity; example tasks must keep their tier | **P1 ⭐** |
| 13 | p7 §3.3 "On Parallel, memorization helps…", L367–377 | `6_discussion.tex:99–100` | "published-form recovery **0.85** Canon → **0.59** Moat"; "full-law ≈ **0.08** at every tier"; "**0%** GPT-4o-mini → **21%** GPT-5.5"; "Beer–Lambert: all **nine** models recover the textbook law" | tiers + main-exp S_S; frontier identity; Beer–Lambert must stay Canon | **P1 ⭐** |
| 14 | p10 §5 conclusion, L485–492 | `7_conclusion.tex:5,7` | qualitative | — | P3 |

## 2. Appendix

| # | Where (PDF) | tex | What it says now | Depends on | Pri |
|---|---|---|---|---|---|
| 15 | p23–26 **App D** (Table 2, Figs 8–10) | `appendix_human_alignment.tex` | κ study; p23 L1218 "15 from each of the **nine** main-table models, for 135 items" | frozen sample; only the wording implies the panel size | P3 (wording) |
| 16 | p27 E.1, L1417–1419 | `appendix_memorization.tex:24–26` | "**nine** main-table models spanning **six** vendors, giving **1062** task-model cells" | audit | P1 |
| 17 | p27 E.2–E.3 | `…:28–88` | protocol, prompt, judge, ≥3/5 rule | — | — |
| 18 | p28 E.4 first para, L1495–1500 | `…:93–100` | "Across 118 tasks and **nine** models, **30.7%** … **69.3%**"; "from **15.3%** (gpt-4o-mini) to the mid-30%s"; "glm-5.2 and qwen3.7-max each **35.6%**, slightly above gpt-5.5 at **34.7%**"; "highest … does not cold-recall roughly two-thirds" | audit | P1 |
| 19 | p28 **Table 3**, L1459–1473 | `…:102–129` | 9 rows (order = weak→strong by mean S_N) + "All cells 30.7 / 69.3 / 1062" | audit + Table 1 order | P1 |
| 20 | p28 **Figure 11** recall spectrum, L1474–1490 | `…:143–150`, `figures/memorization_recall_spectrum.pdf` | bars 0…9 = [56,7,7,10,5,2,5,7,5,14]; x-label "(of 9)"; caption "56-task moat / 14-task canon" | audit | P1 |
| 21 | p28 E.4 second para, L1501–1508 | `…:131–141` | "**56** tasks, **47.5%** … **14** tasks, **11.9%**, cold-recalled by all **nine** models" + the **named list of 14 canon tasks** | audit (list can only shrink) | P1 |
| 22 | p28–29 E.5 + **Fig 12** (by type) + **Fig 13** (by domain), L1511–1552 | `…:152–186`, `figures/memorization_ladder_bytype.pdf`, `figures/memorization_domain.pdf` | "**40.2%** vs **23.2%**"; domain means 20.5 / 43.4 / 40.5; frontier (gpt-5.5) 59.1 on Ecology; Fig 13 caption "mean over the nine audited models; dots show gpt-5.5" | audit; Fig 12 x-axis order = Table 1 order; "frontier" dot may change model | P2 |
| 23 | p29 E.6 + **Fig 14** relax, L1555–1565 | `…:188–216`, `figures/memorization_relax.pdf` | 8-OpenAI: +0.85 pp, 16 tasks, 8 cells / 3 tasks flip | frozen 8-OpenAI | — (unless extending) |
| 24 | p30 E.7 AI-Feynman, L1585–1595 | `…:218–235` | 8-OpenAI: 55.9 vs 26.8; frontier 71.0 vs 34.7, gap 36.3 | frozen 8-OpenAI | — (unless extending) |
| 25 | p30 E.8 integrity + **Fig 15** composition, L1567–1610 | `…:237–268`, `figures/memorization_composition.pdf` | 8-OpenAI ladder, 21,888 calls, 0% empty | frozen 8-OpenAI | — (unless extending) |
| 26 | p35 App G **Table 4** + case texts | `appendix_case_studies.tex:17–45, 247, 316` | "eight tasks … nine-model panel", "7 of 9 models", "all 9 models", "best of the nine" | main experiment (collaborator) | P2 (theirs) |
| 27 | Homepage `SCILAWS-BENCH/docs/index.html` L400–405, `docs/assets/data.js` (`"models": 9`, leaderboard), `docs/assets/figures/mem_figure5.webp` | release repo | 11.9 / 47.5 / 21→54 / 0.59→0.85 / 0.08 / "all nine" | same as #10–13 | P2 |

### What does NOT change
κ validation (§3.1 sentence, App D); protocol text (E.2/E.3); the 8-OpenAI controls (Fig 2, E.6, E.7, E.8, Fig 14, Fig 15) — they are explicitly scoped to "the eight OpenAI models" in the text. If the new model is an OpenAI model and you *want* to extend the OpenAI ladder, all of E.6–E.8 + Fig 2/14/15 + §2.2 + intro L089 become live again and need a Feynman leg (100 prompts × 5) plus the relax/composition re-runs.

---

## 3. What must be produced for each new model

| Input | Who | Used by |
|---|---|---|
| **D1 samples**: 118 prompts × 5 completions (temperature 0.8 where the API allows, otherwise provider default; reasoning effort medium; max output 28,800 tokens; single user message, no system prompt, no data) | the person running the model, with their own API layer | everything in §1–2 marked "audit" |
| **Judging** (gpt-4.1, ≈1,395 calls, ≈$4.7) | us | same |
| **Per-task S_N and S_S** of the new model, in `model_task_scores.json` format (`real_numeric`, `parallel_structure` keyed by task) | main-experiment collaborators | Fig 5 (a)(b)(c); #8, #12, #13; row order of Table 3 / Fig 12 |
| *(optional, OpenAI models only)* Feynman D1 samples: 100 prompts × 5 | runner | Fig 2, E.7; then E.6/E.8 re-runs |

Decision to record: new legs use the **frozen `runner/bench` prompts** (the ruler of 6 of the 9 panel models). Optionally re-run gpt-4o-mini / gpt-5-mini / gpt-5.5 on the same prompts (≈$1 + $5 + ~$60) so the whole panel is one prompt version; the observed drift is cosmetic for recall (see §0).

---

## 4. Code map (current state)

| Step | File | Hard-coded model lists to parametrize |
|---|---|---|
| probe (subject + judge + verdict) | `mem_pack/runner/{run_mem.py,pilot_v3.py,client.py,models.json,tasks_118.json,bench/}` | `models.json`, `run_mem.py` `ALIAS`/`EST` |
| raw results | `mem_pack/results/runs/<run>/<task>/<model>/result.json` (+ `merge_rows.py`) | — |
| numbers | `mem_pack/xvendor/recompute.py` → `xvendor_recompute.json` (gates on 8-OpenAI) | `LADDER8`, `NONOAI_COMPLETE`, `QWEN`, `CLEAN`, `VENDOR`, `panels{}` |
| Fig 11 / 12 / 13 | `gen_9model_appendix_data.py` → `v3_d1only_data_9model.json` → `make_9model_appendix_figs.py` | `MT9`, `F2R`, `SHORT` |
| Fig 5 | `fig6_final.py` (reads `fig5_data.json` + `xvendor_recompute.json`) | `SH` |
| `fig5_data.json` | **no builder on disk** — it is a flat dump of `model_task_scores.json`; needs a 20-line script | display names |
| Fig 2 | `f0_feynman_compact.py` (8-OpenAI `figures/data/v3_d1only_data.json`) | frozen |
| Fig 14 / 15 | `figures/v3_d1only_figs.py` via `figures/make_figures.py` (8-OpenAI) | frozen |
| public repo | `SCILAWS-BENCH/` has **no** memorization code | — |

## 5. Plan (agreed direction: the runner uses their own API layer → we ship a *prompt contract*, not a backend)

Kit for the runner (zip or a `memorization/` folder in the GitHub repo):
`README_runbook_zh.md` (one page) · `prompts_scilaws118.json` (task_id → exact prompt text, pre-rendered from the frozen bench) · `prompts_feynman100.json` (optional) · `sampling_spec.json` · `output_schema.json` + `example_output.json` · `check_output.py` (completeness / format self-check) · `run_mem.py` (reference runner, optional).
They return `samples_<model>.json` (5 raw completions per task + model id, finish reasons, usage). No judging on their side, no bench, no formula code (prompts contain only public metadata text).

Our side, one command (`mem_pack/new_model_update/run_update.py`): judge → `results/runs/<date>/…/result.json` → `recompute.py` (new panel, double gate: 8-OpenAI numbers *and* current `maintable9` numbers) → `fig5_data.json` from the new `model_task_scores.json` → Fig 5 / 11 / 12 / 13 → `paper_numbers.json` + `PAPER_DELTA.md` listing every item #1–#27 with old → new value, regenerated Table 3 rows, tier-membership diff, sanity checks (monotone moat/canon; gravity∈Canon, DNA∈Moat, Beer–Lambert∈Canon; frontier identity; Table 1 order). Manual: paste into tex; collaborator updates Table 1 / App G / B.1 roster; homepage.
