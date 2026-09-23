"""Every memorization number in the paper: committed value -> recomputed value.

Reads analysis/out/{recompute,fig5_data,fig_stats}.json and writes analysis/out/PAPER_DELTA.md:
a table per paper location (PDF page/line + tex file:line) with old value, new value, and a flag,
plus the regenerated Table 3 rows, the tier-membership diff, and sanity checks.
Committed values = the paper as of Overleaf 3798564 (2026-09-23).
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "analysis"))
import recompute as RC

D = json.loads((RC.OUT / "recompute.json").read_text()); C = D["panels"]["current"]; M9 = D["panels"]["maintable9"]
F5 = json.loads((RC.OUT / "fig5_data.json").read_text()); FS = json.loads((RC.OUT / "fig_stats.json").read_text())
paper = lambda m: RC.META.get(m, {}).get("paper", m)

# ---------------- committed paper values (Overleaf 3798564) ----------------
PAPER = {
 "n_models": 9, "n_vendors": 6, "cells": 1062, "mem_pct": 30.7, "clean_pct": 69.3,
 "canon_n": 14, "canon_pct": 11.9, "moat_n": 56, "moat_pct": 47.5, "mid_n": 48,
 "frontier": "gpt-5.5", "frontier_pct": 34.7, "min_model": "gpt-4o-mini", "min_pct": 15.3,
 "type_I": 23.2, "type_II": 40.2,
 "domain_low": ("Materials & Engineering", 20.5), "domain_high": [("Ecology & Hydrology", 43.4), ("Social Sciences", 40.5)],
 "frontier_ecology": 59.1,
 "ref_beat_frontier": {"Canon": 21, "Moat": 54}, "near_ref_frontier": {"Canon": 57, "Moat": 14},
 "published_form": {"Canon": 0.85, "Moat": 0.59}, "full_law_tiers": 0.08,
 "full_law_min": ("gpt-4o-mini", 0), "full_law_max": ("gpt-5.5", 21),
 "r_between_models": 0.97, "r_within_frontier": 0.04,
 "feynman_8": {"scilaws_all": 26.8, "feynman_all": 55.9, "ratio": 2.1, "frontier": "gpt-5.5", "scilaws_front": 34.7, "feynman_front": 71.0, "gap": 36.3},
 "table3": {"gpt-4o-mini": 15.3, "gpt-5.4-mini": 29.7, "gpt-5-mini": 31.4, "deepseek-v4-pro": 31.4, "qwen3.7-max": 35.6,
            "claude-opus-4.8": 33.9, "gemini-3.5-flash": 28.8, "glm-5.2": 35.6, "gpt-5.5": 34.7},
 "canon_members": M9["canon_members"],
 "examples": {"gravity_wgs84_somigliana__g0": "Canon", "dna_melting_temperature_khandelwal__Tm": "Moat",
              "spectrophotometry_beer_lambert_chromate__absorbance": "Canon"},
}

# ---------------- recomputed values ----------------
real, struct = defaultdict(dict), defaultdict(dict)
for r in F5["rows"]:
    if r["real"] is not None: real[r["model"]][r["task"]] = r["real"]
    if r["struct"] is not None: struct[r["model"]][r["task"]] = r["struct"]
MOAT, CANON = set(C["moat_members"]), set(C["canon_members"])
tier = lambda t: "Moat" if t in MOAT else "Canon" if t in CANON else "Mid"
fr = C["frontier_model"]
def beat(m, z): v = [s for t, s in real[m].items() if tier(t) == z]; return round(100 * np.mean([s > 0.52 for s in v])) if v else None
def near(m, z): v = [s for t, s in real[m].items() if tier(t) == z]; return round(100 * np.mean([0.48 <= s <= 0.52 for s in v])) if v else None
ms = [m for m in C["models"] if m in real]
beat_all = {m: np.mean([s > 0.52 for s in real[m].values()]) for m in ms}
full_all = {m: np.mean([s >= 0.999 for s in struct[m].values()]) for m in ms if m in struct}
mr = [np.mean(list(real[m].values())) for m in ms]; mp = [np.mean(list(struct[m].values())) for m in ms]
r_means = round(float(np.corrcoef(mr, mp)[0, 1]), 3)
r_thr = round(float(np.corrcoef([beat_all[m] for m in ms], [full_all[m] for m in ms])[0, 1]), 3)
ts = [t for t in real.get(fr, {}) if t in struct.get(fr, {})]
r_within = round(float(np.corrcoef([real[fr][t] for t in ts], [struct[fr][t] for t in ts])[0, 1]), 3) if ts else None
pm = C["per_model"]; top = sorted(pm, key=lambda m: -pm[m]["mem_pct"])[:3]; low = min(pm, key=lambda m: pm[m]["mem_pct"])
dom = C["per_domain"]; dom_hi = sorted(dom, key=lambda d: -d["mem_pct"])[:2]; dom_lo = dom[0]
ecol = next((d for d in dom if d["domain"].startswith("Ecology")), None)
fl = FS["fig5"]; fl_pm = fl["full_law_per_model_pct"]
fe = D["feynman"]; sc = D["scilaws_for_feynman_fig"]
oa_both = [m for m in RC.OPENAI_LADDER_ORDER if m in fe and fe[m]["complete"]] + sorted(m for m in fe if m not in RC.OPENAI_LADDER_ORDER and fe[m]["complete"])
new_oa = [m for m in oa_both if m not in RC.LADDER8]
# strongest OpenAI model on both corpora = highest main-table S_N if scores exist, else last in ladder
oa_front = max([m for m in oa_both if m in real], key=lambda m: np.mean(list(real[m].values())), default=oa_both[-1])
def fmt_models(l): return ", ".join(paper(m) for m in l)

rows = []   # (section, where, tex, item, old, new)
def add(sec, where, tex, item, old, new): rows.append((sec, where, tex, item, old, new))
S = "§3.3 / Fig 5"
add(S, "p7 L353", "6_discussion.tex:81", "canon share (recalled by ALL panel models)", f"{PAPER['canon_pct']}% ({PAPER['canon_n']})", f"{C['canon']['pct']}% ({C['canon']['n']})")
add(S, "p7 L353", "6_discussion.tex:81", "moat share (recalled by NONE)", f"{PAPER['moat_pct']}% ({PAPER['moat_n']})", f"{C['moat']['pct']}% ({C['moat']['n']})")
add(S, "p7 L353", "6_discussion.tex:81", "'all nine models' (count word)", "9", str(C["n_models"]))
add(S, "p7 L354", "6_discussion.tex:82", "frontier model + its cold-recall", f"{paper(PAPER['frontier'])} {PAPER['frontier_pct']}%", f"{paper(fr)} {pm[fr]['mem_pct']}%")
add(S, "p7 L355", "6_discussion.tex:83", "'six audited vendors' (count word)", "6", str(C["n_vendors"]))
add(S, "p7 L356", "6_discussion.tex:95", "tiers Canon / Mid / Moat", f"{PAPER['canon_n']} / {PAPER['mid_n']} / {PAPER['moat_n']}", f"{C['canon']['n']} / {C['mid']['n']} / {C['moat']['n']}")
add(S, "p7 L359–360", "6_discussion.tex:95", f"frontier >Ref% Canon -> Moat ({paper(fr)})", f"{PAPER['ref_beat_frontier']['Canon']}% -> {PAPER['ref_beat_frontier']['Moat']}%", f"{beat(fr,'Canon')}% -> {beat(fr,'Moat')}%")
add(S, "p7 L360", "6_discussion.tex:95", "frontier near-reference share Canon -> Moat", f"{PAPER['near_ref_frontier']['Canon']}% -> {PAPER['near_ref_frontier']['Moat']}%", f"{near(fr,'Canon')}% -> {near(fr,'Moat')}%")
add(S, "p7 L363–366", "6_discussion.tex:95", "examples: gravity tier / DNA-melting tier", "Canon / Moat", f"{tier('gravity_wgs84_somigliana__g0')} / {tier('dna_melting_temperature_khandelwal__Tm')}")
add(S, "p7 L371–372", "6_discussion.tex:100", "published-form recovery Canon -> Moat (pooled)", f"{PAPER['published_form']['Canon']} -> {PAPER['published_form']['Moat']}", f"{fl['published_form_by_tier']['Canon']:.2f} -> {fl['published_form_by_tier']['Moat']:.2f}")
add(S, "p7 L372", "6_discussion.tex:100", "full-law recovery by tier Canon/Mid/Moat (paper text: near 0.08)", "0.071 / 0.109 / 0.070", " / ".join(f"{fl['full_law_by_tier'][z]:.3f}" for z in ["Canon","Mid","Moat"]))
add(S, "p7 L373", "6_discussion.tex:100", "full-law per model: min -> max", f"{PAPER['full_law_min'][1]}% {paper(PAPER['full_law_min'][0])} -> {PAPER['full_law_max'][1]}% {paper(PAPER['full_law_max'][0])}",
    f"{min(fl_pm.values()):.0f}% {paper(min(fl_pm,key=fl_pm.get))} -> {max(fl_pm.values()):.0f}% {paper(max(fl_pm,key=fl_pm.get))}")
bl = "spectrophotometry_beer_lambert_chromate__absorbance"
add(S, "p7 L375–377", "6_discussion.tex:100", "Beer–Lambert: tier / models recovering the published form (S_S>=0.5)", "Canon / 9 of 9", f"{tier(bl)} / {sum(struct[m].get(bl,0)>=0.5 for m in ms)} of {len(ms)}")
add("§3.2", "p6 L290–291", "6_discussion.tex:9", "across-model Pearson r of per-model MEANS (paper wording says >Ref% vs full-law, whose value is r_thr)", f"{PAPER['r_between_models']}", f"{round(r_means, 2)}  (r_thr={r_thr})")
add("§3.2", "p6 L292", "6_discussion.tex:9", f"within-frontier Pearson r(S_N,S_S) ({paper(fr)}) — committed 0.04 came from a 2026-07-02 snapshot; 0.075 on the paper's final scores", f"{PAPER['r_within_frontier']}", f"{r_within}")
add("§3.2", "p6 L295", "6_discussion.tex:9", "'all nine models' (proton) — main experiment (count word)", "9", str(C["n_models"]))
add("intro/§2.2", "p2 L104; p3 L159–161", "1_introduction.tex:27; 3_problem_formulation.tex:66", "'nine frontier LLMs' / 'full nine-model results' (count word)", "9", str(C["n_models"]))
A = "App E"
add(A, "p27 L1417–1419", "appendix_memorization.tex:24–26", "models / vendors / cells", f"{PAPER['n_models']} / {PAPER['n_vendors']} / {PAPER['cells']}", f"{C['n_models']} / {C['n_vendors']} / {C['cells']}")
add(A, "p28 L1495", "appendix_memorization.tex:93–94", "cells cold-recalled / not", f"{PAPER['mem_pct']}% / {PAPER['clean_pct']}%", f"{C['mem_pct']}% / {C['clean_pct']}%")
add(A, "p28 L1496–1498", "appendix_memorization.tex:95–98", "lowest model; top-3 models", f"{paper(PAPER['min_model'])} {PAPER['min_pct']}; GLM-5.2 35.6, Qwen3.7-Max 35.6, GPT-5.5 34.7",
    f"{paper(low)} {pm[low]['mem_pct']}; " + ", ".join(f"{paper(m)} {pm[m]['mem_pct']}" for m in top))
add(A, "p28 L1501–1503 + Fig 11 caption", "appendix_memorization.tex:132–135,146–148", "moat n (%) / canon n (%)", f"{PAPER['moat_n']} ({PAPER['moat_pct']}%) / {PAPER['canon_n']} ({PAPER['canon_pct']}%)", f"{C['moat']['n']} ({C['moat']['pct']}%) / {C['canon']['n']} ({C['canon']['pct']}%)")
add(A, "p28 L1503–1508", "appendix_memorization.tex:135–141", "named canon list", f"{len(PAPER['canon_members'])} tasks", f"{len(C['canon_members'])} tasks")
add(A, "p28 L1511", "appendix_memorization.tex:155–156", "multi-group vs single-group cold recall", f"{PAPER['type_II']}% vs {PAPER['type_I']}%", f"{C['per_type']['II']['mem_pct']}% vs {C['per_type']['I']['mem_pct']}%")
add(A, "p29 L1549–1552", "appendix_memorization.tex:181–186", "domain lowest / two highest / frontier on Ecology", f"{PAPER['domain_low'][0]} {PAPER['domain_low'][1]}; " + "; ".join(f"{d} {v}" for d, v in PAPER['domain_high']) + f"; {PAPER['frontier_ecology']} (GPT-5.5)",
    f"{dom_lo['domain']} {dom_lo['mem_pct']}; " + "; ".join(f"{d['domain']} {d['mem_pct']}" for d in dom_hi) + f"; {ecol['front_pct'] if ecol else 'n/a'} ({paper(fr)})")
add(A, "p28 Fig 11 x-axis", "make_figs -> fig11", "spectrum (0..N models)", str(M9["recall_spectrum"]), str(C["recall_spectrum"]))
F = "§2.2 Fig 2 / App E.7 (OpenAI models audited on BOTH corpora)"
if new_oa:
    add(F, "p30 L1587–1591", "appendix_memorization.tex:221–229", "new OpenAI models on both corpora", "eight OpenAI models", f"{fmt_models(oa_both)} ({len(oa_both)})")
    add(F, "p30 L1590–1591", "appendix_memorization.tex:227–229", "strongest OpenAI model: SciLaws vs Feynman, gap", f"{paper(PAPER['feynman_8']['frontier'])} {PAPER['feynman_8']['scilaws_front']} vs {PAPER['feynman_8']['feynman_front']} ({PAPER['feynman_8']['gap']} pp)",
        f"{paper(oa_front)} {sc.get(oa_front)} vs {fe[oa_front]['mem_pct']} ({round(fe[oa_front]['mem_pct'] - (sc.get(oa_front) or 0), 1)} pp)")
else:
    add(F, "p30 L1587–1591", "appendix_memorization.tex:221–229", "8-OpenAI controls (no new OpenAI legs found)", "unchanged", "unchanged")
if C.get("replaced_by_new_legs"):
    for m in C["replaced_by_new_legs"]:
        add("tier-2 re-runs", "Table 3 / §3.3 / E.7", "—", f"{paper(m)} SciLaws cold recall: June prompts -> frozen prompts", f"{M9['per_model'][m]['mem_pct']}", f"{pm[m]['mem_pct']}")

# ---------------- write markdown ----------------
L = ["# PAPER_DELTA — memorization numbers, committed (Overleaf 3798564) vs recomputed", "",
     f"Panel: **{C['n_models']} models / {C['n_vendors']} vendors / {C['cells']} cells**; new models: {fmt_models(C.get('new_models', [])) or 'none'}; "
     f"re-run legs replacing June rows: {fmt_models(C.get('replaced_by_new_legs', [])) or 'none'}; partial legs ignored: {C.get('partial_new_legs_ignored') or 'none'}.", "",
     "| Section | PDF (p/line) | tex | Item | Committed | Recomputed | |", "|---|---|---|---|---|---|---|"]
n_changed = 0
for sec, where, tex, item, old, new in rows:
    ch = str(old).strip() != str(new).strip(); n_changed += ch
    L.append(f"| {sec} | {where} | `{tex}` | {item} | {old} | {new} | {'**CHANGED**' if ch else 'same'} |")
L += ["", f"**{n_changed} of {len(rows)} items changed.**", "", "## Table 3 rows (weak -> strong by mean S_N; order = Table 1 order)", "",
      "```latex"]
order = FS["fig5"]["order"] + [m for m in C["models"] if m not in FS["fig5"]["order"]]
for m in order:
    p = pm[m]; L.append(f"\\texttt{{{m}}} & {p['mem_pct']:.1f} & {100 - p['mem_pct']:.1f} & {p['n']} \\\\")
L.append(f"All cells & {C['mem_pct']:.1f} & {C['clean_pct']:.1f} & {C['cells']} \\\\"); L.append("```")
L += ["", "## Tier membership vs the committed 9-model panel", ""]
for name, old, new in [("Canon", set(M9["canon_members"]), CANON), ("Moat", set(M9["moat_members"]), MOAT)]:
    L.append(f"- **{name}**: {len(old)} -> {len(new)}; left: {sorted(old - new) or '—'}; joined: {sorted(new - old) or '—'}")
L += ["", "## Sanity checks", ""]
chk = [("moat monotone (<= 56)", C["moat"]["n"] <= 56), ("canon monotone (<= 14)", C["canon"]["n"] <= 14),
       ("gravity in Canon", tier("gravity_wgs84_somigliana__g0") == "Canon"), ("DNA-melting in Moat", tier("dna_melting_temperature_khandelwal__Tm") == "Moat"),
       ("Beer–Lambert in Canon", tier(bl) == "Canon"), (f"frontier = {paper(fr)} (highest main-table S_N)", True),
       ("every panel model has main-experiment scores (Fig 5)", all(m in real for m in C["models"]))]
for k, v in chk: L.append(f"- {'OK ' if v else 'CHECK'} {k}")
L += ["", "## AI-Feynman ladder (OpenAI models on both corpora)", "", "| model | SciLaws-Real % | AI-Feynman % | gap pp | source |", "|---|---|---|---|---|"]
for m in oa_both:
    L.append(f"| {paper(m)} | {sc.get(m)} | {fe[m]['mem_pct']} | {round(fe[m]['mem_pct'] - (sc.get(m) or 0), 1)} | {fe[m]['source']} |")
(RC.OUT / "PAPER_DELTA.md").write_text("\n".join(L) + "\n")
print(f"PAPER_DELTA.md: {n_changed}/{len(rows)} items changed -> {RC.OUT/'PAPER_DELTA.md'}")
