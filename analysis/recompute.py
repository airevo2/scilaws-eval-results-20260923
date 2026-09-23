"""Recompute every memorization number in the paper, for the current panel + any new legs.

Inputs (all relative to the kit root):
  data/openai8_scilaws_rows.json     8-OpenAI June rows (task, formula, model, d1_hits)   [paper's 8-OpenAI controls]
  data/runs_2026-07/**/result.json   the 6 later legs (gpt-5.4-mini + 5 non-OpenAI vendors)
  runs/scilaws/**/result.json        NEW legs returned by the collaborator (any number of models)
  data/openai8_feynman_rows.json     8-OpenAI June AI-Feynman rows
  runs/feynman/**/result.json        NEW AI-Feynman legs (OpenAI models)
  data/best_baseline_map.json, data/domains.json

Rule (frozen): a (task, model) cell is cold-recalled iff d1_hits >= 3 on the task's best-baseline formula.
Panels:
  openai8     the 8 June OpenAI models                      -> gate: moat 63 / canon 12 / mem 26.8
  maintable9  the paper's 9 main-table models (Sep 2026)    -> gate: moat 56 / canon 14 / mem 30.7
  current     maintable9 + every complete NEW model in runs/scilaws; a new leg of a model that is
              already in the panel REPLACES its old rows (e.g. the 3 OpenAI re-runs on frozen prompts)
Usage:  python3 analysis/recompute.py [--panel id1,id2,...]      -> analysis/out/recompute.json
"""
from __future__ import annotations
import argparse, collections, glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "out"; OUT.mkdir(exist_ok=True, parents=True)
META = {k: v for k, v in json.loads((ROOT / "analysis" / "models_meta.json").read_text()).items() if not k.startswith("_")}
ALIAS = {a: k for k, v in META.items() for a in v.get("aliases", [])}
MEMHIT = 3
EXCLUDE = {"qg_turbulence_subgrid_pyqg__Sq"}          # dropped from the benchmark (119 -> 118)
LADDER8 = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-5.5"]   # the June 8-OpenAI ladder (fixed)
OPENAI_LADDER_ORDER = [k for k, v in sorted(META.items(), key=lambda kv: (kv[1]["openai_ladder_pos"] or 99)) if v["openai_ladder_pos"]]  # + new OpenAI models, for the Feynman figure
MAINTABLE9 = ["gpt-4o-mini", "gpt-5-mini", "gpt-5.5", "gpt-5.4-mini", "claude-opus-4.8",
              "gemini-3.5-flash", "deepseek-v4-pro", "glm-5.2", "qwen3.7-max"]
GATES = {"openai8": {"moat_n": 63, "canon_n": 12, "mem": 26.8, "tasks": 118},
         "maintable9": {"moat_n": 56, "canon_n": 14, "mem": 30.7, "tasks": 118}}


def canon_id(raw: str) -> str:
    return ALIAS.get(raw, raw.split("/")[-1] if "/" in raw else raw)


def vendor(m: str) -> str:
    return META.get(m, {}).get("vendor", "?")


def rows_from_result_dirs(pattern: str, tag: str) -> dict[str, list[dict]]:
    """Formula-level rows per canonical model id from result.json files."""
    out: dict[str, dict[tuple, dict]] = collections.defaultdict(dict)
    for rj in glob.glob(pattern, recursive=True):
        try:
            d = json.loads(Path(rj).read_text())
        except Exception:
            continue
        if d.get("status") or not d.get("task"):
            continue
        m = canon_id(d["model"]); t = d["task"]
        if t in EXCLUDE:
            continue
        for f in d.get("formulas", []):
            out[m][(t, f["formula"])] = {"task": t, "formula": f["formula"], "model": m, "type": d.get("type"),
                                         "d1_hits": f.get("d1_hits"), "source": tag}
    return {m: list(v.values()) for m, v in out.items()}


def load_all():
    june = collections.defaultdict(list)
    for r in json.loads((ROOT / "data" / "openai8_scilaws_rows.json").read_text()):
        if r["task"] in EXCLUDE: continue
        june[canon_id(r["model"])].append({"task": r["task"], "formula": r["formula"], "model": canon_id(r["model"]),
                                           "type": r["type"], "d1_hits": r["d1_hits"], "source": "june2026_openai8"})
    july = rows_from_result_dirs(str(ROOT / "data" / "runs_2026-07" / "**" / "result.json"), "july2026_legs")
    new = rows_from_result_dirs(str(ROOT / "runs" / "scilaws" / "**" / "result.json"), "new_leg")
    return june, july, new


def collapse(rows_by_model: dict[str, list[dict]], bb: dict) -> dict[str, dict[str, dict]]:
    """cell[task][model] = best-baseline row (only tasks in the map)."""
    cell: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for m, rows in rows_by_model.items():
        idx = {(r["task"], r["formula"]): r for r in rows}
        for t, b in bb.items():
            r = idx.get((t, b["formula"]))
            if r is not None:
                cell[t][m] = r
    return cell


def mem(r) -> bool:
    return (r["d1_hits"] or 0) >= MEMHIT


def panel_stats(cell, models, task2dom, frontier=None) -> dict:
    full = {t: mm for t, mm in cell.items() if all(m in mm for m in models)}
    NT = len(full); cells = [mm[m] for mm in full.values() for m in models]
    ncell = len(cells) or 1; memc = sum(mem(r) for r in cells)
    moat = sorted(t for t, mm in full.items() if all(not mem(mm[m]) for m in models))
    canon = sorted(t for t, mm in full.items() if all(mem(mm[m]) for m in models))
    per_model = {}
    for m in models:
        rs = [mm[m] for mm in full.values()]; n = len(rs) or 1
        per_model[m] = {"mem_pct": round(100 * sum(mem(r) for r in rs) / n, 1), "n": len(rs), "vendor": vendor(m),
                        "source": rs[0]["source"] if rs else None}
    isII = lambda r: "II" in str(r.get("type", ""))
    per_type, ladder_by_type = {}, {"I": {}, "II": {}}
    for ty, flag in [("I", False), ("II", True)]:
        sub = [r for r in cells if isII(r) == flag]; n = len(sub) or 1
        per_type[ty] = {"mem_pct": round(100 * sum(mem(r) for r in sub) / n, 1), "n": len(sub)}
        for m in models:
            rs = [mm[m] for mm in full.values() if isII(mm[m]) == flag]
            ladder_by_type[ty][m] = round(100 * sum(mem(r) for r in rs) / (len(rs) or 1), 2)
    spec = collections.Counter(sum(mem(mm[m]) for m in models) for mm in full.values())
    frontier = frontier or (max(per_model, key=lambda m: per_model[m]["mem_pct"]))
    doms = collections.defaultdict(list)
    for t, mm in full.items(): doms[task2dom.get(t, "?")].append(mm)
    per_domain = []
    for d, mms in doms.items():
        cs = [mm[m] for mm in mms for m in models]; fr = [mm[frontier] for mm in mms]
        per_domain.append({"domain": d, "tasks": len(mms), "mem_pct": round(100 * sum(mem(x) for x in cs) / len(cs), 1),
                           "front_pct": round(100 * sum(mem(r) for r in fr) / (len(fr) or 1), 1)})
    return {"models": models, "n_models": len(models), "n_vendors": len({vendor(m) for m in models}),
            "tasks_evaluable": NT, "cells": len(cells),
            "mem_pct": round(100 * memc / ncell, 1), "clean_pct": round(100 * (ncell - memc) / ncell, 1),
            "moat": {"n": len(moat), "pct": round(100 * len(moat) / max(1, NT), 1)},
            "canon": {"n": len(canon), "pct": round(100 * len(canon) / max(1, NT), 1)},
            "mid": {"n": NT - len(moat) - len(canon)},
            "per_model": per_model, "per_type": per_type, "ladder_by_type": ladder_by_type,
            "recall_spectrum": [spec.get(k, 0) for k in range(len(models) + 1)],
            "frontier_model": frontier, "per_domain": sorted(per_domain, key=lambda z: z["mem_pct"]),
            "moat_members": moat, "canon_members": canon}


def feynman_block(new_rows_dir_pattern: str) -> dict:
    """Per-model AI-Feynman cold recall (single formula per task) for the 8 June OpenAI models + new legs."""
    june = collections.defaultdict(dict)
    for r in json.loads((ROOT / "data" / "openai8_feynman_rows.json").read_text()):
        june[canon_id(r["model"])][r["task"]] = r["d1_hits"]
    new = collections.defaultdict(dict)
    for rj in glob.glob(new_rows_dir_pattern, recursive=True):
        d = json.loads(Path(rj).read_text())
        if d.get("status"): continue
        f = d["formulas"][0] if d.get("formulas") else None
        if f: new[canon_id(d["model"])][d["task"]] = f["d1_hits"]
    out = {}
    for m, hits in list(june.items()) + list(new.items()):
        n = len(hits)
        out[m] = {"mem_pct": round(100 * sum(h >= MEMHIT for h in hits.values()) / (n or 1), 1), "n": n,
                  "complete": n >= 100, "source": "new_leg" if m in new else "june2026"}
    return out


def load_scores() -> dict:
    """Main-experiment per-task scores: data/model_task_scores.json + any runs/scores/*.json (same schema)."""
    files = [ROOT / "data" / "model_task_scores.json"] + sorted((ROOT / "runs" / "scores").glob("*.json"))
    tasks: dict = {}
    for f in files:
        if not f.exists(): continue
        d = json.loads(f.read_text()); d = d.get("tasks", d)
        for t, v in d.items():
            cur = tasks.setdefault(t, {"type": v.get("type"), "real_numeric": {}, "parallel_structure": {}})
            for k in ("real_numeric", "parallel_structure"):
                cur[k].update(v.get(k, {}))
    return tasks


def frontier_by_sn(models: list[str]) -> str | None:
    """The panel model with the highest mean S_N in the main experiment (the paper's 'frontier')."""
    inv = {v["main_exp_id"]: k for k, v in META.items() if v.get("main_exp_id")}
    tasks = load_scores(); means = {}
    for t, v in tasks.items():
        for mid, s in v["real_numeric"].items():
            k = inv.get(mid)
            if k in models and s is not None:
                means.setdefault(k, []).append(s)
    return max(means, key=lambda k: sum(means[k]) / len(means[k])) if means else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", help="explicit comma-separated model ids for the 'current' panel")
    a = ap.parse_args(argv)
    bb = json.loads((ROOT / "data" / "best_baseline_map.json").read_text())
    bb = {t: v for t, v in bb.items() if t not in EXCLUDE}
    dom = json.loads((ROOT / "data" / "domains.json").read_text())
    task2dom = {t: g["name"] for g in dom["domains"] for t in g["tasks"]}
    june, july, new = load_all()
    print("rows: june-openai8 =", {m: len({r['task'] for r in v}) for m, v in june.items()})
    print("rows: july-legs     =", {m: len({r['task'] for r in v}) for m, v in july.items()})
    print("rows: NEW legs      =", {m: len({r['task'] for r in v}) for m, v in new.items()} or "none")

    # committed-panel rows (June OpenAI + July legs); current panel rows = same but NEW legs override
    committed = {**june, **july}
    current_rows = dict(committed)
    complete_new = [m for m, v in new.items() if len({r["task"] for r in v}) >= 118]
    partial_new = [m for m in new if m not in complete_new]
    for m in complete_new:
        current_rows[m] = new[m]
    cell_c = collapse(committed, bb); cell_n = collapse(current_rows, bb)

    panels = {"openai8": panel_stats(cell_c, LADDER8, task2dom, frontier="gpt-5.5"),
              "maintable9": panel_stats(cell_c, MAINTABLE9, task2dom, frontier="gpt-5.5")}
    cur_models = [m.strip() for m in a.panel.split(",")] if a.panel else MAINTABLE9 + [m for m in complete_new if m not in MAINTABLE9]
    panels["current"] = panel_stats(cell_n, cur_models, task2dom, frontier=frontier_by_sn(cur_models) or "gpt-5.5")
    panels["current"]["frontier_rule"] = "panel model with the highest mean S_N in the main experiment"
    panels["current"]["replaced_by_new_legs"] = [m for m in complete_new if m in MAINTABLE9]
    panels["current"]["new_models"] = [m for m in complete_new if m not in MAINTABLE9]
    panels["current"]["partial_new_legs_ignored"] = partial_new

    ok = True
    for name, g in GATES.items():
        p = panels[name]
        checks = {"moat_n": (p["moat"]["n"], g["moat_n"]), "canon_n": (p["canon"]["n"], g["canon_n"]),
                  "mem": (p["mem_pct"], g["mem"]), "tasks": (p["tasks_evaluable"], g["tasks"])}
        for k, (got, exp) in checks.items():
            flag = "OK" if abs(got - exp) < 0.06 else "**MISMATCH**"; ok &= flag == "OK"
            print(f"  gate {name:11s} {k:8s} got {got:<6} expect {exp:<6} {flag}")
    if not ok:
        print("ABORT: the committed paper numbers do not reproduce — inputs or code changed.", file=sys.stderr); sys.exit(1)

    out = {"rule": f"cold-recalled iff d1_hits >= {MEMHIT} on best-baseline", "panels": panels,
           "feynman": feynman_block(str(ROOT / "runs" / "feynman" / "**" / "result.json")),
           "scilaws_for_feynman_fig": {m: panels["current"]["per_model"].get(m, {}).get("mem_pct")
                                       or panel_stats(cell_n, [m], task2dom)["per_model"][m]["mem_pct"]
                                       for m in set(LADDER8) | set(m for m in current_rows if vendor(m) == "OpenAI")}}
    (OUT / "recompute.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    c = panels["current"]
    print(f"\ncurrent panel: {c['n_models']} models / {c['n_vendors']} vendors / {c['cells']} cells | "
          f"mem {c['mem_pct']}% | moat {c['moat']['n']} ({c['moat']['pct']}%) | canon {c['canon']['n']} ({c['canon']['pct']}%) | "
          f"frontier {c['frontier_model']} {c['per_model'][c['frontier_model']]['mem_pct']}%")
    print("wrote", OUT / "recompute.json")


if __name__ == "__main__":
    main()
