"""Flatten the main-experiment per-task scores into the rows Fig 5 needs, tagged with the
current memorization tier of every task.

Inputs : data/model_task_scores.json (paper run) + runs/scores/*.json (same schema, new models)
         analysis/out/recompute.json (tiers of the 'current' panel)
Output : analysis/out/fig5_data.json  {"models": [...], "rows": [{task, model, type, tier, real, struct}]}
Model ids in the scores files are the collaborator's directory names (main_exp_id in models_meta.json).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "analysis"))
import recompute as RC

def main():
    rc = json.loads((RC.OUT / "recompute.json").read_text())["panels"]["current"]
    moat, canon = set(rc["moat_members"]), set(rc["canon_members"])
    inv = {v["main_exp_id"]: k for k, v in RC.META.items() if v.get("main_exp_id")}
    tasks = RC.load_scores()
    rows, models, unknown = [], set(), set()
    for t, v in sorted(tasks.items()):
        if t in RC.EXCLUDE: continue
        tier = "Moat" if t in moat else "Canon" if t in canon else "Mid"
        for mid in set(v["real_numeric"]) | set(v["parallel_structure"]):
            k = inv.get(mid)
            if k is None:
                unknown.add(mid); continue
            if k not in rc["models"]: continue
            models.add(k)
            rows.append({"task": t, "model": k, "type": v.get("type"), "tier": tier,
                         "real": v["real_numeric"].get(mid), "struct": v["parallel_structure"].get(mid)})
    if unknown:
        print("WARNING: score ids not in analysis/models_meta.json (add main_exp_id):", sorted(unknown))
    missing = [m for m in rc["models"] if m not in models]
    if missing:
        print("WARNING: panel models WITHOUT main-experiment scores (Fig 5 will omit them):", missing)
    out = {"models": sorted(models), "rows": rows, "tiers": {"Moat": len(moat), "Canon": len(canon),
           "Mid": rc["mid"]["n"]}}
    (RC.OUT / "fig5_data.json").write_text(json.dumps(out, indent=1))
    print(f"fig5_data: {len(rows)} rows, {len(models)} models, tiers {out['tiers']} -> {RC.OUT/'fig5_data.json'}")

if __name__ == "__main__":
    main()
