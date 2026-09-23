"""Self-check of a runs/ tree before sending it back (or after receiving it).

    python3 check_run.py            # checks ./runs
    python3 check_run.py <dir>      # any runs root (e.g. an unzipped mem_runs.zip)

Reports, per bench and model: tasks completed / expected, empty D1 samples, judge parse
failures, prompt-hash mismatches against prompts_frozen.json, billed cost, missing tasks.
Exit code 1 if any leg is incomplete or any prompt hash mismatches.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECT = {"scilaws": 118, "feynman": 100}


def main(argv=None):
    argv = argv or sys.argv[1:]
    root = Path(argv[0]) if argv else HERE / "runs"
    if (root / "runs").is_dir():                       # an unzipped mem_runs.zip
        root = root / "runs"
    frozen = json.loads((HERE / "prompts_frozen.json").read_text()) if (HERE / "prompts_frozen.json").exists() else {}
    def _tasks(b, n):
        raw = json.loads((HERE / "tasks" / f"{b}{n}.json").read_text())
        return set(raw["typeI"] + raw["typeII"]) if isinstance(raw, dict) else set(raw)
    tasks_all = {b: _tasks(b, n) for b, n in EXPECT.items()}
    bad = False
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name in EXPECT):
        bench = bench_dir.name
        per = defaultdict(lambda: {"tasks": set(), "empty": 0, "jfail": 0, "hash_bad": 0, "billed": 0.0})
        for rj in bench_dir.rglob("result.json"):
            r = json.loads(rj.read_text())
            if r.get("status"):
                continue
            m = r["model"]; s = per[m]
            s["tasks"].add(r["task"]); s["empty"] += r.get("n_empty_d1_samples", 0)
            s["jfail"] += r.get("judge_parse_failures", 0)
            fh = frozen.get(bench, {}).get(r["task"])
            if fh and r.get("d1_prompt_sha256_16") and fh != r["d1_prompt_sha256_16"]:
                s["hash_bad"] += 1
            for f in rj.parent.glob("*_raw/*.json"):
                try: s["billed"] += json.loads(f.read_text()).get("billed_usd", 0) or 0
                except Exception: pass
        n_exp = EXPECT.get(bench, 0)
        print(f"\n[{bench}]  expected {n_exp} tasks per model")
        print(f"{'model':<30}{'done':>6}{'empty-d1':>9}{'judge-fail':>11}{'hash-bad':>9}{'billed $':>10}  status")
        for m, s in sorted(per.items()):
            n = len(s["tasks"]); ok = n >= n_exp and s["hash_bad"] == 0
            bad |= not ok
            print(f"{m:<30}{n:6}{s['empty']:9}{s['jfail']:11}{s['hash_bad']:9}{s['billed']:10.2f}  "
                  f"{'COMPLETE' if n >= n_exp else f'PARTIAL ({n_exp - n} missing)'}{'' if not s['hash_bad'] else '  PROMPT-HASH MISMATCH'}")
            missing = sorted(tasks_all.get(bench, set()) - s["tasks"])
            if missing:
                print(f"   missing: {', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
    print("\nRESULT:", "PROBLEMS FOUND — see above" if bad else "all legs complete, prompts verified")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
