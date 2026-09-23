"""One-command entry point for the memorization probe.

    python3 run_mem.py --list                                   # models known to models.json
    python3 run_mem.py --dry-run --bench scilaws                # render + verify the frozen prompts, NO API calls
    python3 run_mem.py --bench scilaws --models gpt-6 --limit 2 # smoke test (2 tasks, ~$0.1), do this first
    python3 run_mem.py --bench scilaws --models gpt-6           # full leg: 118 tasks x 5 samples + judge
    python3 run_mem.py --bench feynman --models gpt-6           # AI-Feynman leg: 100 tasks (OpenAI models)
    python3 run_mem.py --models a,b,c ...                       # several models in one go
    python3 run_mem.py --pack                                   # zip runs/ to send back

Output: runs/<bench>/<task>/<model>/{result.json, d1_samples.txt, d1_raw/, judge_raw/}
Keys:   OPENAI_API_KEY (+ OPENAI_BASE_URL for a gateway), OPENROUTER_API_KEY for openrouter models,
        or MEM_FORCE_BACKEND=custom with custom_client.py filled in. The judge gpt-4.1 must be reachable.
"""
from __future__ import annotations
import argparse, json, os, sys, zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from client import MODELS, model_cfg          # noqa: E402
import probe                                  # noqa: E402

JUDGE = "gpt-4.1"                             # frozen
BENCHES = {
    "scilaws": {"dir": HERE / "bench" / "scilaws118", "tasks": HERE / "tasks" / "scilaws118.json", "n": 118},
    "feynman": {"dir": HERE / "bench" / "feynman100", "tasks": HERE / "tasks" / "feynman100.json", "n": 100},
}
FROZEN = HERE / "prompts_frozen.json"         # sha256[:16] of every D1 prompt, per bench/task


def cmd_list() -> None:
    print(f"{'model id':<30}{'backend':<12}{'kind':<11}{'provider':<11}note")
    for m, c in MODELS.items():
        print(f"{m:<30}{c['backend']:<12}{c.get('kind','chat'):<11}{c.get('provider','-'):<11}{c.get('role','')}")
    print(f"\njudge = {JUDGE} (frozen).  benches = {', '.join(BENCHES)}.  probes = d1 (cold recall) by default.")
    print("A gpt-*/o* id not listed here still works on the openai backend; other ids must be added to models.json.")


def check_keys(models: list[str]) -> None:
    backends = {model_cfg(m)["backend"] for m in models} | {model_cfg(JUDGE)["backend"]}
    need = set()
    if "openai" in backends: need.add("OPENAI_API_KEY")
    if "openrouter" in backends: need.add("OPENROUTER_API_KEY")
    missing = [k for k in sorted(need) if not os.environ.get(k)]
    if missing:
        raise SystemExit("missing environment variable(s): " + ", ".join(missing))


def dry_run(bench: str) -> int:
    b = BENCHES[bench]
    tasks = probe.load_tasks(b["tasks"])
    frozen = json.loads(FROZEN.read_text())[bench] if FROZEN.exists() else {}
    bad = 0
    for t in tasks:
        h = probe.sha(probe.render_d1_prompt(b["dir"] / t))
        if frozen and frozen.get(t) != h:
            bad += 1; print(f"  PROMPT MISMATCH: {t}")
    print(f"{bench}: {len(tasks)} prompts rendered; frozen-hash mismatches = {bad}"
          + ("" if frozen else "  (no prompts_frozen.json to compare against)"))
    ex = tasks[0]
    print(f"\n----- example prompt ({ex}) -----\n{probe.render_d1_prompt(b['dir'] / ex)}\n")
    return bad


def coverage(out_root: Path, models: list[str], n_expect: int) -> None:
    print(f"\n{'model':<30}{'tasks':>7}{'empty-d1':>9}{'judge-parse-fail':>17}{'billed $':>10}")
    for m in models:
        done, empties, jfail, billed = 0, 0, 0, 0.0
        for rj in out_root.glob(f"*/{m}/result.json"):
            r = json.loads(rj.read_text()); done += 1
            empties += r.get("n_empty_d1_samples", 0); jfail += r.get("judge_parse_failures", 0)
        for f in out_root.glob(f"*/{m}/*_raw/*.json"):
            try: billed += json.loads(f.read_text()).get("billed_usd", 0) or 0
            except Exception: pass
        flag = "" if done >= n_expect else f"   <- INCOMPLETE, expected {n_expect}"
        print(f"{m:<30}{done:7}{empties:9}{jfail:17}{billed:10.2f}{flag}")


def pack() -> Path:
    zpath = HERE / "mem_runs.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((HERE / "runs").rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(HERE))
    print(f"packed -> {zpath}  ({zpath.stat().st_size / 1e6:.1f} MB)")
    return zpath


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="render + verify prompts, no API calls")
    ap.add_argument("--bench", choices=list(BENCHES), default="scilaws")
    ap.add_argument("--models", help="comma-separated model ids (see --list / models.json)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks (smoke test)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--probes", default="d1", help="d1 (paper) or d1,d2")
    ap.add_argument("--pack", action="store_true", help="zip runs/ (can be used alone)")
    a = ap.parse_args(argv)

    if a.list:
        cmd_list(); return
    if a.dry_run:
        raise SystemExit(1 if dry_run(a.bench) else 0)
    if a.pack and not a.models:
        pack(); return
    if not a.models:
        ap.error("--models is required (or use --list / --dry-run / --pack)")

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    for m in models:
        model_cfg(m)                              # fails early for unknown vendor/model ids
    check_keys(models)
    b = BENCHES[a.bench]
    n = a.limit or b["n"]
    out_root = HERE / "runs" / a.bench
    print(f"bench: {a.bench} ({n} tasks)   models: {', '.join(models)}   judge: {JUDGE}   probes: {a.probes}   workers: {a.workers}")
    if dry_run(a.bench):
        raise SystemExit("frozen prompts do not match — do not run; report this")

    probe.main(["--tasks", str(b["tasks"]), "--bench", str(b["dir"]), "--models", ",".join(models),
                "--judge", JUDGE, "--probes", a.probes, "--workers", str(a.workers),
                "--out-root", str(out_root)] + (["--limit", str(a.limit)] if a.limit else []))
    coverage(out_root, models, n)
    if a.pack:
        pack()


if __name__ == "__main__":
    main()
