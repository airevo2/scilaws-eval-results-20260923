"""End-to-end test with NO API calls: a fake custom_client answers every call.
Subject calls get a python block (the task's own best formula for task #1 -> hits; a wrong
linear form for task #2 -> misses); judge calls get a JSON verdict computed by string match.
Checks: result.json schema, hit counts, coverage table, check_run.py, --pack, --dry-run.
Run:  python3 tests/mock_e2e.py
"""
import json, os, shutil, subprocess, sys, types
from pathlib import Path
K = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(K))
os.environ["MEM_FORCE_BACKEND"] = "custom"
RUNS = K / "runs_mocktest"; shutil.rmtree(RUNS, ignore_errors=True)

fake = types.ModuleType("custom_client")
def _complete(model, prompt, temperature, n, max_tokens, kind):
    if prompt.startswith("You compare two Python functions"):          # judge
        ref = prompt.split("REFERENCE (ground truth):")[1].split("CANDIDATE")[0]
        cand = prompt.split("CANDIDATE (model output):")[1]
        eq = "MOCK_HIT" in cand and "MOCK_HIT" not in ref and "def predict" in ref
        return [json.dumps({"equivalent": eq, "reason": "mock"})]
    tag = "MOCK_HIT" if "Season total runs" in prompt else "MOCK_MISS"      # baseball = 1st task -> hits
    return [f"```python\nimport numpy as np\ndef predict(X):\n    # {tag}\n    return X[:, 0] * 1.0\n```" for _ in range(n)]
fake.complete = _complete
sys.modules["custom_client"] = fake

import probe, run_mem
# 2 tasks x 1 mock model, scilaws bench
probe.main(["--tasks", str(K/"tasks/scilaws118.json"), "--bench", str(K/"bench/scilaws118"),
            "--models", "mock-model", "--judge", "gpt-4.1", "--probes", "d1", "--workers", "2",
            "--out-root", str(RUNS/"scilaws"), "--limit", "2"])
res = sorted((RUNS/"scilaws").glob("*/mock-model/result.json"))
assert len(res) == 2, res
r = json.loads(res[0].read_text())
for k in ["task","model","type","judge_model","probe_version","d1_prompt_sha256_16","judge_parse_failures","formulas","n_empty_d1_samples"]:
    assert k in r, k
hits = {json.loads(p.read_text())["task"]: [f["d1_hits"] for f in json.loads(p.read_text())["formulas"]] for p in res}
print("d1_hits per task:", hits)
assert any(h == 5 for hs in hits.values() for h in hs), "expected 5/5 hits on the MOCK_HIT task"
assert (res[0].parent/"d1_samples.txt").exists() and list((res[0].parent/"judge_raw").glob("*.json")) and list((res[0].parent/"d1_raw").glob("*.json"))
frozen = json.load(open(K/"prompts_frozen.json"))["scilaws"]
assert all(json.loads(p.read_text())["d1_prompt_sha256_16"] == frozen[json.loads(p.read_text())["task"]] for p in res), "prompt hash mismatch"
run_mem.coverage(RUNS/"scilaws", ["mock-model"], 2)
# --only subset + dict-style custom returns (usage / billed_usd logged per sample)
fake.complete = lambda model, prompt, temperature, n, max_tokens, kind: (
    [json.dumps({"equivalent": False, "reason": "mock"})] if prompt.startswith("You compare two Python functions")
    else [{"content": "```python\nimport numpy as np\ndef predict(X):\n    return X[:, 0]\n```", "usage": {"output_tokens": 7}, "billed_usd": 0.001} for _ in range(n)])
probe.main(["--tasks", str(K/"tasks/scilaws118.json"), "--bench", str(K/"bench/scilaws118"), "--models", "mock2", "--judge", "gpt-4.1",
            "--workers", "2", "--out-root", str(RUNS/"scilaws"), "--only", "gravity_wgs84_somigliana__g0,spirometry_nhanes__FVC_L"])
assert len(list((RUNS/"scilaws").glob("*/mock2/result.json"))) == 2
raw = json.loads(next((RUNS/"scilaws").glob("gravity_wgs84_somigliana__g0/mock2/d1_raw/*.json")).read_text())
assert raw["per_sample"][0]["usage"]["output_tokens"] == 7 and abs(raw["billed_usd"] - 0.005) < 1e-9, raw
try:
    probe.main(["--tasks", str(K/"tasks/scilaws118.json"), "--bench", str(K/"bench/scilaws118"), "--models", "mock2", "--out-root", str(RUNS/"x"), "--only", "no_such_task"])
    raise AssertionError("unknown --only task was not rejected")
except SystemExit:
    pass
# check_run on the mock tree (expects PARTIAL because only 2/118) and dry-run / list as subprocesses
p = subprocess.run([sys.executable, str(K/"check_run.py"), str(RUNS)], capture_output=True, text=True); print(p.stdout[-600:]); assert "PARTIAL (116 missing)" in p.stdout and p.returncode == 1
p = subprocess.run([sys.executable, str(K/"run_mem.py"), "--dry-run", "--bench", "feynman"], capture_output=True, text=True); assert "mismatches = 0" in p.stdout and p.returncode == 0, p.stdout[-300:]+p.stderr[-300:]
p = subprocess.run([sys.executable, str(K/"run_mem.py"), "--list"], capture_output=True, text=True); assert "gpt-5.5" in p.stdout and p.returncode == 0
shutil.rmtree(RUNS)
print("MOCK E2E: OK")
