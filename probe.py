"""The memorization probe (frozen instrument). Same logic as the paper's `pilot_v3.py`.

For every (task, model):
  D1  cold recall — the model sees ONLY the target + input descriptions (no data, no citation),
      writes `def predict(X)`, 5 samples at temperature 0.8.
  judge — gpt-4.1 (temperature 0) decides, per sample and per reference formula, whether the
      two functions have the SAME functional form (coefficients fold; exponents / term count /
      variables / function family are structural).
  result.json — per formula: d1_hits (0..5). The paper's verdict: cold-recalled iff d1_hits >= 3
      on the task's best-baseline formula (computed later, in analysis/).
D2 (cued recall with an "Author (year)" hint) is kept for completeness but is OFF by default —
the paper does not use it.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, re, sys, time
from pathlib import Path
import yaml

try:
    from .client import complete
except ImportError:
    from client import complete

HERE = Path(__file__).resolve().parent
PROBE_VERSION = "pilot_v3 / kit 2026-09"

# ----------------------------------------------------------------------------- prompts (FROZEN — byte-identical to the paper runs)
D1_PROMPT = """You are an expert in {domain}. Using ONLY your own knowledge — you are given NO data — \
write the standard closed-form relationship for the quantity below as a Python function.

Quantity to predict:
- {t_name} ({t_sym}){t_unit} — {t_desc}

Inputs (X is a 2-D numpy array; column X[:, i] is the i-th input listed):
{inputs_block}

Write a single Python function `def predict(X):` (use `import numpy as np`) that returns a 1-D \
array of {t_sym}. Fitted coefficients may be literal numbers or named constants — your choice. \
If several canonical forms exist, write the most commonly cited one. Output ONLY one \
```python ... ``` block, nothing else."""

D2_CITE = """

Recall the canonical closed form for {t_sym} exactly as published in {citation}, \
using your general knowledge of that specific work."""

JUDGE_PROMPT = """You compare two Python functions that both predict {t_sym}. Decide whether they \
implement the **same functional form**.

The input variables (carry the same meaning in both; NOT renameable): {inputs}. Everything else \
(literal numbers, named constants, free parameters) is a fittable COEFFICIENT.

Equivalence rules:
(1) Standard algebra preserves equivalence (commutativity, factoring, exp(log x)=x, log/exp laws...).
(2) **Coefficient fold (critical).** At any multiplicative-factor / additive-offset / ratio / \
internal-rate / pivot position, a literal number on one side and a free symbol (or a different \
number) on the other are EQUIVALENT. A baked number is STRONGER recall, NOT a different form. \
Renaming free coefficients is allowed. Combinations of coefficients fold to one coefficient.
(3) max(_,0)/np.maximum(_,0)/clipping is identity on the valid range.
(4) **Exponents and structural integer powers are NOT coefficients — they must MATCH in value** \
(approximately): `x**2` vs `x**(5/3)` is a DIFFERENT form; `x**1.667` vs `x**(5/3)` is the SAME. \
Likewise the count/degree of terms is structural.
(5) Different functional FAMILIES at a structural position fail: exp(-x) vs 1/(1+x), tanh vs \
x/sqrt(1+x^2), polynomial vs log, VBGF vs Gompertz, etc. This and (4) are the ONLY axes along \
which two functions are "different".

Judge by the FORM only; do not run or fit. Ignore which side uses numbers vs symbols for coefficients.

REFERENCE (ground truth):
```python
{gt_code}
```

CANDIDATE (model output):
```python
{cand_code}
```

Output JSON only: {{"equivalent": true|false, "reason": "<one sentence>"}}"""


# ----------------------------------------------------------------------------- task assets
def load_meta(task_dir: Path) -> dict:
    return yaml.safe_load((task_dir / "metadata.yaml").read_text())


def io_spec(meta: dict) -> tuple[dict, list[dict]]:
    t = meta["target"]
    target = {"name": t.get("name", "y"), "sym": t.get("symbol") or t.get("name", "y"),
              "unit": t.get("unit", "") or "", "desc": t.get("description", "") or ""}
    inputs = [{"name": i.get("name", ""), "sym": i.get("symbol") or i.get("name", ""),
               "unit": i.get("unit", "") or "", "desc": i.get("description", "") or "",
               "range": i.get("range")}
              for i in meta["inputs"] if i.get("name") != "group_id"]
    return target, inputs


def inputs_block(inputs: list[dict]) -> str:
    out = []
    for i, v in enumerate(inputs):
        u = f" [{v['unit']}]" if v["unit"] else ""
        rng = f"  (typical range {v['range']})" if v.get("range") else ""
        out.append(f"- X[:, {i}] = {v['name']} ({v['sym']}){u} — {v['desc']}{rng}")
    return "\n".join(out)


def render_d1_prompt(task_dir: Path) -> str:
    """The exact D1 prompt for a task (what the subject model receives)."""
    meta = load_meta(task_dir)
    target, inputs = io_spec(meta)
    t_unit = f" [{target['unit']}]" if target["unit"] else ""
    return D1_PROMPT.format(domain=meta.get("domain", "science"), t_name=target["name"],
                            t_sym=target["sym"], t_unit=t_unit, t_desc=target["desc"],
                            inputs_block=inputs_block(inputs))


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


_DOC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)


def _strip_docstrings(node):
    for n in ast.walk(node):
        if isinstance(n, _DOC_TYPES):
            b = getattr(n, "body", [])
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                    and isinstance(b[0].value.value, str):
                n.body = b[1:] or [ast.Pass()]


def gt_code(formula_py: Path) -> tuple[str, list[str]]:
    """Reference code for the judge: USED_INPUTS / *_CONSTANTS / *_FITTABLE assigns + all
    module-level functions except `fit`, docstrings stripped."""
    src = formula_py.read_text()
    tree = ast.parse(src)
    used: list[str] = []
    keep: list[ast.stmt] = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            if name == "USED_INPUTS":
                try: used = list(ast.literal_eval(n.value))
                except Exception: pass
            if name in ("USED_INPUTS",) or name.endswith("_CONSTANTS") or name.endswith("_FITTABLE"):
                keep.append(n)
        elif isinstance(n, ast.FunctionDef) and n.name != "fit":
            keep.append(n)
    mod = ast.Module(body=keep, type_ignores=[])
    _strip_docstrings(mod)
    ast.fix_missing_locations(mod)
    return ast.unparse(mod), used


def citation_of(formula_py: Path) -> str:
    """'aab_2019' -> 'Aab (2019)'; descriptive stems without a year -> '' (D2 == D1)."""
    stem = formula_py.stem
    m = re.search(r"(\d{4})", stem)
    if not m:
        return ""
    authors = stem[:m.start()].rstrip("_").replace("_", " ").strip()
    if not authors:
        return ""
    return f"{' '.join(w.capitalize() for w in authors.split())} ({m.group(1)})"


# ----------------------------------------------------------------------------- elicitation + judge
def extract_code(raw: str) -> str:
    m = re.search(r"```(?:python)?\s*(.+?)```", raw, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:^|\n)(import .+|def predict.+)", raw, re.S)
    if m:
        return raw[m.start():].strip()
    return raw.strip()


def judge(gt_src: str, cand_src: str, t_sym: str, input_syms: list[str],
          judge_model: str, log_dir: Path) -> dict:
    prompt = JUDGE_PROMPT.format(t_sym=t_sym, inputs=", ".join(input_syms) or "(none)",
                                 gt_code=gt_src, cand_code=cand_src)
    out = complete(judge_model, prompt, temperature=0.0, max_tokens=400, n=1, log_dir=log_dir)[0]
    m = re.search(r"\{.*\}", out, re.S)
    try:
        v = json.loads(m.group(0)) if m else {}
    except Exception:
        v = {}
    return {"equivalent": bool(v.get("equivalent", False)), "parsed": bool(v),
            "reason": str(v.get("reason", ""))[:200]}


def verdict(d1_rate: float, d2_rate: float) -> str:
    """Legacy 3-way label kept in result.json; the paper uses only d1_hits >= 3 (binary)."""
    EPS = 1e-3
    if d1_rate >= 0.60 - EPS or d2_rate >= 2.0 / 3.0 - EPS:
        return "memorized"
    if d1_rate >= 0.20 - EPS or d2_rate >= 1.0 / 3.0 - EPS:
        return "suspect"
    return "clean"


# ----------------------------------------------------------------------------- per-task run
def run_task(task_dir: Path, model: str, judge_model: str, probes: list[str], out_root: Path) -> dict:
    meta = load_meta(task_dir)
    target, inputs = io_spec(meta)
    input_syms = [i["sym"] for i in inputs]
    fdir = task_dir / "formulas"
    formula_files = [p for p in sorted(fdir.glob("*.py")) if not p.stem.startswith("_")]
    formulas = [{"stem": fp.stem, "gt": gt_code(fp)[0], "py": fp} for fp in formula_files]
    if not formulas:
        return {"status": "no-formulas"}

    out_dir = out_root / task_dir.name / model
    out_dir.mkdir(parents=True, exist_ok=True)
    jlog = out_dir / "judge_raw"
    d1_prompt = render_d1_prompt(task_dir)

    res = {"task": task_dir.name, "model": model, "type": meta.get("type"),
           "primary": (meta.get("domain", "") or "").split("/")[0],
           "judge_model": judge_model, "probe_version": PROBE_VERSION,
           "d1_prompt_sha256_16": sha(d1_prompt), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "judge_parse_failures": 0, "formulas": []}

    d1_hits = [0] * len(formulas)
    d1_samples: list[str] = []
    if "d1" in probes:
        raws = complete(model, d1_prompt, temperature=0.8, max_tokens=900, n=5, log_dir=out_dir / "d1_raw")
        d1_samples = [extract_code(r) for r in raws]
        for fi, f in enumerate(formulas):
            for c in d1_samples:
                if not c.strip():
                    continue                       # empty answer = no hit, no judge call
                j = judge(f["gt"], c, target["sym"], input_syms, judge_model, jlog)
                res["judge_parse_failures"] += int(not j["parsed"])
                d1_hits[fi] += int(j["equivalent"])

    d2_hits = [0] * len(formulas)
    d2_how: list[str] = []
    if "d2" in probes:
        for fi, f in enumerate(formulas):
            cite = citation_of(f["py"])
            extra, how = (D2_CITE.format(t_sym=target["sym"], citation=cite), "citation") if cite else ("", "no-cite")
            d2_how.append(how)
            prompt = d1_prompt.replace("\n\nWrite a single Python", extra + "\n\nWrite a single Python")
            raws = complete(model, prompt, temperature=0.3, max_tokens=900, n=3, log_dir=out_dir / "d2_raw")
            for r in raws:
                c = extract_code(r)
                if c.strip() and judge(f["gt"], c, target["sym"], input_syms, judge_model, jlog)["equivalent"]:
                    d2_hits[fi] += 1

    for fi, f in enumerate(formulas):
        d1r = d1_hits[fi] / 5 if "d1" in probes else 0.0
        d2r = d2_hits[fi] / 3 if "d2" in probes else 0.0
        res["formulas"].append({
            "formula": f["stem"], "d1_hits": d1_hits[fi], "d1_rate": round(d1r, 3),
            "d2_hits": d2_hits[fi], "d2_rate": round(d2r, 3),
            "d2_context": (d2_how[fi] if "d2" in probes else None),
            "verdict": verdict(d1r, d2r),
        })
    res["n_empty_d1_samples"] = sum(1 for c in d1_samples if not c.strip())
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out_dir / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    (out_dir / "d1_samples.txt").write_text("\n\n=====\n\n".join(d1_samples))
    return res


def load_tasks(path: Path) -> list[str]:
    raw = json.loads(path.read_text())
    return (raw["typeI"] + raw["typeII"]) if isinstance(raw, dict) else raw


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--bench", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--judge", default="gpt-4.1")
    p.add_argument("--probes", default="d1")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", default="", help="comma-separated subset of task ids")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out-root", required=True)
    a = p.parse_args(argv)

    tasks = load_tasks(Path(a.tasks))
    if a.only:
        want = [t.strip() for t in a.only.split(",") if t.strip()]
        unknown = [t for t in want if t not in tasks]
        if unknown:
            raise SystemExit("unknown task id(s): " + ", ".join(unknown))
        tasks = [t for t in tasks if t in want]
    if a.limit:
        tasks = tasks[:a.limit]
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    probes = [x.strip() for x in a.probes.split(",") if x.strip()]
    bench, out_root = Path(a.bench), Path(a.out_root)
    alt_judge = "gpt-4.1-mini" if a.judge != "gpt-4.1-mini" else "gpt-4.1"   # judge must differ from subject

    def do(job):
        tname, model = job
        jm = a.judge if model != a.judge else alt_judge
        return tname, model, run_task(bench / tname, model, jm, probes, out_root)

    rows = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    jobs = [(t, m) for t in tasks for m in models]
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(do, j): j for j in jobs}
        for fut in as_completed(futs):
            tname, model = futs[fut]
            try:
                _, _, r = fut.result()
                if r.get("status"):
                    print(f"[skip] {tname} | {model}: {r['status']}", file=sys.stderr); continue
                for f in r["formulas"]:
                    rows.append({**f, "task": tname, "model": model, "type": r["type"]})
                    print(f"{tname[:38]:38s} | {model:22s} | {f['formula'][:22]:22s} | D1 {f['d1_hits']}/5"
                          + (f" D2 {f['d2_hits']}/3" if "d2" in probes else ""), flush=True)
            except Exception as e:
                print(f"[ERR] {tname} | {model}: {e}", file=sys.stderr, flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "rows_last_invocation.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    return rows


if __name__ == "__main__":
    main()
