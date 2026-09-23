"""Ready-made `custom_client.py` for teams whose LLM access is the SciLaws agent baseline's
`call_llm_api.py` (the file with `api_source_mapping` and `call_llm_api(messages, alias, ...)`).

Setup:
  1. cp examples/custom_client_call_llm_api.py custom_client.py
  2. export MEM_CALL_LLM_API_DIR=/path/to/baseline_agent     # the directory that contains call_llm_api.py
  3. export MEM_FORCE_BACKEND=custom                          # every call (subjects AND the judge) goes through call_llm_api
  4. python3 run_mem.py --bench scilaws --models cc-opus-5-5 --limit 2

Model ids on the command line are the ALIASES of api_source_mapping (cc-opus-5-5, gpt55, or-claude-opus-4-7, ...);
register each one in models.json with backend "custom" and the right kind (reasoning / chat). The judge
"gpt-4.1" is mapped to the alias "gpt41" below, so OPENAI_API_KEY must be set (as call_llm_api requires).

cc-* aliases (claude -p on a Claude subscription): effort is fixed to medium below (the setting used for the
paper's reasoning legs and for the baseline's own cc runs); the CLI session scratch dir and a per-call cost
log go under runs/. Tested 2026-09-23 with cc-opus-5-5 (Claude Code 2.1.280): 6 tasks x 5 samples, ~$0.013
list-price equivalent and ~7 s per call, hits in line with the paper's panel.
  Caveat of the cc route: `claude -p` always prepends its own short agent system prompt plus an environment
  block (cwd, OS, date, model name) to the probe prompt; no flag removes it. The paper's Claude leg (Opus 4.8)
  was run with no system prompt through OpenRouter, so for a paper leg prefer the kit's own openrouter backend
  (models.json already has anthropic/claude-opus-5.5; export OPENROUTER_API_KEY, no custom client needed).
"""
from __future__ import annotations
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CC_EFFORT", "medium")                                       # read by call_llm_api at import
os.environ.setdefault("CC_WORKDIR", os.path.join(_HERE, "runs", "_cc_cwd"))
os.environ.setdefault("CC_CALL_LOG", os.path.join(_HERE, "runs", "_cc_calls.jsonl"))
os.makedirs(os.path.join(_HERE, "runs"), exist_ok=True)

CALL_LLM_API_DIR = os.environ.get("MEM_CALL_LLM_API_DIR") or "/path/to/SCILAWS-BENCH/baseline_agent"
sys.path.insert(0, CALL_LLM_API_DIR)
from call_llm_api import call_llm_api  # noqa: E402

ALIAS = {"gpt-4.1": "gpt41", "gpt-4.1-mini": "gpt41mini"}     # kit model id -> api_source_mapping alias


def complete(model: str, prompt: str, temperature: float, n: int, max_tokens: int, kind: str) -> list:
    alias = ALIAS.get(model, model)
    outs = []
    for i in range(n):
        try:
            content, _reasoning, info = call_llm_api(
                [{"role": "user", "content": prompt}], alias, temperature=temperature,
                trial_info={"trial_id": f"mem-{alias}-{i}"})
            outs.append({"content": content or "", "usage": info,
                         "cost_usd_equiv": (info or {}).get("cost_usd_equiv")})
        except Exception as e:                    # call_llm_api already retried; an empty sample = no answer
            print(f"[custom_client] {alias}: {e}", file=sys.stderr, flush=True)
            outs.append({"content": "", "error": str(e)[:300]})
    return outs
