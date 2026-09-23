"""API layer for the memorization probe. ONE function matters: `complete()`.

    complete(model, prompt, *, temperature, max_tokens, n, log_dir) -> list[str]   # n completions

Three backends, chosen per model in `models.json` (or forced for everything with
`MEM_FORCE_BACKEND=custom|openai|openrouter`):

  openai      native OpenAI SDK. Honors OPENAI_API_KEY and, if set, OPENAI_BASE_URL — so any
              OpenAI-compatible gateway works without code changes. This path is the one that
              produced every OpenAI number in the paper; do not change its request shape.
  openrouter  OpenAI-compatible endpoint for non-OpenAI vendors (n-loop, explicit reasoning
              effort, first-party provider pinning). Produced the 5 non-OpenAI legs in the paper.
  custom      YOUR OWN client: fill in `custom_client.py` (one function) if your API is not
              OpenAI-compatible. Everything else in the kit stays untouched.

Every call is persisted (prompt, completions, usage, provider, billed cost when the API returns
it) under `runs/<bench>/<task>/<model>/{d1_raw,d2_raw,judge_raw}/`.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MODELS: dict[str, dict] = {
    k: v for k, v in json.loads((HERE / "models.json").read_text()).items()
    if not k.startswith("_")
}

OR_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS_CEILING = 28800   # frozen ceiling for reasoning models (= max(900*32, 16384)); a cap, not a spend
OR_HARD_TIMEOUT = 45         # seconds before an OpenRouter call that accepted-then-hung is abandoned

_client = None
_or_client = None


def model_cfg(model: str) -> dict:
    """Config for `model`. Unknown OpenAI-style names fall back to the native path; the
    reasoning/chat split is inferred from the name (gpt-5*, gpt-6*, o* = reasoning)."""
    if model in MODELS:
        cfg = dict(MODELS[model])
    elif "/" in model:
        raise ValueError(f"{model!r} is not in models.json — add it (backend/kind/provider)")
    else:
        cfg = {"backend": "openai",
               "kind": "reasoning" if re.match(r"^(gpt-[5-9]|o\d)", model) else "chat"}
    forced = os.environ.get("MEM_FORCE_BACKEND")
    if forced:
        cfg["backend"] = forced
    cfg.setdefault("kind", "chat")
    return cfg


def _get_client():
    global _client
    if _client is not None:
        return _client
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    _client = OpenAI(api_key=key)          # base_url comes from OPENAI_BASE_URL if you set it
    return _client


def _get_or_client():
    global _or_client
    if _or_client is not None:
        return _or_client
    from openai import OpenAI
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    _or_client = OpenAI(api_key=key, base_url=OR_BASE_URL, max_retries=0)
    return _or_client


def _write_log(log_dir: Path | None, record: dict) -> None:
    if log_dir is None:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{int(time.time())}_{uuid.uuid4().hex[:8]}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------- custom
def _complete_custom(model, prompt, temperature, n, max_tokens, log_dir, cfg) -> list[str]:
    import custom_client                      # the file YOU fill in
    t0 = time.time()
    out = custom_client.complete(model=model, prompt=prompt, temperature=temperature, n=n,
                                 max_tokens=max_tokens, kind=cfg.get("kind", "chat"))
    if not isinstance(out, list):
        raise TypeError("custom_client.complete must return a list of strings")
    out = [(s or "") for s in out][:n] + [""] * max(0, n - len(out))
    _write_log(log_dir, {
        "backend": "custom", "model_requested": model, "kind": cfg.get("kind"),
        "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens, "n": n,
        "completions": out, "finish_reasons": ["custom"] * n,
        "latency_sec": round(time.time() - t0, 3),
    })
    return out


# --------------------------------------------------------------------------- openrouter
def _complete_openrouter(model, messages, temperature, n, log_dir, prompt, max_retries, cfg) -> list[str]:
    client = _get_or_client()
    body: dict[str, Any] = {"usage": {"include": True}}
    if isinstance(cfg.get("reasoning"), dict):
        body["reasoning"] = cfg["reasoning"]
    if cfg.get("provider"):
        order = [p.strip() for p in str(cfg["provider"]).split(",") if p.strip()]
        body["provider"] = {"order": order, "allow_fallbacks": len(order) > 1}
    completions, finish, served = [], [], []
    p_tok = c_tok = r_tok = 0
    cost = 0.0
    returned = None
    t0 = time.time()
    for _ in range(n):
        delay, resp = 1.0, None
        for attempt in range(max_retries):
            try:
                fut = ThreadPoolExecutor(max_workers=1).submit(
                    client.chat.completions.create, model=model, messages=messages, n=1,
                    max_tokens=MAX_TOKENS_CEILING, temperature=temperature, extra_body=body,
                    timeout=OR_HARD_TIMEOUT + 10)
                resp = fut.result(timeout=OR_HARD_TIMEOUT)
                if resp.choices:
                    break
                raise RuntimeError("empty choices (upstream provider returned no completion)")
            except FuturesTimeout:
                resp = None
                break
            except Exception:
                resp = None
                if attempt == max_retries - 1:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        if resp is None or not resp.choices:
            completions.append(""); finish.append("empty")
            continue
        ch = resp.choices[0]
        completions.append(ch.message.content or "")
        finish.append(ch.finish_reason)
        returned = getattr(resp, "model", None)
        served.append(getattr(resp, "provider", None))
        u = getattr(resp, "usage", None)
        if u is not None:
            p_tok += getattr(u, "prompt_tokens", 0) or 0
            c_tok += getattr(u, "completion_tokens", 0) or 0
            extra = getattr(u, "model_extra", None) or {}
            cost += float(extra.get("cost") or 0.0)
            det = getattr(u, "completion_tokens_details", None)
            if det is not None:
                r_tok += getattr(det, "reasoning_tokens", 0) or 0
    _write_log(log_dir, {
        "backend": "openrouter", "model_requested": model, "model_returned": returned,
        "providers_served": served, "reasoning_sent": body.get("reasoning"),
        "provider_pinned": cfg.get("provider"), "finish_reasons": finish,
        "prompt": prompt, "temperature": temperature, "max_tokens": MAX_TOKENS_CEILING, "n": n,
        "completions": completions,
        "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok,
                  "reasoning_tokens": r_tok, "total_tokens": p_tok + c_tok},
        "billed_usd": round(cost, 6), "latency_sec": round(time.time() - t0, 3),
    })
    return completions


# --------------------------------------------------------------------------- openai native
def _complete_openai(model, messages, temperature, n, max_tokens, log_dir, prompt, max_retries, cfg) -> list[str]:
    client = _get_client()
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "n": n}
    if cfg.get("kind") == "reasoning":
        # gpt-5* / o* lock temperature at the service default and spend most of the budget on
        # hidden reasoning tokens, hence the x32 ceiling. No reasoning_effort is sent: the
        # canonical OpenAI runs sent nothing (= provider default, medium).
        kwargs["max_completion_tokens"] = max(max_tokens * 32, 16384)
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
    delay = 1.0
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    completions = [c.message.content or "" for c in resp.choices]
    usage: dict[str, Any] = {}
    if getattr(resp, "usage", None) is not None:
        usage = {"prompt_tokens": resp.usage.prompt_tokens,
                 "completion_tokens": resp.usage.completion_tokens,
                 "total_tokens": resp.usage.total_tokens}
        det = getattr(resp.usage, "completion_tokens_details", None)
        if det is not None:
            usage["completion_tokens_details"] = det.model_dump() if hasattr(det, "model_dump") else dict(det)
    _write_log(log_dir, {
        "backend": "openai", "model_requested": model, "model_returned": getattr(resp, "model", None),
        "response_id": getattr(resp, "id", None), "kind": cfg.get("kind"),
        "base_url": os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        "reasoning_sent": None, "finish_reasons": [c.finish_reason for c in resp.choices],
        "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens, "n": n,
        "completions": completions, "usage": usage, "latency_sec": round(time.time() - t0, 3),
    })
    return completions


def complete(model: str, prompt: str, *, temperature: float = 0.0, max_tokens: int = 1024,
             n: int = 1, log_dir: Path | None = None, max_retries: int = 5) -> list[str]:
    """Single-turn completion, one user message, no system prompt. Returns n strings.

    D1 calls this with temperature=0.8, n=5, max_tokens=900; the judge with temperature=0,
    n=1, max_tokens=400. Those values are the frozen protocol (PROTOCOL.md)."""
    messages = [{"role": "user", "content": prompt}]
    cfg = model_cfg(model)
    b = cfg["backend"]
    if b == "custom":
        return _complete_custom(model, prompt, temperature, n, max_tokens, log_dir, cfg)
    if b == "openrouter":
        return _complete_openrouter(model, messages, temperature, n, log_dir, prompt, max_retries, cfg)
    if b == "openai":
        return _complete_openai(model, messages, temperature, n, max_tokens, log_dir, prompt, max_retries, cfg)
    raise ValueError(f"unknown backend {b!r} for {model!r}")
