"""Fill this in ONLY if your API is not OpenAI-compatible (otherwise just set OPENAI_BASE_URL).

Then run with:  export MEM_FORCE_BACKEND=custom   (routes every call, subjects AND the judge, here)

Contract — return exactly `n` completions (strings) for ONE user message `prompt`:
  * temperature : use it if your API accepts it (0.8 for the recall probe, 0 for the judge);
                  reasoning models that reject a temperature may ignore it.
  * max_tokens  : output cap. 900 for the probe / 400 for the judge on chat models; for
                  reasoning models use a large cap (>= 28800) so the visible answer is not cut.
  * kind        : "chat" or "reasoning" (from models.json) — tells you which of the above applies.
  * Do NOT add a system prompt, few-shot examples, or any extra text. Send `prompt` verbatim.
  * If a call fails after your own retries, return "" for that sample (scored as no answer).
"""
from __future__ import annotations


def complete(model: str, prompt: str, temperature: float, n: int, max_tokens: int, kind: str) -> list[str]:
    raise NotImplementedError(
        "custom_client.complete: wire this to your LLM API (see docstring), "
        "or unset MEM_FORCE_BACKEND / set OPENAI_BASE_URL for an OpenAI-compatible endpoint")
    # Example for an OpenAI-compatible SDK object `my_client`:
    # outs = []
    # for _ in range(n):
    #     kw = dict(model=model, messages=[{"role": "user", "content": prompt}])
    #     if kind == "reasoning":
    #         kw["max_completion_tokens"] = max(max_tokens * 32, 16384)
    #     else:
    #         kw["max_tokens"] = max_tokens; kw["temperature"] = temperature
    #     r = my_client.chat.completions.create(**kw)
    #     outs.append(r.choices[0].message.content or "")
    # return outs
