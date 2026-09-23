"""MP transport for the frozen memorization protocol.

MP's Chat Completions compatibility endpoint returns one choice for an n=5
request on the evaluated models. This adapter issues n independent n=1 requests.
Each request contains the same single user prompt and the protocol parameters.
No API credential is written to the returned raw-call records.
"""
from __future__ import annotations

import os
import threading
import time

_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from openai import OpenAI
                _client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],
                                 timeout=600.0, max_retries=0)
    return _client


def complete(model: str, prompt: str, temperature: float, n: int,
             max_tokens: int, kind: str) -> list[dict]:
    client = _get_client()
    records = []
    for sample_index in range(n):
        request = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'n': 1}
        if kind == 'reasoning':
            request['max_completion_tokens'] = max(max_tokens * 32, 16384)
        else:
            request.update(max_tokens=max_tokens, temperature=temperature)
        started = time.monotonic()
        errors = []
        response = None
        for attempt in range(5):
            try:
                response = client.chat.completions.create(**request)
                if len(response.choices) != 1:
                    raise RuntimeError(f'Expected one choice, received {len(response.choices)}')
                break
            except Exception as exc:
                response = None
                message = str(exc)
                key = os.environ.get('OPENAI_API_KEY')
                if key:
                    message = message.replace(key, '[REDACTED]')
                errors.append({'attempt': attempt + 1, 'type': type(exc).__name__,
                               'message': message[:2000]})
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 16))
        record = {'content': '', 'sample_index': sample_index,
                  'request': request, 'attempts': len(errors) + int(response is not None),
                  'transport_errors': errors,
                  'latency_sec': round(time.monotonic() - started, 3)}
        if response is not None:
            choice = response.choices[0]
            record.update(content=choice.message.content or '', response_id=response.id,
                          model_returned=response.model, finish_reason=choice.finish_reason,
                          usage=response.usage.model_dump() if response.usage is not None else {})
        else:
            record['finish_reason'] = 'transport_error'
        records.append(record)
    return records
