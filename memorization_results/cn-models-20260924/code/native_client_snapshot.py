"""Native reasoning model transport for the 2026-09-24 memorization campaign.

Each of five samples is an independent n=1 request. A process-wide semaphore
bounds API concurrency, including judge calls. The canonical prompts and judge
rule are supplied by the unchanged upstream runner.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

PROFILES = {'glm-5.3': 'max', 'kimi-k3': 'medium',
            'deepseek-v4.1-flash': 'medium', 'step-5-preview': 'medium'}
_client = None
_client_lock = threading.Lock()
_slots = threading.BoundedSemaphore(int(os.environ.get('MEM_API_CONCURRENCY', '12')))


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from openai import OpenAI
                _client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], timeout=600, max_retries=0)
    return _client


def complete(model, prompt, temperature, n, max_tokens, kind):
    client = _get_client()
    call_id = uuid.uuid4().hex
    def sample(index):
        request = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'n': 1}
        if kind == 'reasoning':
            request['max_tokens'] = max(max_tokens * 32, 16384)
            request['reasoning_effort'] = PROFILES[model]
        else:
            request.update(max_tokens=max_tokens, temperature=temperature)
        started = time.monotonic(); errors = []; response = None
        for attempt in range(5):
            try:
                with _slots:
                    response = client.chat.completions.create(**request)
                if len(response.choices) != 1:
                    raise RuntimeError(f'Expected one choice, got {len(response.choices)}')
                break
            except Exception as exc:
                response = None
                message = str(exc)
                key = os.environ.get('OPENAI_API_KEY')
                if key: message = message.replace(key, '[REDACTED]')
                errors.append({'attempt': attempt+1, 'type': type(exc).__name__, 'message': message[:2000]})
                if attempt < 4: time.sleep(min(2**attempt, 16))
        record = {'content': '', 'call_id': call_id, 'sample_index': index,
                  'request': request, 'transport_errors': errors,
                  'attempts': len(errors)+int(response is not None),
                  'latency_sec': round(time.monotonic()-started, 3)}
        if response is not None:
            choice=response.choices[0]; message=choice.message
            record.update(content=message.content or '', response_id=response.id,
                          model_returned=response.model, finish_reason=choice.finish_reason,
                          reasoning_content=getattr(message, 'reasoning_content', None) or getattr(message, 'reasoning', None) or '',
                          usage=response.usage.model_dump() if response.usage else {})
        else:
            record['finish_reason']='transport_error'
        if os.environ.get('MEM_TRANSPORT_JOURNAL'):
            target=Path(os.environ['MEM_TRANSPORT_JOURNAL']);target.mkdir(parents=True,exist_ok=True)
            path=target/f'{call_id}-{index}.json'
            path.write_text(json.dumps({'model':model,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),**record},ensure_ascii=False)+'\n')
        return record
    if n==1: return [sample(0)]
    with ThreadPoolExecutor(max_workers=min(n,5)) as executor:
        return list(executor.map(sample,range(n)))
