"""Audit frozen prompts, five actual samples, judge inputs, and reported hit counts.

This script performs no API calls. It can inspect partial runs, and exits nonzero
with --require-complete unless all four requested benchmark/model legs are valid.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import probe

MODELS = ('gpt-5.6-sol', 'gpt-6-astra')
BENCHES = {'scilaws': ('scilaws118', 118), 'feynman': ('feynman100', 100)}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit_cell(bench, task, model, result_path, frozen):
    folder = result_path.parent
    result = json.loads(result_path.read_text())
    check(result['model'] == model and result['task'] == task, 'Result identity mismatch')
    check(result['judge_model'] == 'gpt-4.1', 'Wrong judge model')
    check(result['judge_parse_failures'] == 0, 'Reported judge parse failures')
    task_dir = ROOT / 'bench' / BENCHES[bench][0] / task
    prompt = probe.render_d1_prompt(task_dir)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    check(prompt_hash == frozen[bench][task], 'Frozen prompt mismatch')
    check(result['d1_prompt_sha256_16'] == prompt_hash, 'Recorded prompt hash mismatch')
    subject_logs = list((folder / 'd1_raw').glob('*.json'))
    check(len(subject_logs) == 1, f'Expected one accepted D1 batch; found {len(subject_logs)}')
    raw = json.loads(subject_logs[0].read_text())
    check(raw['model_requested'] == model and raw['n'] == 5, 'D1 batch model/count mismatch')
    check(raw['prompt'] == prompt, 'Actual D1 prompt mismatch')
    check(len(raw['completions']) == 5 and len(raw['per_sample']) == 5, 'Not five recorded samples')
    ids = []
    retries = 0
    finish_reasons = Counter()
    subject_usage = Counter()
    for sample in raw['per_sample']:
        request = sample['request']
        check(request == {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                          'n': 1, 'max_completion_tokens': 28800}, 'Subject request protocol mismatch')
        check(sample.get('finish_reason') != 'transport_error', 'Exhausted subject transport retries')
        check(sample.get('model_returned', '').startswith(model), 'Unexpected served subject model')
        check(bool(sample.get('response_id')), 'Missing subject response ID')
        check((sample.get('usage', {}).get('completion_tokens') or 0) <= 28800, 'Subject exceeded the requested output cap')
        ids.append(sample['response_id'])
        retries += len(sample.get('transport_errors', []))
        finish_reasons[sample.get('finish_reason', 'unknown')] += 1
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            subject_usage[key] += sample.get('usage', {}).get(key, 0) or 0
    check(len(set(ids)) == 5, 'Duplicate subject response IDs within the five-sample batch')
    candidates = [probe.extract_code(answer) for answer in raw['completions']]
    check((folder / 'd1_samples.txt').read_text() == '\n\n=====\n\n'.join(candidates),
          'Saved candidate code differs from the selected raw responses')
    empty = sum(not candidate.strip() for candidate in candidates)
    check(result['n_empty_d1_samples'] == empty, 'Empty-answer count mismatch')
    meta = probe.load_meta(task_dir)
    target, inputs = probe.io_spec(meta)
    reference_files = sorted(p for p in (task_dir / 'formulas').glob('*.py') if not p.stem.startswith('_'))
    check([f['formula'] for f in result['formulas']] == [p.stem for p in reference_files],
          'Reference formula identity/order mismatch')
    judge_files = sorted((folder / 'judge_raw').glob('*.json'), key=lambda p: (p.stat().st_mtime_ns, p.name))
    expected_judges = len(reference_files) * (5 - empty)
    check(len(judge_files) == expected_judges,
          f'Judge-call count mismatch: {len(judge_files)} vs {expected_judges}')
    index = 0
    judge_usage = Counter()
    formula_hits = {}
    for reference, reported in zip(reference_files, result['formulas']):
        hits = 0
        for candidate in candidates:
            if not candidate.strip():
                continue
            expected_prompt = probe.JUDGE_PROMPT.format(
                t_sym=target['sym'], inputs=', '.join(i['sym'] for i in inputs) or '(none)',
                gt_code=probe.gt_code(reference)[0], cand_code=candidate)
            judge = json.loads(judge_files[index].read_text())
            index += 1
            check(judge['model_requested'] == 'gpt-4.1' and judge['n'] == 1,
                  'Judge model/count mismatch')
            check(judge['prompt'] == expected_prompt, 'Judge prompt/reference/candidate mismatch')
            check(len(judge['completions']) == 1 and len(judge['per_sample']) == 1,
                  'Judge raw-record count mismatch')
            call = judge['per_sample'][0]
            expected_request = {'model': 'gpt-4.1', 'messages': [{'role': 'user', 'content': expected_prompt}],
                                'n': 1, 'max_tokens': 400, 'temperature': 0.0}
            check(call['request'] == expected_request, 'Judge request protocol mismatch')
            check(call.get('finish_reason') != 'transport_error', 'Exhausted judge transport retries')
            check(bool(re.fullmatch(r'gpt-4\.1(?:-\d{4}-\d{2}-\d{2})?', call.get('model_returned', ''))), 'Unexpected served judge model')
            check((call.get('usage', {}).get('completion_tokens') or 0) <= 400, 'Judge exceeded the requested output cap')
            retries += len(call.get('transport_errors', []))
            match = re.search(r'\{.*\}', judge['completions'][0], re.S)
            verdict = json.loads(match.group()) if match else {}
            check(isinstance(verdict.get('equivalent'), bool), 'Judge did not return a JSON boolean')
            hits += int(verdict['equivalent'])
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                judge_usage[key] += call.get('usage', {}).get(key, 0) or 0
        check(reported['d1_hits'] == hits, 'Reported hit count differs from raw judge decisions')
        check(reported['d1_rate'] == round(hits / 5, 3), 'Reported hit rate mismatch')
        formula_hits[reference.stem] = hits
    return {'bench': bench, 'task': task, 'model': model, 'type': result.get('type'),
            'formula_hits': formula_hits, 'subject_responses': 5, 'judge_calls': expected_judges,
            'empty_samples': empty, 'transport_retry_errors': retries,
            'finish_reasons': dict(finish_reasons), 'subject_usage': dict(subject_usage),
            'judge_usage': dict(judge_usage),
            'result_path': str(result_path.relative_to(ROOT)),
            'subject_raw_path': str(subject_logs[0].relative_to(ROOT))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT / '.aris/memorization-20260923/audit-live.json')
    args = parser.parse_args()
    frozen = json.loads((ROOT / 'prompts_frozen.json').read_text())
    checked, missing, failures = [], [], []
    per_leg = {}
    for bench, (asset_dir, count) in BENCHES.items():
        tasks = probe.load_tasks(ROOT / 'tasks' / (asset_dir + '.json'))
        check(len(tasks) == count, 'Unexpected task grid size')
        for model in MODELS:
            n = 0
            for task in tasks:
                path = ROOT / 'runs' / bench / task / model / 'result.json'
                identity = f'{bench}/{task}/{model}'
                if not path.exists():
                    missing.append(identity)
                    continue
                try:
                    checked.append(audit_cell(bench, task, model, path, frozen))
                    n += 1
                except Exception as error:
                    failures.append({'cell': identity, 'error': str(error)})
            per_leg[bench + '/' + model] = {'audited': n, 'expected': count}
    record = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'complete': not missing and not failures and len(checked) == 436,
              'expected_cells': 436, 'audited_cells': len(checked), 'per_leg': per_leg,
              'subject_responses': sum(c['subject_responses'] for c in checked),
              'judge_calls': sum(c['judge_calls'] for c in checked),
              'empty_samples': sum(c['empty_samples'] for c in checked),
              'transport_retry_errors': sum(c['transport_retry_errors'] for c in checked),
              'missing': missing, 'failures': failures, 'cells': checked}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ['complete', 'audited_cells', 'expected_cells',
                      'per_leg', 'subject_responses', 'judge_calls', 'empty_samples',
                      'transport_retry_errors']}, indent=2))
    if failures:
        print('Audit failures:', json.dumps(failures[:10], ensure_ascii=False))
    if args.require_complete and not record['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
