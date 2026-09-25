"""Audit frozen prompts, five actual samples, judge inputs, and reported hit counts.

This script performs no API calls. It can inspect partial runs, and exits nonzero
with --require-complete unless all six active benchmark/model legs are valid.
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

MODELS = ('glm-5.3', 'kimi-k3', 'deepseek-v4.1-flash')
EFFORTS = {'glm-5.3': 'max', 'kimi-k3': 'medium', 'deepseek-v4.1-flash': 'medium'}
BENCHES = {'scilaws': ('scilaws118', 118), 'feynman': ('feynman100', 100)}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def scientific_request(sample):
    request = dict(sample['request'])
    if 'stream' in request or 'stream_options' in request:
        check(sample.get('transport') == 'streaming-recovery-v1', 'Unrecorded streaming transport change')
        check(request.pop('stream', None) is True and request.pop('stream_options', None) == {'include_usage': True},
              'Unexpected recovery transport arguments')
        check(sample.get('read_timeout_seconds') == 600, 'Recovery changed the read timeout')
    return request


def audit_recovery_lineage(folder, raw, task_dir):
    """Validate first-valid selection and immutable original subject/judge evidence."""
    ledger_path = folder / 'recovery.json'
    if not ledger_path.exists():
        return None
    ledger = json.loads(ledger_path.read_text())
    before = ROOT / ledger['before']
    manifests = json.loads((ROOT / ledger['before_manifest']).read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for name, expected in manifests.items():
        check(digest(before / name) == expected, 'Recovery original artifact hash mismatch')
    subjects = list((before / 'd1_raw').glob('*.json'))
    check(len(subjects) == 1, 'Recovery original batch count mismatch')
    original = json.loads(subjects[0].read_text())
    invalid = [i for i, sample in enumerate(original['per_sample']) if sample.get('finish_reason') == 'transport_error']
    check(ledger['replaced_subject_indices'] == invalid, 'Recovery replaced a successful subject sample')
    discarded_errors = sum(len(original['per_sample'][i].get('transport_errors', [])) for i in invalid)
    for index in range(5):
        if index not in invalid:
            check(raw['per_sample'][index] == original['per_sample'][index]
                  and raw['completions'][index] == original['completions'][index],
                  'Recovery changed an original successful subject response')
            continue
        source = ROOT / ledger['subject_sources'][str(index)]
        replacement = json.loads(source.read_text())
        check(replacement['n'] == 1 and replacement['model_requested'] == original['model_requested']
              and replacement['prompt'] == original['prompt'], 'Recovered subject request mismatch')
        check(scientific_request(replacement['per_sample'][0]) == scientific_request(original['per_sample'][index]),
              'Recovered subject generation parameters changed')
        check(raw['per_sample'][index] == replacement['per_sample'][0]
              and raw['completions'][index] == replacement['completions'][0],
              'Recovered subject does not match its accepted API record')
        attempts = sorted(source.parent.parent.glob('round-*/*.json'))
        check(attempts[-1] == source, 'Recovery called subject again after success')
        for previous in attempts[:-1]:
            p = json.loads(previous.read_text())
            discarded_errors += len(p['per_sample'][0].get('transport_errors', []))
            check(scientific_request(p['per_sample'][0]) == scientific_request(replacement['per_sample'][0]), 'Subject recovery attempt parameters changed')
            check(p['per_sample'][0].get('finish_reason') == 'transport_error',
                  'Recovery skipped a valid subject response')
    candidates = [probe.extract_code(answer) for answer in original['completions']]
    formulas = sorted(p for p in (task_dir / 'formulas').glob('*.py') if not p.stem.startswith('_'))
    positions = [(fi, si) for fi in range(len(formulas)) for si, code in enumerate(candidates) if code.strip()]
    old_judges = sorted((before / 'judge_raw').glob('*.json'), key=lambda p: (p.stat().st_mtime_ns, p.name))
    check(len(positions) == len(old_judges), 'Original judge positions missing')
    old_lookup = dict(zip(positions, old_judges))
    def valid_judge(record):
        try:
            sample = record['per_sample'][0]
            match = re.search(r'\{.*\}', record['completions'][0], re.S)
            verdict = json.loads(match.group()) if match else {}
            return sample.get('finish_reason') != 'transport_error' and bool(sample.get('response_id')) and isinstance(verdict.get('equivalent'), bool)
        except (ValueError, TypeError, IndexError, KeyError):
            return False
    expected_positions = [(fi, si) for fi in range(len(formulas))
                          for si, answer in enumerate(raw['completions']) if probe.extract_code(answer).strip()]
    check([(x['formula_index'], x['sample_index']) for x in ledger['judges']] == expected_positions,
          'Recovered judge position mapping mismatch')
    preserved = 0
    for item in ledger['judges']:
        key = (item['formula_index'], item['sample_index'])
        source = ROOT / item['source']
        destination = folder / 'judge_raw' / item['canonical_name']
        check(source.read_bytes() == destination.read_bytes(), 'Recovery changed accepted judge bytes')
        previous = old_lookup.get(key)
        if previous is not None and key[1] not in invalid and valid_judge(json.loads(previous.read_text())):
            check(item['action'] == 'preserved' and source == previous,
                  'Recovery replaced a valid original judge')
            preserved += 1
        else:
            check(item['action'] != 'preserved', 'Recovered judge lacks appropriate eligibility')
            if previous is not None:
                discarded_errors += len(json.loads(previous.read_text())['per_sample'][0].get('transport_errors', []))
            check(key[1] in invalid or (previous is not None and not valid_judge(json.loads(previous.read_text()))),
                  'Recovered judge was neither missing for a recovered subject nor invalid')
            attempts = sorted(source.parent.parent.glob('round-*/*.json'))
            check(attempts[-1] == source, 'Recovery called judge again after valid verdict')
            check(all(not valid_judge(json.loads(p.read_text())) for p in attempts[:-1]),
                  'Recovery skipped a valid judge response')
            accepted_request = scientific_request(json.loads(source.read_text())['per_sample'][0])
            for attempt in attempts[:-1]:
                previous_raw = json.loads(attempt.read_text())
                discarded_errors += len(previous_raw['per_sample'][0].get('transport_errors', []))
                check(scientific_request(previous_raw['per_sample'][0]) == accepted_request, 'Judge recovery attempt parameters changed')
    return {'replaced_subject_slots': invalid, 'preserved_judges': preserved, 'history': ledger['history'],
            'discarded_transport_errors': discarded_errors}


def reviewed_usage_deviation(folder, raw_path, index, sample, answer):
    path = folder / 'provider_usage_review.json'
    if not path.exists():
        return None
    review = json.loads(path.read_text())
    if (review.get('classification') != 'provider_reported_one_token_overrun_with_no_visible_answer'
            or review.get('raw_file') != raw_path.name
            or review.get('raw_sha256') != hashlib.sha256(raw_path.read_bytes()).hexdigest()):
        return None
    request = scientific_request(sample)
    usage = sample.get('usage', {})
    reported = usage.get('completion_tokens')
    reasoning = usage.get('completion_tokens_details', {}).get('reasoning_tokens')
    if not (request.get('max_tokens') == 28800 and reported == 28801 and reasoning == 28801
            and sample.get('finish_reason') == 'length' and answer == ''
            and usage.get('total_tokens') == usage.get('prompt_tokens', 0) + reported):
        return None
    expected = {'sample_index': index, 'response_id': sample.get('response_id'),
                'requested_max_tokens': 28800, 'reported_completion_tokens': 28801,
                'reported_reasoning_tokens': 28801, 'finish_reason': 'length',
                'visible_answer_characters': 0}
    if expected not in review.get('samples', []):
        return None
    return {**expected, 'review_path': str(path.relative_to(ROOT)),
            'classification': review['classification'], 'raw_sha256': review['raw_sha256']}


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
    provider_usage_deviations = []
    for sample_index, sample in enumerate(raw['per_sample']):
        request = scientific_request(sample)
        check(request == {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                          'n': 1, 'max_tokens': 28800, 'reasoning_effort': EFFORTS[model]}, 'Subject request protocol mismatch')
        check(sample.get('finish_reason') != 'transport_error', 'Exhausted subject transport retries')
        check(sample.get('model_returned', '').startswith(model), 'Unexpected served subject model')
        check(bool(sample.get('response_id')), 'Missing subject response ID')
        if (sample.get('usage', {}).get('completion_tokens') or 0) > 28800:
            deviation = reviewed_usage_deviation(folder, subject_logs[0], sample_index, sample, raw['completions'][sample_index])
            check(deviation is not None, 'Subject exceeded the requested output cap without reviewed evidence')
            provider_usage_deviations.append(deviation)
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
            check(scientific_request(call) == expected_request, 'Judge request protocol mismatch')
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
    recovery = audit_recovery_lineage(folder, raw, task_dir)
    if recovery:
        retries += recovery['discarded_transport_errors']
    return {'bench': bench, 'task': task, 'model': model, 'type': result.get('type'),
            'recovery': recovery, 'provider_usage_deviations': provider_usage_deviations,
            'formula_hits': formula_hits, 'subject_responses': 5, 'judge_calls': expected_judges,
            'empty_samples': empty, 'transport_retry_errors': retries,
            'finish_reasons': dict(finish_reasons), 'subject_usage': dict(subject_usage),
            'judge_usage': dict(judge_usage),
            'result_path': str(result_path.relative_to(ROOT)),
            'subject_raw_path': str(subject_logs[0].relative_to(ROOT))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT / '.aris/cn-models-20260924/audit-live.json')
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
              'complete': not missing and not failures and len(checked) == 654,
              'expected_cells': 654, 'audited_cells': len(checked), 'per_leg': per_leg,
              'subject_responses': sum(c['subject_responses'] for c in checked),
              'judge_calls': sum(c['judge_calls'] for c in checked),
              'empty_samples': sum(c['empty_samples'] for c in checked),
              'transport_retry_errors': sum(c['transport_retry_errors'] for c in checked),
              'missing': missing, 'failures': failures, 'cells': checked}
    record['provider_reported_cap_overruns'] = sum(len(c.get('provider_usage_deviations', [])) for c in checked)
    record['strict_reported_token_cap_compliant'] = record['provider_reported_cap_overruns'] == 0
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
