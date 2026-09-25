"""Recover failed native API slots while replaying the unchanged probe.

Only exhausted transport errors and malformed judge records are eligible. All
successful subject responses (including empty/length) and valid judges are reused
byte-for-byte. Every attempted replacement remains in recovery/ for an audit.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import client
import probe
from tools import audit_cn_mem_results as audit

CAMPAIGN = 'cn-models-20260924'
HISTORY = ROOT / 'recovery' / CAMPAIGN
api_complete = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def rel(path):
    return str(Path(path).relative_to(ROOT))


def subject_ok(record):
    return record.get('finish_reason') != 'transport_error' and bool(record.get('response_id'))


def judge_ok(raw):
    try:
        sample = raw['per_sample'][0]
        match = re.search(r'\{.*\}', raw['completions'][0], re.S)
        verdict = json.loads(match.group()) if match else {}
        return subject_ok(sample) and isinstance(verdict.get('equivalent'), bool)
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def judge_positions(task_dir, completions):
    candidates = [probe.extract_code(answer) for answer in completions]
    formulas = sorted(p for p in (task_dir / 'formulas').glob('*.py') if not p.stem.startswith('_'))
    return [(fi, si) for fi in range(len(formulas)) for si, candidate in enumerate(candidates) if candidate.strip()]


def sorted_judges(folder):
    return sorted((folder / 'judge_raw').glob('*.json'), key=lambda p: (p.stat().st_mtime_ns, p.name))


def cell_eligible(folder):
    subjects = list((folder / 'd1_raw').glob('*.json'))
    if len(subjects) != 1:
        raise ValueError(f'Expected one original D1 batch: {folder}')
    raw = load(subjects[0])
    invalid = [i for i, sample in enumerate(raw['per_sample']) if not subject_ok(sample)]
    for i in invalid:
        if raw['per_sample'][i].get('finish_reason') != 'transport_error':
            raise ValueError('Missing response evidence is not automatically a retryable transport failure')
    invalid_judges = [p for p in sorted_judges(folder) if not judge_ok(load(p))]
    return invalid, invalid_judges


def first_valid_call(attempt_dir, model, prompt, temperature, max_tokens, validate):
    """Resume first-valid selection across invocation failures; never redo success."""
    attempt_dir.mkdir(parents=True, exist_ok=True)
    previous = sorted(attempt_dir.glob('round-*/*.json'))
    for index, path in enumerate(previous):
        raw = load(path)
        assert raw['model_requested'] == model and raw['prompt'] == prompt
        assert raw['temperature'] == temperature and raw['max_tokens'] == max_tokens and raw['n'] == 1
        if validate(raw):
            assert index == len(previous) - 1, 'Found API attempts after a valid response'
            return path, raw
    for round_id in range(len(previous) + 1, 5):
        round_dir = attempt_dir / f'round-{round_id:02d}'
        assert not round_dir.exists(), f'Unfinished API write; inspect {round_dir}'
        (api_complete or client.complete)(model, prompt, temperature=temperature, max_tokens=max_tokens, n=1, log_dir=round_dir)
        files = list(round_dir.glob('*.json'))
        assert len(files) == 1
        raw = load(files[0])
        if validate(raw):
            return files[0], raw
    raise RuntimeError(f'Recovery attempts exhausted; preserved at {attempt_dir}')


def repair_cell(bench, task, model):
    canonical = ROOT / 'runs' / bench / task / model
    frozen = load(ROOT / 'prompts_frozen.json')
    try:
        audit.audit_cell(bench, task, model, canonical / 'result.json', frozen)
        return {'cell': f'{bench}/{task}/{model}', 'status': 'already_valid'}
    except ValueError:
        pass
    invalid, invalid_judges = cell_eligible(canonical)
    if not invalid and not invalid_judges:
        raise ValueError(f'Cell needs manual review, not API replay: {canonical}')
    task_dir = ROOT / 'bench' / audit.BENCHES[bench][0] / task
    history = HISTORY / bench / task / model
    before = history / 'before'
    if not before.exists():
        before.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(canonical, before)
        manifest = {str(p.relative_to(before)): sha(p) for p in before.rglob('*') if p.is_file()}
        dump(history / 'before-sha256.json', manifest)
    else:
        assert not (history / 'accepted.json').exists(), 'Existing accepted repair needs installation review'
    before_hashes = load(history / 'before-sha256.json')
    for path, digest in before_hashes.items():
        assert sha(before / path) == digest
        assert sha(canonical / path) == digest, 'Original cell changed while recovering'
    source_file = next((before / 'd1_raw').glob('*.json'))
    source = load(source_file)
    assert source['n'] == 5 and len(source['per_sample']) == 5
    d1_prompt = probe.render_d1_prompt(task_dir)
    assert source['prompt'] == d1_prompt
    assert probe.sha(d1_prompt) == frozen[bench][task]
    accepted = copy.deepcopy(source)
    replacement_files = {}
    for index in invalid:
        path, raw = first_valid_call(history / f'subject-{index}', model, d1_prompt, 0.8, 900,
                                    lambda r: subject_ok(r['per_sample'][0]))
        assert audit.scientific_request(raw['per_sample'][0]) == audit.scientific_request(source['per_sample'][index])
        accepted['completions'][index] = raw['completions'][0]
        accepted['per_sample'][index] = raw['per_sample'][0]
        replacement_files[index] = path
    accepted['recovery_note'] = 'Five accepted slots; valid original slots retained, failed API slots recovered independently.'
    accepted['recovery_history'] = rel(history)
    accepted['latency_sec'] = source['latency_sec'] + sum(load(p)['latency_sec'] for p in replacement_files.values())
    old_positions = judge_positions(task_dir, source['completions'])
    old_judges = sorted_judges(before)
    assert len(old_positions) == len(old_judges), 'Missing judge writes require explicit review'
    old_by_position = dict(zip(old_positions, old_judges))
    new_positions = judge_positions(task_dir, accepted['completions'])
    work_root = history / 'replay'
    replay_folder = work_root / task / model
    if replay_folder.exists():
        # Only derived replay files are regenerated; actual API attempts stay above.
        shutil.rmtree(replay_folder)
    ledger = {'cell': f'{bench}/{task}/{model}', 'started_at': datetime.now(timezone.utc).isoformat(),
              'history': rel(history), 'before': rel(before), 'before_manifest': rel(history / 'before-sha256.json'),
              'replaced_subject_indices': invalid,
              'subject_sources': {str(i): rel(p) for i, p in replacement_files.items()}, 'judges': []}
    position_index = 0
    d1_used = False
    original_complete = probe.complete

    def replay_complete(called_model, prompt, *, temperature, max_tokens, n, log_dir, **kwargs):
        nonlocal position_index, d1_used
        log_dir.mkdir(parents=True, exist_ok=True)
        if called_model == model:
            assert not d1_used and prompt == d1_prompt and (temperature, max_tokens, n) == (0.8, 900, 5)
            d1_used = True
            dump(log_dir / 'accepted-five-slots.json', accepted)
            return accepted['completions']
        assert called_model == 'gpt-4.1' and (temperature, max_tokens, n) == (0.0, 400, 1)
        fi, si = new_positions[position_index]
        previous = old_by_position.get((fi, si))
        if previous is not None and si not in invalid:
            previous_raw = load(previous)
            assert previous_raw['prompt'] == prompt, 'Preserved judge prompt mismatch'
        else:
            previous = None
        if previous is not None and judge_ok(load(previous)):
            path, raw, action = previous, load(previous), 'preserved'
        else:
            path, raw = first_valid_call(history / f'judge-{fi:03d}-{si}', 'gpt-4.1', prompt, 0.0, 400, judge_ok)
            action = 'recovered_invalid' if previous is not None else 'new_for_recovered_subject'
        target = log_dir / f'{position_index:05d}.json'
        target.write_bytes(path.read_bytes())
        # Sequential writes plus zero-padded names preserve original judge order.
        ledger['judges'].append({'formula_index': fi, 'sample_index': si, 'action': action,
                                 'source': rel(path), 'canonical_name': target.name,
                                 'original': rel(previous) if previous else None})
        position_index += 1
        return raw['completions']

    try:
        probe.complete = replay_complete
        probe.run_task(task_dir, model, 'gpt-4.1', ['d1'], work_root)
    finally:
        probe.complete = original_complete
    assert d1_used and position_index == len(new_positions)
    audit.audit_cell(bench, task, model, replay_folder / 'result.json', frozen)
    ledger['finished_at'] = datetime.now(timezone.utc).isoformat()
    dump(replay_folder / 'recovery.json', ledger)
    # Recheck original bytes immediately before installing the validated replay.
    for path, digest in before_hashes.items():
        assert sha(canonical / path) == digest
    old = history / 'original-installed-folder'
    assert not old.exists()
    canonical.rename(old)
    replay_folder.rename(canonical)
    dump(history / 'accepted.json', ledger)
    audit.audit_cell(bench, task, model, canonical / 'result.json', frozen)
    return {'cell': ledger['cell'], 'status': 'repaired', 'subject_slots': invalid,
            'preserved_judges': sum(x['action'] == 'preserved' for x in ledger['judges']),
            'new_judges': sum(x['action'] != 'preserved' for x in ledger['judges'])}


def main():
    global api_complete
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells', nargs='*')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--stream-recovery', action='store_true', help='Use streaming only for failed API slots')
    args = parser.parse_args()
    os.environ['OPENAI_API_KEY'] = os.environ['STEPCODE_API_KEY']
    os.environ['OPENAI_BASE_URL'] = os.environ.get('STEPCODE_BASE_URL', 'https://models-proxy.stepfun-inc.com/v1')
    os.environ['MEM_FORCE_BACKEND'] = 'custom'
    os.environ['MEM_TRANSPORT_PROFILE'] = 'native-cn-20260924'
    os.environ['MEM_API_CONCURRENCY'] = '2'
    os.environ['MEM_TRANSPORT_JOURNAL'] = str(HISTORY / 'transport-journal')
    if args.stream_recovery:
        from tools.streaming_mem_recovery import complete as streaming_complete
        api_complete = streaming_complete
    targets = args.cells
    if targets is None:
        targets = []
        for bench in audit.BENCHES:
            for path in sorted((ROOT / 'runs' / bench).glob('*/*/result.json')):
                if path.parent.name not in audit.MODELS:
                    continue
                invalid, judges = cell_eligible(path.parent)
                if invalid or judges:
                    targets.append(f'{bench}/{path.parent.parent.name}/{path.parent.name}')
    print(json.dumps({'eligible_cells': targets}), flush=True)
    if args.plan:
        return
    for cell in targets:
        print(json.dumps(repair_cell(*cell.split('/'))), flush=True)


if __name__ == '__main__':
    main()
