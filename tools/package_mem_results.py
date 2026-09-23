"""Create a public, checksum-verified delivery after the strict memory audit passes."""
from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'memorization_results/gpt56-gpt6-20260923'
CONTROL = ROOT / '.aris/memorization-20260923'
MODELS = ('gpt-5.6-sol', 'gpt-6-astra')
SECRET_PATTERNS = [re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),
                   re.compile(rb'github_pat_[A-Za-z0-9_]{40,}'),
                   re.compile(rb'hf_[A-Za-z0-9]{25,}'),
                   re.compile(rb'\bsk-[A-Za-z0-9_-]{24,}')]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / 'tools/audit_mem_results.py'),
                    '--require-complete', '--out', str(OUT / 'AUDIT.json')], cwd=ROOT, check=True)
    audit = json.loads((OUT / 'AUDIT.json').read_text())
    assert audit['complete'] and audit['audited_cells'] == 436 and audit['subject_responses'] == 2180
    checked = subprocess.run([sys.executable, str(ROOT / 'check_run.py')], cwd=ROOT,
                             text=True, capture_output=True, check=True)
    (OUT / 'CHECK_RUN.txt').write_text(checked.stdout + checked.stderr)
    frozen = json.loads((CONTROL / 'source-hashes.json').read_text())
    configuration = json.loads((CONTROL / 'env-spec.json').read_text())
    assert hashlib.sha256((ROOT / 'custom_client.py').read_bytes()).hexdigest() == configuration['adapter_sha256']
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
               for name, expected in frozen.items())
    best = json.loads((ROOT / 'data/best_baseline_map.json').read_text())
    rows = []
    for cell in audit['cells']:
        formula = best[cell['task']]['formula'] if cell['bench'] == 'scilaws' else next(iter(cell['formula_hits']))
        assert formula in cell['formula_hits']
        hits = cell['formula_hits'][formula]
        rows.append({'bench': cell['bench'], 'model': cell['model'], 'task': cell['task'],
                     'type': cell['type'], 'reference_formula': formula, 'd1_hits': hits,
                     'd1_rate': hits / 5, 'cold_recalled': int(hits >= 3),
                     'empty_samples': cell['empty_samples'], 'result_path': cell['result_path']})
    rows.sort(key=lambda r: (r['bench'], r['model'], r['task']))
    with (OUT / 'per_task.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary_rows = []
    for bench, expected in [('scilaws', 118), ('feynman', 100)]:
        for model in MODELS:
            selected = [row for row in rows if row['bench'] == bench and row['model'] == model]
            assert len(selected) == expected
            histogram = Counter(row['d1_hits'] for row in selected)
            summary_rows.append({'bench': bench, 'model': model, 'tasks': expected,
                                 'cold_recalled': sum(row['cold_recalled'] for row in selected),
                                 'cold_recall_rate': sum(row['cold_recalled'] for row in selected) / expected,
                                 'best_formula_sample_hits': sum(row['d1_hits'] for row in selected),
                                 'samples': expected * 5,
                                 'hit_histogram': {str(i): histogram[i] for i in range(6)}})
    finish = Counter()
    subject_usage = Counter(); judge_usage = Counter()
    for cell in audit['cells']:
        finish.update(cell['finish_reasons'])
        subject_usage.update(cell['subject_usage']); judge_usage.update(cell['judge_usage'])
    summary = {'run_id': 'gpt56-gpt6-20260923', 'complete': True,
               'completed_at': datetime.now(timezone.utc).isoformat(),
               'rule': 'At least 3 of 5 samples match the task best-baseline functional form.',
               'subject_models': list(MODELS), 'judge_model': 'gpt-4.1',
               'source_commit': 'f036d101a08b4554dc2f99e7e7da7c5d4b2be6d9',
               'scientific_cells': 436, 'subject_completions': 2180,
               'judge_calls': audit['judge_calls'], 'empty_subject_samples': audit['empty_samples'],
               'transport_retry_errors': audit['transport_retry_errors'],
               'subject_finish_reasons': dict(finish), 'subject_usage': dict(subject_usage),
               'judge_usage': dict(judge_usage), 'billed_cost_usd': None,
               'results': summary_rows}
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    copied = {'env-spec.json': 'RUN_CONFIG.json', 'source-hashes.json': 'FROZEN_SOURCE_HASHES.json',
              'api_preflight.json': 'API_PREFLIGHT.json',
              'environment-validation.json': 'ENVIRONMENT_VALIDATION.json'}
    for source, target in copied.items():
        shutil.copy2(CONTROL / source, OUT / target)
    manifest = {'run_id': summary['run_id'], 'primary_cells': 436,
                'subject_completions': 2180, 'archives': [], 'files': []}
    for bench in ('scilaws', 'feynman'):
        for model in MODELS:
            paths = sorted(p for folder in (ROOT / 'runs' / bench).glob('*/' + model)
                           for p in folder.rglob('*') if p.is_file())
            archive_name = bench + '-' + model + '.zip'
            with zipfile.ZipFile(OUT / archive_name, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for path in paths:
                    data = path.read_bytes()
                    assert not any(pattern.search(data) for pattern in SECRET_PATTERNS), str(path)
                    relative = str(path.relative_to(ROOT))
                    archive.writestr(relative, data)
                    manifest['files'].append({'path': relative, 'size': len(data),
                                              'sha256': hashlib.sha256(data).hexdigest(),
                                              'archive': archive_name})
            archive_path = OUT / archive_name
            manifest['archives'].append({'filename': archive_name, 'files': len(paths),
                                         'bytes': archive_path.stat().st_size,
                                         'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest()})
    (OUT / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    for entry in manifest['archives']:
        with zipfile.ZipFile(OUT / entry['filename']) as archive:
            records = [row for row in manifest['files'] if row['archive'] == entry['filename']]
            assert len(archive.namelist()) == len(records)
            for row in records:
                data = archive.read(row['path'])
                assert len(data) == row['size'] and hashlib.sha256(data).hexdigest() == row['sha256']
    table = '\n'.join(f"| {row['model']} | {row['bench']} | {row['cold_recalled']}/{row['tasks']} | {100 * row['cold_recall_rate']:.2f}% |"
                      for row in summary_rows)
    report = f'''已完成 GPT-5.6-Sol 与 GPT-6-Astra 的公式冷回忆 D1 检测，共 436 个任务—模型组合、2,180 条实际生成的回答。每个模型分别覆盖 SciLaws 118 题与 AI-Feynman 100 题。

| 模型 | Benchmark | 可召回任务数 | 可召回比例 |
|---|---|---:|---:|
{table}

判定规则沿用上游：同一任务的五条回答中，至少三条与指定 best-baseline 参考公式具有相同函数结构，即记为可召回。SciLaws 使用 `data/best_baseline_map.json` 指定的参考公式；AI-Feynman 每题使用其单一参考公式。逐任务命中数和选用的参考公式见 `per_task.csv`。

被测模型输出上限为 28,800 token；不显式发送 temperature 或 reasoning effort。同一 MP 服务的默认 effort 已检查为 medium。评委为 GPT-4.1，temperature=0、max_tokens=400。检测输入只含目标量和输入变量的文字描述，不提供观测数据。

MP 的 n=5 兼容接口只返回一条回答。本次通过上游允许的 custom 接口发送五次独立 n=1 请求。实际请求参数、回答、响应 ID、模型 ID、用量、结束原因和传输重试记录均保存在各压缩包中。环境与适配器检查记录见 `API_PREFLIGHT.json`、`ENVIRONMENT_VALIDATION.json` 和 `RUN_CONFIG.json`。

完整性检查已逐条核对全部 436 个组合：五次实际回答、固定 prompt 哈希、判分请求和参考公式、原始判分 JSON，以及从原始判分重新计算的命中数。原始 `check_run.py` 同时通过。核对细节见 `AUDIT.json` 和 `CHECK_RUN.txt`。

本次空回答数为 {audit['empty_samples']}，触及输出上限的回答数为 {finish.get('length', 0)}；发生 {audit['transport_retry_errors']} 次传输/API 异常重试，最终用于评分的调用均有成功响应。有效回答未按得分重采样。API 未提供可用的美元计费信息。

四个 ZIP 压缩包采用 `runs/<bench>/<task>/<model>/` 目录结构，可解压到同一个目录。包内文件的 SHA-256 与大小记录在 `MANIFEST.json`，交付文件校验值记录在 `SHA256SUMS`。解压后的 `d1_raw/` 保存回答调用，`judge_raw/` 保存判分调用，`result.json` 保存逐参考公式结果。

无数据召回反映模型的已有知识或推导能力，不能单独证明训练数据污染；未召回也不能证明任务未出现在训练数据中。这里报告的是本次两个模型的独立检测结果。
'''
    (OUT / 'README.md').write_text(report)
    hashes = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != 'SHA256SUMS':
            content = path.read_bytes()
            if path.suffix != '.zip':
                assert not any(pattern.search(content) for pattern in SECRET_PATTERNS), str(path)
            hashes.append(hashlib.sha256(content).hexdigest() + '  ' + path.name)
    (OUT / 'SHA256SUMS').write_text('\n'.join(hashes) + '\n')
    baseline = json.loads((ROOT / 'agent_baseline_results/gpt56-gpt6-20260923/collected_results.json').read_text())
    baseline_table = []
    for model in MODELS:
        real = next(r for r in baseline['table'] if r['model'] == model and r['mode'] == 'real' and r['type'] == 'ALL')
        parallel = next(r for r in baseline['table'] if r['model'] == model and r['mode'] == 'parallel' and r['type'] == 'ALL')
        baseline_table.append(f"| {model} | {real['S_N']:.6f} | {real['S_V']:.6f} | {parallel['S_S']:.6f} |")
    root_readme = f'''本仓库保存 GPT-5.6-Sol 与 GPT-6-Astra 的两组完整评测结果：SciLaws-Bench agent baseline，以及公式冷回忆 D1 检测。

Agent baseline 已完成 472 个运行组合，两模型分别覆盖 Real 与 Parallel 模式各 118 个任务。S_N 为数值拟合得分，S_V 为科学有效性得分，S_S 为 Parallel 模式的结构恢复得分。完整报告和原始轨迹位于 `agent_baseline_results/gpt56-gpt6-20260923/`。

| 模型 | Real S_N | Real S_V | Parallel S_S |
|---|---:|---:|---:|
{chr(10).join(baseline_table)}

公式冷回忆检测已完成 436 个任务—模型组合、2,180 条回答。输入只包含目标量和变量描述，不提供观测数据；每题采样五次，由 GPT-4.1 判断与参考公式的结构等价性，命中至少三次即记为可召回。

| 模型 | Benchmark | 可召回任务数 | 可召回比例 |
|---|---|---:|---:|
{table}

Mem 逐任务结果、四个原始记录压缩包、审计结果及校验清单位于 `memorization_results/gpt56-gpt6-20260923/`。两组评测均保留原始记录及运行限制说明。

上游检测程序来自 `tREeFrOGcoder/scilaw_mem_pack_20260922`，固定提交 `f036d101a08b4554dc2f99e7e7da7c5d4b2be6d9`。原始 README 保留在 `docs/MEMORIZATION_KIT_README.md`；固定检测协议见 `PROTOCOL.md`，本次运行方法及 MP 接口适配见 `docs/MEMORIZATION_RUN.md`。复核工具 `tools/audit_mem_results.py` 可从解压后的原始记录重新验证全部命中数，不调用模型 API。

该冷回忆检测衡量模型在无数据条件下恢复已知公式的能力，不能单独证明训练数据污染，也不能以未召回作为无污染证明。
'''
    (ROOT / 'README.md').write_text(root_readme)
    print('Packaged complete memorization results:', OUT)
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print('Verified raw files:', len(manifest['files']))


if __name__ == '__main__':
    main()
