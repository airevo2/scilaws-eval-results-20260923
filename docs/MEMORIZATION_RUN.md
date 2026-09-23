本次运行使用 Python 3.12.3、openai 3.14.1、PyYAML 6.0.3，通过 OpenAI-compatible MP 接口调用 GPT-5.6-Sol、GPT-6-Astra 与判分模型 GPT-4.1。

本次接口兼容性检查发现，向两个被测模型发送 `n=5` 时，服务只返回一条回答。因此使用上游预留的 `custom_client.py` 接口，将一次五样本调用实现为五次独立的 `n=1` 请求。每次均发送相同的单条 user prompt，不增加 system prompt 或示例。

| 参数 | 被测模型 | 判分模型 |
|---|---|---|
| 模型 | GPT-5.6-Sol、GPT-6-Astra | GPT-4.1 |
| 每题采样 | 5 条独立回答 | 每条候选、每个参考公式判分 1 次 |
| 输出限制 | `max_completion_tokens=28800` | `max_tokens=400` |
| temperature | 不显式发送 | `0.0` |
| reasoning effort | 不显式发送；同一服务事后检查回显 `medium` | 不设置 |
| 输入 | 原始单条 user prompt | 原始结构等价判分 prompt |

每次 API 调用最多尝试 5 次，仅对传输/API 异常重试，重试错误记录在原始调用日志中。有效回答不按分数重采样。传输失败耗尽与空回答会单独统计，不能作为无异常运行处理。

两个 benchmark 分别使用 12 个任务 worker。冒烟阶段每个 benchmark 先运行 2 个任务、两个模型；通过检查的冒烟结果纳入正式结果，其任务从后续批次中排除。

复现时，在环境中配置 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`，然后运行：

```bash
python tests/mock_e2e.py
python tests/test_mp_transport.py
python run_mem.py --dry-run --bench scilaws
python run_mem.py --dry-run --bench feynman
MEM_FORCE_BACKEND=custom python run_mem.py --bench scilaws --models gpt-5.6-sol,gpt-6-astra --workers 12
MEM_FORCE_BACKEND=custom python run_mem.py --bench feynman --models gpt-5.6-sol,gpt-6-astra --workers 12
python check_run.py
python run_mem.py --pack
```

重复执行全量命令会覆盖已有任务结果。恢复中断批次时，使用 `--tasks` 指定缺失任务，并保留此前调用记录。

原始结果位于 `runs/<bench>/<task>/<model>/`。`d1_raw/` 保存五次调用的实际请求参数、回答、响应 ID、模型 ID、用量和传输重试信息；`judge_raw/` 保存判分调用；`result.json` 保存逐参考公式命中数。公开交付时会将这些目录完整打包，并提供文件级校验清单。
