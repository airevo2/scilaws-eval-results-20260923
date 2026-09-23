已完成 GPT-5.6-Sol 与 GPT-6-Astra 的公式冷回忆 D1 检测，共 436 个任务—模型组合、2,180 条实际生成的回答。每个模型分别覆盖 SciLaws 118 题与 AI-Feynman 100 题。

| 模型 | Benchmark | 可召回任务数 | 可召回比例 |
|---|---|---:|---:|
| gpt-5.6-sol | scilaws | 41/118 | 34.75% |
| gpt-6-astra | scilaws | 40/118 | 33.90% |
| gpt-5.6-sol | feynman | 65/100 | 65.00% |
| gpt-6-astra | feynman | 76/100 | 76.00% |

判定规则沿用上游：同一任务的五条回答中，至少三条与指定 best-baseline 参考公式具有相同函数结构，即记为可召回。SciLaws 使用 `data/best_baseline_map.json` 指定的参考公式；AI-Feynman 每题使用其单一参考公式。逐任务命中数和选用的参考公式见 `per_task.csv`。

被测模型输出上限为 28,800 token；不显式发送 temperature 或 reasoning effort。同一 MP 服务的默认 effort 已检查为 medium。评委为 GPT-4.1，temperature=0、max_tokens=400。检测输入只含目标量和输入变量的文字描述，不提供观测数据。

MP 的 n=5 兼容接口只返回一条回答。本次通过上游允许的 custom 接口发送五次独立 n=1 请求。实际请求参数、回答、响应 ID、模型 ID、用量、结束原因和传输重试记录均保存在各压缩包中。环境与适配器检查记录见 `API_PREFLIGHT.json`、`ENVIRONMENT_VALIDATION.json` 和 `RUN_CONFIG.json`。

完整性检查已逐条核对全部 436 个组合：五次实际回答、固定 prompt 哈希、判分请求和参考公式、原始判分 JSON，以及从原始判分重新计算的命中数。原始 `check_run.py` 同时通过。核对细节见 `AUDIT.json` 和 `CHECK_RUN.txt`。

本次空回答数为 0，触及输出上限的回答数为 0；发生 49 次传输/API 异常重试，最终用于评分的调用均有成功响应。有效回答未按得分重采样。API 未提供可用的美元计费信息。

四个 ZIP 压缩包采用 `runs/<bench>/<task>/<model>/` 目录结构，可解压到同一个目录。包内文件的 SHA-256 与大小记录在 `MANIFEST.json`，交付文件校验值记录在 `SHA256SUMS`。解压后的 `d1_raw/` 保存回答调用，`judge_raw/` 保存判分调用，`result.json` 保存逐参考公式结果。

无数据召回反映模型的已有知识或推导能力，不能单独证明训练数据污染；未召回也不能证明任务未出现在训练数据中。这里报告的是本次两个模型的独立检测结果。
