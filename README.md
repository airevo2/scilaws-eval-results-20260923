本仓库保存 GPT-5.6-Sol 与 GPT-6-Astra 的 SciLaws-Bench agent baseline 评测结果，以及公式冷回忆（memorization probe）检测程序和运行结果。

| 评测 | 当前状态 | 范围 | 结果位置 |
|---|---|---|---|
| Agent baseline | 已完成 | 两模型 × Real/Parallel × 118 任务，共 472 条主轨迹 | `agent_baseline_results/gpt56-gpt6-20260923/` |
| 公式冷回忆 D1 | 正在运行 | 两模型 ×（SciLaws 118 题 + AI-Feynman 100 题），每题 5 次采样 | 完成后保存至 `memorization_results/gpt56-gpt6-20260923/` |

Agent baseline 的原始报告、评分、补跑记录和文件校验清单均位于对应结果目录。两模型各包含 236 条主轨迹，轨迹压缩包保留原始 JSON 内容。

公式冷回忆检测只向模型提供目标量和输入变量的文字描述，不提供观测数据。模型独立生成 5 个 `predict(X)`；GPT-4.1 对照参考公式判定函数结构是否一致。按照上游协议，在任务指定的 best-baseline 公式上命中至少 3 次即记为可召回。检测覆盖和固定参数见 `PROTOCOL.md`；运行方法见 `docs/MEMORIZATION_RUN.md`。

无数据召回反映模型已有知识或推导能力，不能单独证明训练数据污染，也不能用未召回证明某个任务从未出现在训练数据中。

来源：`tREeFrOGcoder/scilaw_mem_pack_20260922`，固定提交 `f036d101a08b4554dc2f99e7e7da7c5d4b2be6d9`。上游 README 保留在 `docs/MEMORIZATION_KIT_README.md`。本次通过上游允许的 `custom_client.py` 接口适配 MP 服务；提示词、判分代码和固定协议保留上游版本。
