本仓库保存 GPT-5.6-Sol 与 GPT-6-Astra 的两组完整评测结果：SciLaws-Bench agent baseline，以及公式冷回忆 D1 检测。

Agent baseline 已完成 472 个运行组合，两模型分别覆盖 Real 与 Parallel 模式各 118 个任务。S_N 为数值拟合得分，S_V 为科学有效性得分，S_S 为 Parallel 模式的结构恢复得分。完整报告和原始轨迹位于 `agent_baseline_results/gpt56-gpt6-20260923/`。

| 模型 | Real S_N | Real S_V | Parallel S_S |
|---|---:|---:|---:|
| gpt-5.6-sol | 0.401080 | 0.809363 | 0.480932 |
| gpt-6-astra | 0.513294 | 0.829129 | 0.724576 |

公式冷回忆检测已完成 436 个任务—模型组合、2,180 条回答。输入只包含目标量和变量描述，不提供观测数据；每题采样五次，由 GPT-4.1 判断与参考公式的结构等价性，命中至少三次即记为可召回。

| 模型 | Benchmark | 可召回任务数 | 可召回比例 |
|---|---|---:|---:|
| gpt-5.6-sol | scilaws | 41/118 | 34.75% |
| gpt-6-astra | scilaws | 40/118 | 33.90% |
| gpt-5.6-sol | feynman | 65/100 | 65.00% |
| gpt-6-astra | feynman | 76/100 | 76.00% |

Mem 逐任务结果、四个原始记录压缩包、审计结果及校验清单位于 `memorization_results/gpt56-gpt6-20260923/`。两组评测均保留原始记录及运行限制说明。

上游检测程序来自 `tREeFrOGcoder/scilaw_mem_pack_20260922`，固定提交 `f036d101a08b4554dc2f99e7e7da7c5d4b2be6d9`。原始 README 保留在 `docs/MEMORIZATION_KIT_README.md`；固定检测协议见 `PROTOCOL.md`，本次运行方法及 MP 接口适配见 `docs/MEMORIZATION_RUN.md`。复核工具 `tools/audit_mem_results.py` 可从解压后的原始记录重新验证全部命中数，不调用模型 API。

该冷回忆检测衡量模型在无数据条件下恢复已知公式的能力，不能单独证明训练数据污染，也不能以未召回作为无污染证明。
