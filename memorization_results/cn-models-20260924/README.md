本批次完成三个可用模型的公式冷回忆 D1 检测：GLM-5.3、Kimi K3、DeepSeek V4.1 Flash。共 654 个任务—模型组合、3270 条实际 API 回答。Step-5-Preview 因当前 MP 账号返回 HTTP 403 未运行，结果中不填零分。

| 模型 | Benchmark | 请求的 effort | 可召回任务数 | 可召回比例 | 空回答 | 达到输出上限 |
|---|---|---|---:|---:|---:|---:|
| glm-5.3 | scilaws | max | 24/118 | 20.34% | 261 | 263 |
| kimi-k3 | scilaws | medium | 34/118 | 28.81% | 0 | 0 |
| deepseek-v4.1-flash | scilaws | medium | 31/118 | 26.27% | 98 | 98 |
| glm-5.3 | feynman | max | 61/100 | 61.00% | 94 | 94 |
| kimi-k3 | feynman | medium | 62/100 | 62.00% | 0 | 0 |
| deepseek-v4.1-flash | feynman | medium | 70/100 | 70.00% | 21 | 21 |

每个模型分别覆盖 SciLaws 118 题与 AI-Feynman 100 题，每题五次独立 n=1 请求。目标和变量描述沿用冻结 prompt；不提供数据。GPT-4.1 按原始 prompt 判定函数结构是否等价；在指定 best-baseline 公式上命中至少三次，记为可召回。逐任务结果见 per_task.csv。

GLM-5.3 明确拒绝 medium，只支持 low/high/max。本批次使用其原生默认档 max，并明确作为推理档位变体保存；Kimi K3 和 DeepSeek V4.1 Flash 接受请求的 medium。供应商未回显实际内部推理档位，不将这些标签视作等量计算，也不将本批次当作统一 medium 的严格复现或直接并入原论文的统一档位面板。

三个模型的输出上限均为 28,800 token，temperature 不显式发送。空回答和达到输出上限的响应按原协议保留，不按得分或是否生成可见回答重采样。实际请求、回答、reasoning_content、响应 ID、用量和传输重试信息保存在六个 ZIP 中。ZIP 解压到同一位置后形成 runs/<bench>/<task>/<model>/。

全量核对覆盖五个实际响应、冻结 prompt 哈希、评委参数、原始判分与命中数重算，以及压缩包内每个文件的 SHA-256。共有 1181 次传输异常重试；正式样本不包含耗尽重试的调用。耗尽重试后只恢复失败的单个样本，保留其余成功响应及已有有效判分。后续恢复使用流式传输，600秒读超时及全部生成参数不变；实际请求包含stream/include_usage，配置见TRANSPORT_RECOVERY_PROFILES.json；原始失败、恢复调用和首次有效选择记录位于ZIP内的recovery/目录，RECOVERY_SUMMARY.json记录恢复范围。审计见 AUDIT.json，原始检查程序输出见 CHECK_RUN.txt，实际执行的适配器快照见 code/。

本批次有 3 条服务端用量回报超过请求上限的响应，已逐条核对原始响应ID和文件哈希并保留记录。已审阅的情形为：请求28,800 token，回报28,801 token，全部为reasoning且没有可见回答；该空回答按未命中保留，没有重采样或修改用量。这些记录不应被称为实际用量严格不超过28,800 token。逐条证据见原始目录中的provider_usage_review.json及AUDIT.json。

该检测衡量本配置和预算下无数据恢复已知公式的能力，不能单独证明训练数据污染，也不能以未召回证明无污染。
