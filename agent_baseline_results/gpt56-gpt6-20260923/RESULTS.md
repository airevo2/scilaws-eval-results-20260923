已完成 SciLaws-Bench 的 GPT-5.6-Sol 与 GPT-6 评测，共 472 个运行组合。全部运行结果、官方评分输出及汇总均已完成核对。更新时间：2026-09-23T04:37:03+08:00。

每个模型分别评测 Real 与 Parallel 两种模式，每种模式包含 118 个任务，其中 Type I 为 66 个，Type II 为 52 个。每个组合保留一次完整科学结果；Type II 数值评分内部的三个固定拟合种子沿用官方实现。

| 模型 | Real S_N | Real S_V | Parallel S_S | Real 提交数 | Parallel 提交数 |
|---|---:|---:|---:|---:|---:|
| GPT-5.6-Sol | 0.401080 | 0.809363 | 0.480932 | 118/118 | 118/118 |
| GPT-6（gpt-6-astra） | 0.513294 | 0.829129 | 0.724576 | 118/118 | 118/118 |

S_N 为数值拟合得分，其中最强公开参考公式对应 0.5、完美拟合对应 1.0；S_V 为科学有效性得分；S_S 为 Parallel 模式的结构恢复得分。数值范围均为 0–1。总体分母固定为 118 个任务；未提交或执行失败的处理沿用官方规则。

| 模型 | 任务类型 | 任务数 | Real S_N | Real S_V | Parallel S_S |
|---|---|---:|---:|---:|---:|
| GPT-5.6-Sol | typeI | 66 | 0.372408 | 0.849013 | 0.481061 |
| GPT-5.6-Sol | typeII | 52 | 0.437470 | 0.759037 | 0.480769 |
| GPT-6（gpt-6-astra） | typeI | 66 | 0.526741 | 0.882070 | 0.708333 |
| GPT-6（gpt-6-astra） | typeII | 52 | 0.496226 | 0.761934 | 0.745192 |

运行采用官方 `baseline_agent/run_baseline.py` 与 `run_batch.py`：最多 30 轮交互，并保留官方最后提交处理；`max_completion_tokens=65536`，不显式设置 reasoning effort，API 超时 120 秒、SDK 重试 1 次。Real 保留官方公开测试输入范围，Parallel 使用官方模拟器接口。

API 通过 MP 调用，实际模型 ID 为 `gpt-5.6-sol` 与 `gpt-6-astra`。Key 从已有环境变量继承，未写入结果文件。两个模型使用相同评委 `gpt-5.4-mini`，reasoning effort 为 `xhigh`，Codex CLI 为 0.153.4。有效性与结构评分均由仓库原始脚本及其原始提示词执行。每个评分批次最多包含 3 个任务，每组 4 个 worker，四个独立模型/模式组合同时评分，总并发上限为 16。

全部模型调用、提交执行和评分在 RJob `scilaws-gpt56-gpt6-0923c` 中完成，资源为 24 CPU、128 GiB 内存、0 GPU。首轮主批次采用 128 个 worker；API 补跑总并发上限为 16。

首轮保留了 454 个正常结束的运行结果，另有 18 个组合因明确的 API 服务错误或超时中断。额外执行了 20 次基础设施补跑；每个组合接受首个正常结束的运行结果，不根据分数重跑。原始日志、部分轨迹、全部补跑尝试及选择记录均保留在 `infrastructure_retries/`。初次正常终止结果的文件校验值保持一致。

| 模型 | Real 数值评分状态 | Real 有效性评分状态 | Parallel 结构评分状态 |
|---|---|---|---|
| GPT-5.6-Sol | exec_error=2；ok=116 | anti_hacking_fail=5；ok=111；validity_null=2 | contract_invalid=2；ok=116 |
| GPT-6（gpt-6-astra） | exec_error=3；ok=115 | anti_hacking_fail=1；ok=114；validity_null=3 | contract_invalid=1；ok=117 |

执行失败、提交接口检查不通过和有效性反作弊项不通过属于保留的科学结果，未从汇总分母中删除。数值执行失败的具体原因及原始零分记录见文末的数值执行错误证据链接。

仓库版本为 `9239f66b921cb89c7a9d14061f782fdce49dcfb5`。唯一已跟踪源码改动是在 `baseline_agent/call_llm_api.py` 中增加两个模型 ID 及已有 reasoning 模型集合的注册，共新增 3 行。Agent、提示词、任务加载、沙箱、批处理和评分逻辑均保持原始代码。数据为 `RealSR/SciLaws-Bench`，固定 revision `f97c105a111439ce44ce3c61686b541c41d86fa4`；1115 个文件、1,108,331,145 字节的校验已通过。

初次评分有 7 份输出缺失或不符合原始输出要求，累计补评了 7 个任务条目。每条均接受首个符合原始格式与计算要求的输出；其余 465 份初始完整评分保持不变。所有原始及补跑评分保存在 `judge_repairs/`。

官方仓库未明确给出公开排行榜所用的评委模型配置，因此本次两个模型之间使用统一评委的比较成立，与公开排行榜的严格配置一致性尚无法确认。

结果文件：

- [汇总数据](/data/SciLaws-Bench/results/gpt56-gpt6-20260923/collected_results.json)
- [逐任务运行与数值结果](/data/SciLaws-Bench/results/gpt56-gpt6-20260923/summary.tsv)
- [评分输出](/data/SciLaws-Bench/results/gpt56-gpt6-20260923/judges)
- [API 中断与补跑记录](/data/SciLaws-Bench/results/gpt56-gpt6-20260923/infrastructure_retries/attempt-ledger.jsonl)
- [最终核对记录](/data/SciLaws-Bench/.aris/runs/gpt56-gpt6-20260923/audit-final.json)
- [数值执行错误证据](/data/SciLaws-Bench/.aris/runs/gpt56-gpt6-20260923/numeric-error-witnesses.json)
- [模型和评委配置](/data/SciLaws-Bench/.aris/runs/gpt56-gpt6-20260923/judge-protocol.json)
- [源码改动](/data/SciLaws-Bench/.aris/runs/gpt56-gpt6-20260923/model-aliases.patch)
