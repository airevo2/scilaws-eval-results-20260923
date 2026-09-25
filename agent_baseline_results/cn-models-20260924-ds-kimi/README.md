# DeepSeek V4.1 Flash / Kimi K3：Bench 运行与数值评测结果

两个模型均已完成 Real118题、Parallel118题，共472个任务。Real的236个数值结果已齐。**有效性S_V和Parallel结构分S_S尚未进行最终判分，因此本目录不是完整的三指标榜单。**

| 模型 | Real S_N | Real S_V | Parallel S_S |
|---|---:|---|---|
| deepseek-v4.1-flash | 0.405915 | 待判分 | 待判分 |
| kimi-k3 | 0.473568 | 待判分 | 待判分 |

**配置说明：本批Bench没有显式传reasoning_effort，采用接口默认值；不能标成统一medium。** 保留的是当时实际完成的结果。任务、提示词、最多30轮及原始最终提交处理、65,536-token输出上限均有记录。调用并发先为每模型2，后续进程按用户要求提高至8。

S_N按完整118题分母汇总，代码执行失败及契约不合格保留原始零分。没有按得分选择重跑；仅恢复了明确的接口失败。DeepSeek模型ID为deepseek-v4.1-flash。

per_task.tsv包含逐题指标与原始来源路径；所有tar.gz解压到同一目录即可查看原始轨迹、提交代码和日志。infrastructure-history.tar.gz保留初始接口失败及补跑记录。SUMMARY.json含TypeI/TypeII拆分。AUDIT.json说明本次核验范围，MANIFEST.json与SHA256SUMS提供校验信息。

共享的原始总配置和台账可能包含GLM条目，仅用于追溯；本目录的指标表、per_task.tsv及轨迹/提交分片只统计DeepSeek V4.1 Flash和Kimi K3。
