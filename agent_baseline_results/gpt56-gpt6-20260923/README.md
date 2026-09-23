本目录保存 2026-09-23 完成的 SciLaws-Bench agent baseline 评测结果。被测模型为 GPT-5.6-Sol 和 GPT-6-Astra；每个模型分别完成 Real 与 Parallel 两种模式，每种模式包含 118 个任务，共 472 条主轨迹。

本批次使用 SciLaws-Bench 的 agent baseline 与官方评分脚本，实验配置和结果见 `RESULTS.md`。仓库根目录的冷回忆探针采用独立的实验协议。

| 文件 | 内容 |
|---|---|
| `trajectories-gpt-5.6-sol.tar.gz` | GPT-5.6-Sol 的 236 条主轨迹，Real 与 Parallel 各 118 条 |
| `trajectories-gpt-6-astra.tar.gz` | GPT-6-Astra 的 236 条主轨迹，Real 与 Parallel 各 118 条 |
| `run-artifacts.tar.gz` | 运行结果、提交、日志、评分输出、基础设施补跑记录、评分修复记录及审计证据，共 2,659 个文件 |
| `RESULTS.md` | 原始评测报告，包含实验设置、总体结果、分任务类型结果及限制 |
| `collected_results.json` | 最终汇总与完成状态 |
| `summary.tsv`、`summary.json` | 逐任务结果及运行汇总 |
| `run_meta.json` | 原始运行配置 |
| `MANIFEST.json` | 压缩包与包内原始文件的大小、SHA-256 及归属 |
| `default_effort_probe.json` | 评测完成后，对默认 reasoning effort 的补充检查记录 |
| `SHA256SUMS` | 本目录交付文件的 SHA-256 校验清单 |

被测模型的评测请求未显式设置 reasoning effort。补充检查通过同一 MP 服务的 Responses API 发出省略 effort 的最小请求，两模型均回显 `medium`；该文件记录的是评测后的检查。评委使用 GPT-5.4-mini，reasoning effort 为 `xhigh`。

三个压缩包按原始相对目录组织，可解压到同一个目录：

```bash
sha256sum -c SHA256SUMS
mkdir -p extracted
for archive in trajectories-gpt-5.6-sol.tar.gz trajectories-gpt-6-astra.tar.gz run-artifacts.tar.gz; do
    tar -xzf "$archive" -C extracted
done
```

解压后的主轨迹位于 `real/trajectories/<model>/<type>/` 和 `parallel/trajectories/<model>/<type>/`。审计证据位于 `audit/`。原报告、JSON 与轨迹保留原始运行路径；报告中的 `/data/...` 链接指向原运行环境。

压缩包保留原始结果，包括执行失败、接口检查失败、补跑尝试及选择记录。打包时排除了评分阶段复制的 benchmark 输入目录 `stage/`、Python 缓存等临时文件；具体排除规则记录在 `MANIFEST.json` 中。
