# SciLaws-Bench 记忆探针（cold-recall audit）跑测包

> **给跑模型的同学：只需要看这一页。** 这个仓库自包含：题库、冻结的 prompt、判分逻辑、一键脚本都在里面，不依赖 SciLaws-Bench 主仓库。你要做的就是把它接到你们的 API 上，按下面的清单跑完，把 `runs/` 打包发回来。分析、画图、写论文都由我们做，你不用碰。
>
> *For non-Chinese readers: `python3 run_mem.py --list`, then `--dry-run`, then `--bench scilaws --models <id> --limit 2`, then the full leg, then `python3 check_run.py` and `python3 run_mem.py --pack`; send back `mem_runs.zip`. Details below (Chinese).*

## 这在测什么（一段话）

给模型一道已发表科学定律的**目标量 + 输入变量的文字描述，不给任何数据**，让它凭记忆写出这个定律的闭式 `predict(X)`，每题采样 5 次；再由 **gpt-4.1** 逐次判断"写出来的函数形式是否和论文里的参考公式一致"。5 次里 ≥3 次一致 = 该模型"冷回忆"出了这道题。论文用它把 118 道题分成 Canon / Mid / Moat 三档，所以**所有模型必须在完全相同的 prompt、采样和判分设定下跑**，这些都已冻结，不要改。

## 0. 准备

```bash
pip install -r requirements.txt          # 只需要 openai + PyYAML，Python 3.10+
```

模型调用走 `client.py` 里唯一的一个函数 `complete()`，三种接法，选一种：

| 你们的 API | 怎么接 |
|---|---|
| **A. OpenAI 官方，或任何 OpenAI 兼容网关** | `export OPENAI_API_KEY=...`；如果是网关再 `export OPENAI_BASE_URL=https://.../v1`。不改代码。 |
| **B. OpenRouter**（非 OpenAI 厂商模型） | `export OPENROUTER_API_KEY=...`，并在 `models.json` 里给该模型写上 `provider`（第一方托管），见文件里的示例。 |
| **C. 你们自己的 llm_api** | 把 `custom_client.py` 里的一个函数填上（把 `prompt` 原样发出去，返回 n 条字符串），然后 `export MEM_FORCE_BACKEND=custom`。 |

**硬性要求：判分模型是 `gpt-4.1`，冻结，你们的 API 必须能调到它**（走 A 就是 `OPENAI_API_KEY`；走 C 就是你的 custom_client 也要能调 gpt-4.1）。判不了 gpt-4.1 的结果没法和论文里已有的 9 个模型放在一起，请先确认。

## 1. 要跑哪些实验

| 梯队 | 实验 | 模型 | 命令 | 规模 / 时间 / 费用（每个模型） |
|---|---|---|---|---|
| **第一梯队（必跑）** | ① SciLaws-Real 冷回忆 | **你们想加的所有最新模型**，GPT-6 或任何厂商都可以，模型选择完全由你们定 | `--bench scilaws` | 118 题 × 5 次 = 590 次调用 + 约 1,400 次 gpt-4.1 判分（判分约 $5）；推理模型 12 并发约 2–4 小时 |
| **第一梯队（必跑）** | ② AI-Feynman 冷回忆 | **所有最新的 OpenAI 模型**（只需 OpenAI 系；非 OpenAI 模型不用跑这个） | `--bench feynman` | 100 题 × 5 次 = 500 次调用 + 约 500 次判分（约 $2） |
| **第二梯队（有空就跑，非常希望能跑）** | ③ SciLaws-Real 冷回忆，用本包冻结的 prompt 重跑 | `gpt-4o-mini`、`gpt-5-mini`、`gpt-5.5`（这三个只跑 scilaws，不用跑 feynman） | `--bench scilaws --models gpt-4o-mini,gpt-5-mini,gpt-5.5` | 同 ①；gpt-5.5 最贵，约 $60 |

为什么有第二梯队：论文里这三个模型是 6 月在旧版任务描述上跑的，其余模型都在本包这版冻结 prompt 上跑的；补跑后整个 panel 就是同一版。

## 2. 每个模型的步骤

```bash
# 1) 登记模型（OpenAI 名字如 gpt-6 可以跳过这步；其他厂商在 models.json 加一行，照文件里的示例）
python3 run_mem.py --list                                   # 看模型表，确认 backend/kind 对

# 2) 零花费自检：渲染全部 prompt 并和冻结哈希比对，必须显示 mismatches = 0
python3 run_mem.py --dry-run --bench scilaws
python3 run_mem.py --dry-run --bench feynman

# 3) 冒烟（2 题，~$0.1）：看到每行 "D1 x/5" 就说明链路通了
python3 run_mem.py --bench scilaws --models gpt-6 --limit 2

# 4) 正式跑（可以多个模型一起：--models a,b,c）
python3 run_mem.py --bench scilaws --models gpt-6
python3 run_mem.py --bench feynman --models gpt-6           # 仅 OpenAI 模型

# 5) 收尾：自检必须显示 "all legs complete, prompts verified"，然后打包
python3 check_run.py
python3 run_mem.py --pack                                   # 生成 mem_runs.zip，发这个文件回来
```

要点：
- **中断了不用重来**：每题跑完立即落盘，重跑同一命令会覆盖已完成的题；只补缺题的话看 `check_run.py` 打印的 missing 列表。
- **不要改任何参数**：温度 0.8、每题 5 次、判分 gpt-4.1 温度 0、≥3/5 阈值、输出上限 28,800 token、推理档位 medium，全是冻结设定。`kind: reasoning` 的模型不发温度（API 会拒绝），这是预期行为。
- **OpenRouter 余额要留 1.3 倍以上**：它按 `max_tokens` 预授权，余额接近见底时整批 402。
- `runs/` 里的 `*_raw/` 是原始记录（请求、回复、用量、计费、provider），**不要删**，判分可以据此复核或重算。

## 3. 发回来的东西

`mem_runs.zip`（每个模型约 10 MB），解压就是：

```
runs/<scilaws|feynman>/<任务>/<模型>/
    result.json       每条参考公式的命中数 d1_hits(0–5)、判分模型、prompt 哈希
    d1_samples.txt    5 次回忆的原文
    d1_raw/           每次调用的完整请求 + 回复 + 用量 (+ 计费、provider)
    judge_raw/        每次判分调用和它的结论
```

不需要做任何分析。如果某个模型的 `check_run.py` 是 PARTIAL，也一起发，并说明原因。

## 4. 目录

```
run_mem.py          一键入口（你只用这个）        check_run.py     跑完自检
client.py           API 调用层（三种后端）        custom_client.py 自有 API 接口模板（接法 C 才用）
models.json         模型登记表                    probe.py         探针 + 判分（冻结，不要改）
prompts_frozen.json 每题 prompt 的哈希（校验用）   PROTOCOL.md      冻结协议全文
bench/scilaws118/   118 题（metadata + 参考公式）  bench/feynman100/ AI-Feynman 100 题
tasks/              题目清单                      data/, analysis/  已有结果 + 我们的分析脚本（你不用管）
```
