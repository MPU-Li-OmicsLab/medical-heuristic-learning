# Contrast0：LLM 后端对比实验

本实验在完全相同的数据划分、启发式学习流程和评估方式下，对比 7 个 LLM 配置生成医学表格二分类白盒规则的效果。当前唯一入口是 `run_contrast0.py`；它负责构造数据划分、调用完整 Heuristic Learning（HL）流程、在独立测试集上评估，并按随机种子持续写出汇总表。

所有命令都应在仓库根目录执行。

## 当前文件与入口

| 文件 | 作用 | 是否直接运行 |
| --- | --- | --- |
| `experiment/contrast0/run_contrast0.py` | 执行全部或筛选后的 LLM 后端对比任务 | 是 |
| `experiment/contrast0/README.md` | 本实验说明 | 否 |

## 运行前准备

### Python 与依赖

项目要求 Python 3.11 及以上，仓库的 `.python-version` 当前指定 3.11。Contrast0 只使用项目运行时依赖：

```bash
uv sync --no-dev
```

如果还要运行需要传统机器学习、深度学习或 CORELS 的实验，也可以统一执行 `uv sync` 安装开发依赖组。项目不维护锁文件，因此命令不使用 `--locked` 或 `--frozen`。

### 数据文件

| 数据集 | CSV 路径 | 标签列 |
| --- | --- | --- |
| UKB | `data/UKB.csv` | `label` |
| YHD | `data/YHD_bicarbonate.csv` | `hospital_expire_flag` |

标签必须能转换为整数 `0/1`。每个数据集会先抽取 1,000 条平衡验证集和 1,000 条平衡测试集，两者互不重叠；剩余训练池还必须同时包含正、负样本。

### API 密钥与服务地址

只需配置本次选中模型对应的密钥：

| 模型 | 默认服务 | 默认密钥环境变量 |
| --- | --- | --- |
| `deepseek-v4-pro-high` | DeepSeek | `DEEPSEEK_API_KEY` |
| `deepseek-v4-pro-max` | DeepSeek | `DEEPSEEK_API_KEY` |
| `deepseek-v4-flash-high` | DeepSeek | `DEEPSEEK_API_KEY` |
| `deepseek-v4-flash-max` | DeepSeek | `DEEPSEEK_API_KEY` |
| `qwen/qwen3.7-max` | OpenRouter | `OPENROUTER_API_KEY` |
| `gemini-3.1-pro-preview` | vveai | `VVEAI_GEMINI_API_KEY` |
| `gpt-5.5` | vveai | `VVEAI_GPT55_API_KEY` |

```bash
export DEEPSEEK_API_KEY="你的密钥"
export OPENROUTER_API_KEY="你的密钥"
export VVEAI_GEMINI_API_KEY="你的密钥"
export VVEAI_GPT55_API_KEY="你的密钥"
```

支持以下覆盖变量：

| 环境变量 | 默认值或作用 |
| --- | --- |
| `CONTRAST0_DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` |
| `CONTRAST0_DEEPSEEK_KEY_ENV` | `DEEPSEEK_API_KEY` |
| `CONTRAST0_ROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| `CONTRAST0_ROUTER_KEY_ENV` | `OPENROUTER_API_KEY` |
| `CONTRAST0_VVEAI_BASE_URL` | `https://api.vveai.com`，脚本会规范为 `/v1` 地址 |
| `CONTRAST0_VVEAI_GEMINI_KEY_ENV` | `VVEAI_GEMINI_API_KEY` |
| `CONTRAST0_VVEAI_GPT55_KEY_ENV` | `VVEAI_GPT55_API_KEY` |

## 实验设计

每个“模型 × 数据集 × 随机种子”组合独立运行一次完整 HL 流程：

1. 按种子抽取 500 个正例和 500 个负例作为测试集。
2. 再抽取 500 个正例和 500 个负例作为验证集，且不与测试集重叠。
3. 从剩余训练池抽取 1,000 条 1:1 平衡训练数据；某一类别不足 500 条时允许有放回抽样。
4. 使用相同的 HL 配置完成单变量探测、知识生成、初始规则和迭代优化。
5. 加载 `final_heuristic_model.py`，仅在最后对保留测试集评估。
6. 报告 Accuracy、F1、Sensitivity 和 Specificity。

DeepSeek 的四个配置分别组合 `pro/flash` 模型与 `high/max` thinking strength；所有配置温度均为 0。默认种子为 `36 40 42`，完整运行共有：

```text
7 个 LLM 配置 × 2 个数据集 × 3 个种子 = 42 个任务
```

## 入口运行方式

### 完整运行

```bash
uv run python experiment/contrast0/run_contrast0.py
```

LLM 调用耗时和费用都可能较高，建议先执行最小任务确认数据与密钥。

### 最小可运行检查

```bash
uv run python experiment/contrast0/run_contrast0.py \
  --models deepseek-v4-pro-high \
  --datasets UKB \
  --seeds 42
```

### 筛选多个模型、数据集和种子

`--models` 使用逗号分隔，`--datasets` 和 `--seeds` 使用空格分隔：

```bash
uv run python experiment/contrast0/run_contrast0.py \
  --models deepseek-v4-pro-high,qwen/qwen3.7-max \
  --datasets UKB YHD \
  --seeds 36 42
```

### 并行运行

```bash
uv run python experiment/contrast0/run_contrast0.py --workers 2
```

`--workers` 使用多进程并发执行独立 LLM 任务。提高并发前应确认服务商速率限制、账户额度和本机内存。

### 续跑、重试和强制重跑

```bash
uv run python experiment/contrast0/run_contrast0.py --resume
uv run python experiment/contrast0/run_contrast0.py --resume --retry-errors
uv run python experiment/contrast0/run_contrast0.py --rerun-existing
```

单独使用 `--resume` 会跳过汇总表中所有已有任务，包括错误任务；与 `--retry-errors` 同时使用才会重新执行错误项。`--rerun-existing` 会强制重跑当前筛选范围内的已有项。

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--models` | `all` | 全部模型，或逗号分隔的精确模型名 |
| `--seeds` | `36 40 42` | 一个或多个随机种子 |
| `--seed` | 无 | 单种子快捷覆盖；设置后覆盖 `--seeds` |
| `--datasets` | `UKB YHD` | 一个或两个数据集 |
| `--workers` | `1` | LLM 任务并发进程数 |
| `--output-root` | `experiment/contrast0/outputs` | 单次 HL 产物根目录 |
| `--resume` | 关闭 | 按每个种子的汇总 CSV 跳过已有任务 |
| `--retry-errors` | 关闭 | 与 `--resume` 配合，重新执行错误项 |
| `--rerun-existing` | 关闭 | 强制重新执行当前筛选出的已有项 |

## 输出目录与文件

每个种子维护一个汇总 CSV：

```text
experiment/contrast0/contrast0_rerun_seed<seed>.csv
```

脚本在每个任务结束后原子更新 CSV，字段包括大模型、数据集、四项指标、`status` 和 `error`，因此中途终止时已完成结果仍可用于续跑。

每个任务的产物目录为：

```text
<output-root>/<数据集>/<模型名>/<时间戳>/
```

模型名中的 `/` 会替换成 `_`。主要文件：

| 文件 | 内容 |
| --- | --- |
| `probe_univariate_results.csv` | 单变量探测结果 |
| `probe_knowledge.md` | 为规则生成整理的领域知识 |
| `heuristic_system.py` | 从 `v0` 开始逐轮演化的规则系统 |
| `evolution_results.txt` | 各版本验证指标 |
| `iteration_log.json` | 逐轮运行记录 |
| `final_heuristic_model.py` | 最终可直接调用 `predict(features)` 的白盒规则 |
| `final_comparison.txt` | 最终版本比较 |
| `heldout_test_summary.json` | 保留测试集指标及运行元数据 |
| `heldout_test_summary.txt` | 便于人工阅读的测试摘要 |

每次新任务都会创建带时间戳的目录，不应手工复用已有任务目录。汇总 CSV 是续跑状态入口，具体模型和诊断信息以产物目录为准。

## 结果解读与注意事项

- 四项指标都来自未参与训练和规则迭代的 1,000 条平衡测试集。
- 不同种子会改变训练、验证和测试样本，应按模型、数据集汇总三个种子的结果。
- 训练集在类别不足时可能有放回抽样；相关采样元数据会写入测试摘要。
- `status=error` 时先查看汇总表的 `error` 和产物目录，再使用 `--resume --retry-errors`。
- 并发不会改变单任务的数据种子，但可能更快触发 API 限流。
