# 消融实验：单变量探测与知识探测

本实验检验 Heuristic Learning（HL）流程中两类探测信息对最终白盒规则的影响：

- `U`：是否运行单变量统计探测。
- `K`：是否运行知识探测。

当前代码同时提供普通随机训练抽样、1:1 平衡训练抽样和多批次重复运行三个入口。所有入口都生成并迭代真实的 HL 规则，不是只做静态分析。

所有命令都应在仓库根目录执行。

## 当前文件与入口

| 文件 | 作用 | 是否直接运行 |
| --- | --- | --- |
| `experiment/ablation/run_ablation.py` | 使用训练池中的普通随机样本运行消融矩阵 | 是 |
| `experiment/ablation/run_ablation_balance.py` | 使用 1:1 平衡训练样本运行消融矩阵 | 是 |
| `experiment/ablation/run_batches.py` | 以多个随机种子重复执行 `run_ablation.py` 的完整矩阵 | 是 |
| `experiment/ablation/README.md` | 本实验说明 | 否 |

## 运行前准备

### Python 与依赖

项目要求 Python 3.11 及以上，仓库的 `.python-version` 当前指定 3.11。三个入口均使用项目运行时依赖：

```bash
uv sync --no-dev
```

如需同时运行其他基线实验，也可统一执行 `uv sync`。项目不维护锁文件，因此不使用 `--locked` 或 `--frozen`。

### 数据文件

| 数据集 | CSV 路径 | 标签列 |
| --- | --- | --- |
| YHD | `data/YHD_bicarbonate.csv` | `hospital_expire_flag` |
| UKB | `data/UKB.csv` | `label` |

标签必须能转换为 `0/1`。每个数据集会抽取独立的平衡验证集和测试集，因此每个类别至少要满足保留集需求，并在扣除保留集后留下训练样本。

### LLM 配置

```bash
export DEEPSEEK_API_KEY="你的密钥"
```

当前两个消融执行器固定使用：

| 配置 | 值 |
| --- | --- |
| API 地址 | `https://api.deepseek.com/v1` |
| 密钥变量 | `DEEPSEEK_API_KEY` |
| 模型 | `deepseek-v4-pro` |
| 温度 | `0.0` |

即使 `U=0` 且 `K=0`，初始规则生成和规则迭代仍然需要 LLM，因此四种消融配置都必须提供有效密钥。

## 实验设计

### 四种消融配置

| 配置 | 单变量探测 | 知识探测 | 初始规则生成 | 规则迭代 |
| --- | --- | --- | --- | --- |
| `U1_K1` | 开启 | 开启 | 开启 | 开启 |
| `U1_K0` | 开启 | 关闭 | 开启 | 开启 |
| `U0_K1` | 关闭 | 开启 | 开启 | 开启 |
| `U0_K0` | 关闭 | 关闭 | 开启 | 开启 |

该矩阵只改变探测阶段是否为规则生成提供对应信息，其他 HL 主流程保持一致。

### 固定数据划分

每个数据集按本次 `--seed` 构造：

- 验证集：1,000 条，正负各 500 条。
- 测试集：1,000 条，正负各 500 条。
- 训练池：排除验证集和测试集后的数据。
- 训练规模：`3000`、`1000`、`100`、`10`。
- 训练、验证、测试源行互不重叠；平衡入口仅在训练池内部因类别不足而允许重复抽样。

验证集用于规则演化与版本选择，测试集只用于最终 `final_heuristic_model.py` 的保留集评估。报告 Accuracy、F1、Sensitivity 和 Specificity。

### 两种训练抽样方式

`run_ablation.py`：

- 从整个训练池中按规模随机抽样。
- 不强制训练类别比例。
- 不使用有放回抽样；训练池总行数小于目标规模时任务报错。

`run_ablation_balance.py`：

- 每个训练规模固定正负各一半。
- 某一类别可用行数不足时，仅对该类别有放回抽样。
- 输出记录目标类别数、可用类别数、是否替换和唯一源行数。

每个入口的完整矩阵均为：

```text
2 个数据集 × 4 种 U/K 配置 × 4 个训练规模 = 32 个任务
```

## 普通随机抽样入口

### 完整运行

```bash
uv run python experiment/ablation/run_ablation.py
```

默认使用种子 42，输出到 `experiment/ablation/outputs`。

### 最小可运行检查

```bash
uv run python experiment/ablation/run_ablation.py \
  --dataset YHD \
  --ablation U1_K1 \
  --train-size 10 \
  --seed 42 \
  --output-root experiment/ablation/outputs_smoke
```

### 单独运行某个数据集或配置

```bash
uv run python experiment/ablation/run_ablation.py \
  --dataset UKB \
  --ablation U0_K0 \
  --seed 36 \
  --output-root experiment/ablation/outputs_ukb_u0k0_seed36
```

### 并发任务

```bash
uv run python experiment/ablation/run_ablation.py \
  --workers 2 \
  --output-root experiment/ablation/outputs_seed42
```

`--workers` 使用多进程并行执行独立 HL 任务。提高并发前应确认 API 限流、费用和本机内存。

### 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--workers` | `1` | HL 任务并发进程数 |
| `--seed` | `42` | 数据划分、抽样和 HL 随机种子 |
| `--output-root` | `experiment/ablation/outputs` | 本次运行的产物根目录 |
| `--dataset` | 全部 | 只运行 `YHD` 或 `UKB` |
| `--ablation` | 全部 | 只运行四种配置之一 |
| `--train-size` | 全部 | 只运行 `3000`、`1000`、`100` 或 `10` |

该入口没有续跑参数。再次使用同一个输出根目录会在任务层级继续创建新的时间戳目录，但根目录的 `index.json` 和 `ablation.csv` 会被本次所选任务重写。为了保持一次运行的完整记录，应为每次运行指定新的 `--output-root`。

## 1:1 平衡抽样入口

### 完整运行

```bash
uv run python experiment/ablation/run_ablation_balance.py
```

默认输出到 `experiment/ablation/output_balance`。

### 最小可运行检查

```bash
uv run python experiment/ablation/run_ablation_balance.py \
  --dataset YHD \
  --ablation U1_K1 \
  --train-size 10 \
  --seed 42 \
  --output-root experiment/ablation/output_balance_smoke
```

### 筛选配置并并发运行

```bash
uv run python experiment/ablation/run_ablation_balance.py \
  --dataset UKB \
  --ablation U1_K0 \
  --workers 2 \
  --output-root experiment/ablation/output_balance_ukb_u1k0
```

参数集合与普通随机入口相同，区别只有默认 `--output-root` 和训练样本构造策略。同样建议每次使用新的输出根目录。

## 多批次重复入口

`run_batches.py` 只调用普通随机抽样入口 `run_ablation.py`，每批执行完整 32 项矩阵。各批次在外层顺序执行，每批使用由 `secrets` 生成的新随机种子；`--workers` 控制单批内部的 HL 并发。

### 默认运行 10 批

```bash
uv run python experiment/ablation/run_batches.py
```

默认共执行 `10 × 32 = 320` 个 HL 任务。

### 先运行少量批次

```bash
uv run python experiment/ablation/run_batches.py \
  --runs 2 \
  --workers 2 \
  --base-dir experiment/ablation/outputs_batches_smoke
```

### 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--runs` | `10` | 完整消融矩阵的重复次数，必须为正整数 |
| `--workers` | `1` | 每一批内部的任务并发数 |
| `--base-dir` | `experiment/ablation/outputs_batches` | 所有批次目录和批次索引的根目录 |

批次目录格式：

```text
<base-dir>/outputs_<两位批次号>_<时间戳>/
```

脚本要求新批次目录不存在，避免覆盖。所有批次结束后写出：

```text
<base-dir>/batches_<时间戳>.json
```

其中记录每批的序号、随机种子和输出根目录。

## 输出目录与产物

单个消融任务的目录结构：

```text
<output-root>/<数据集>/<U_K 配置>/train<规模>/<时间戳>/
```

每次运行的根目录还会生成：

| 文件 | 内容 |
| --- | --- |
| `ablation.csv` | 本次选中任务的汇总结果 |
| `index.json` | 本次选中任务的结构化索引 |

`ablation.csv` 包含数据集、U、K、训练集数据量、四项指标、最终版本、`status`、`out_dir` 和 `error`。单任务目录主要包含：

| 文件 | 内容 |
| --- | --- |
| `probe_univariate_results.csv` | U=1 时的单变量探测结果 |
| `probe_knowledge.md` | K=1 时的知识探测结果 |
| `heuristic_system.py` | 从初始版本到后续迭代的全部规则 |
| `evolution_results.txt` | 各版本验证指标 |
| `iteration_log.json` | 规则演化日志 |
| `final_heuristic_model.py` | 最终白盒规则，可调用 `predict(features)` |
| `final_comparison.txt` | 最终规则版本比较 |
| `heldout_test_summary.json` | 测试指标、种子、划分和训练抽样元数据 |
| `heldout_test_summary.txt` | 人类可读测试摘要 |
| `error.txt` | 任务异常时的完整堆栈 |

未启用的探测阶段不会产生对应的有效探测信息。应根据 U/K 配置判断哪些文件是本任务的必要产物，而最终模型、测试摘要和汇总状态是所有成功任务的共同检查点。

## 结果解读与运行注意事项

- 比较 U/K 影响时，应固定数据集、训练规模、种子和抽样入口。
- 普通随机抽样反映训练池自身的类别构成；平衡抽样隔离了类别比例变化的影响，两者结果不应混在同一统计组。
- 一个任务失败不会终止整个矩阵；查看 `ablation.csv` 的 `status/error` 和任务目录的 `error.txt`。
- 筛选运行也会重写根目录汇总文件，因此不要用已有完整运行的输出根目录做局部补跑。
- 32 项完整矩阵和多批次入口会产生大量 LLM 调用；先用单数据集、单配置、10 条训练样本验证环境和密钥。
