# Contrast1：训练集规模对比实验

本实验研究训练样本量变化对医学表格二分类性能的影响。当前代码提供两条独立入口：一条运行 10 个机器学习/深度学习基线，另一条运行完整 Heuristic Learning（HL）白盒规则生成流程。两条入口使用相同的数据集、训练规模、平衡抽样原则和独立测试集，可直接比较。

所有命令都应在仓库根目录执行。

## 当前文件与入口

| 文件 | 作用 | 是否直接运行 |
| --- | --- | --- |
| `experiment/contrast1/run_contrast1_balance.py` | 运行 10 个对比基线 | 是 |
| `experiment/contrast1/run_contrast1_balance_hl.py` | 运行 HL，固定启用单变量探测和知识探测 | 是 |
| `experiment/contrast1/README.md` | 本实验说明 | 否 |

## 运行前准备

### Python 与依赖

项目要求 Python 3.11 及以上，仓库的 `.python-version` 当前指定 3.11。运行完整基线集合需要开发依赖组：

```bash
uv sync
```

该命令会安装 LightGBM、PyTorch、XGBoost、DeepTab、InterpretML/APLR 和 CORELS 等依赖。只运行 HL 入口时，`uv sync --no-dev` 即可满足当前脚本的运行时依赖。项目不维护锁文件，所以不使用 `--locked` 或 `--frozen`。

CORELS 1.1.29 从源码构建时需要 C++ 编译工具链。若在 Python 3.11、NumPy 2 环境遇到扩展编译错误，应使用开发依赖中的 Cython 3 从官方源码的 `corels/_corels.pyx` 重新生成 C++ 文件后构建；不要替换成名称相近的非官方包。

### 数据文件

| 数据集 | CSV 路径 | 标签列 |
| --- | --- | --- |
| UKB | `data/UKB.csv` | `label` |
| YHD | `data/YHD_bicarbonate.csv` | `hospital_expire_flag` |

标签必须能转换为 `0/1`。每个类别至少要能提供验证集与测试集所需的样本，并在扣除这两部分后保留可用于训练的样本。

### HL 密钥

运行 `run_contrast1_balance_hl.py` 前设置：

```bash
export DEEPSEEK_API_KEY="你的密钥"
```

HL 入口支持以下环境变量：

| 环境变量 | 默认值 |
| --- | --- |
| `CONTRAST1_HL_BASE_URL` | `https://api.deepseek.com/v1` |
| `CONTRAST1_HL_KEY_ENV` | `DEEPSEEK_API_KEY` |
| `CONTRAST1_HL_MODEL` | `deepseek-v4-pro` |
| `CONTRAST1_HL_TEMPERATURE` | `0.0` |

## 实验设计

### 数据划分

每个数据集和随机种子独立构造：

- 验证集：1,000 条，正负各 500 条。
- 测试集：1,000 条，正负各 500 条。
- 验证集、测试集和训练池互不重叠。
- 训练规模：`3000`、`1000`、`500`、`100`、`50`、`10`。
- 每个训练集固定为 1:1 类别比例；某类剩余样本不足目标数量时，对该类进行有放回抽样。
- 不同训练规模使用确定性种子构造，但它们不是必须彼此嵌套的子集。

最终只在测试集上报告 Accuracy、F1、Sensitivity 和 Specificity。基线的概率到类别转换阈值固定为 0.5。

### 基线模型

当前基线集合由 `experiment/modeling/config.py` 统一定义：

1. `LogisticRegression`
2. `DecisionTree`
3. `MLP`
4. `XGBoost`
5. `LightGBM`
6. `FT-Transformer`
7. `ResNet`
8. `EBM`
9. `APLR`
10. `CORELS`

实验通过数据抽样直接控制类别比例，不启用 `class_weight`、`sample_weight`、加权采样或自动类别平衡。数值与类别特征的预处理、模型固定参数和产物保存由 `experiment/modeling/` 共享实现负责：

- sklearn、APLR 类模型：数值缺失值中位数填充并标准化，类别缺失值众数填充并独热编码。
- EBM：保留表格形态并使用其原生特征处理。
- FT-Transformer、ResNet：使用 DeepTab 的预处理和训练器。
- CORELS：只用训练集拟合二值化规则特征，数值特征按训练分位点生成区间条件，类别特征生成等值条件，并保存规则表与预处理器。

默认三个种子的完整基线矩阵共有：

```text
10 个模型 × 2 个数据集 × 6 个训练规模 × 3 个种子 = 360 个任务
```

### HL 配置

HL 对每个数据集和训练规模执行完整流程：

- `U1`：启用单变量探测。
- `K1`：启用知识探测。
- 启用 `v0` 初始规则生成。
- 启用后续规则迭代。
- 使用独立验证集选择演化结果，再在测试集上评估最终规则。

一次 HL 命令只处理一个种子，共 `2 × 6 = 12` 个任务。

## 基线入口运行方式

### 完整运行

```bash
uv run python experiment/contrast1/run_contrast1_balance.py
```

### 最小可运行检查

```bash
uv run python experiment/contrast1/run_contrast1_balance.py \
  --models LogisticRegression \
  --datasets UKB \
  --train-sizes 10 \
  --seeds 42
```

### 筛选模型、规模、数据集和种子

`--models` 使用逗号分隔，其余多值参数使用空格分隔：

```bash
uv run python experiment/contrast1/run_contrast1_balance.py \
  --models LogisticRegression,LightGBM,EBM \
  --datasets UKB YHD \
  --train-sizes 1000 100 10 \
  --seeds 36 40 42
```

### 续跑、错误重试和强制重跑

```bash
uv run python experiment/contrast1/run_contrast1_balance.py --resume
uv run python experiment/contrast1/run_contrast1_balance.py --resume --retry-errors
uv run python experiment/contrast1/run_contrast1_balance.py --rerun-existing
```

单独使用 `--resume` 会跳过汇总表中已有的成功项和错误项；加上 `--retry-errors` 后只重跑错误项。`--rerun-existing` 强制重跑当前筛选范围内已有的任务。

### 基线命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--models` | `all` | 全部模型，或逗号分隔的精确模型名 |
| `--seeds` | `36 40 42` | 一个或多个种子 |
| `--seed` | 无 | 单种子快捷覆盖；设置后覆盖 `--seeds` |
| `--train-sizes` | `3000 1000 500 100 50 10` | 一个或多个训练规模 |
| `--datasets` | `UKB YHD` | 一个或两个数据集 |
| `--workers` | `1` | 当前入口接受该参数，但模型任务仍顺序执行 |
| `--resume` | 关闭 | 跳过已有结果 |
| `--retry-errors` | 关闭 | 与 `--resume` 配合重跑错误项 |
| `--rerun-existing` | 关闭 | 强制重跑已有项 |

## HL 入口运行方式

### 运行一个种子的完整矩阵

```bash
uv run python experiment/contrast1/run_contrast1_balance_hl.py --seed 42
```

默认产物根目录会随种子变化，例如：

```text
experiment/contrast1/outputs_balance_hl_seed42/
```

### 指定输出目录并并发运行

```bash
uv run python experiment/contrast1/run_contrast1_balance_hl.py \
  --seed 36 \
  --workers 2 \
  --output-root experiment/contrast1/outputs_balance_hl_seed36
```

`--workers` 会多进程并发运行 12 个 LLM 任务。并发前应确认 API 速率限制、费用额度和本机内存。

### 依次运行多个种子

HL 入口没有多种子、筛选或续跑参数，因此应分别调用：

```bash
uv run python experiment/contrast1/run_contrast1_balance_hl.py \
  --seed 36 --output-root experiment/contrast1/outputs_balance_hl_seed36

uv run python experiment/contrast1/run_contrast1_balance_hl.py \
  --seed 40 --output-root experiment/contrast1/outputs_balance_hl_seed40

uv run python experiment/contrast1/run_contrast1_balance_hl.py \
  --seed 42 --output-root experiment/contrast1/outputs_balance_hl_seed42
```

每次调用都会重写固定汇总文件 `experiment/contrast1/contrast1_balance_hl.csv`。如需保留每个种子的汇总，应在下一次运行前将它复制或重命名，例如：

```bash
cp experiment/contrast1/contrast1_balance_hl.csv \
  experiment/contrast1/contrast1_balance_hl_seed36.csv
```

### HL 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--seed` | `42` | 本次 12 个任务共用的随机种子 |
| `--workers` | `1` | LLM 任务并发进程数 |
| `--output-root` | `outputs_balance_hl_seed<seed>` | HL 产物根目录 |

## 输出目录与文件

### 基线结果

每个种子的汇总表：

```text
experiment/contrast1/contrast1_balance_rerun_seed<seed>.csv
```

每个模型任务的产物目录：

```text
experiment/outputs_rerun/contrast1/
└── seed<seed>/<数据集>/train<规模>/<模型>/
```

汇总表记录模型、数据集、训练集数据量、四项指标、最佳 epoch、checkpoint、`status` 和 `error`。任务目录主要包含：

| 文件 | 内容 |
| --- | --- |
| `predictions.csv` | 测试行标识、真实标签、预测标签和正类概率 |
| `metrics.json` | 测试指标 |
| `split_manifest.json` | 数据划分、源行哈希与抽样信息 |
| `run_manifest.json` | 模型、种子、数据集和运行配置 |
| `resolved_config.json` | 实际模型与训练参数 |
| `environment.json` | 关键环境与依赖版本 |
| `model.joblib` / `model.deeptab` / `model.corels` | 按模型类型保存的拟合产物 |
| `corels_rulelist.txt` 等 | CORELS 规则、谓词和预处理资料 |
| `error.txt` | 任务失败时的详细错误 |

### HL 结果

固定汇总表：

```text
experiment/contrast1/contrast1_balance_hl.csv
```

单任务产物目录：

```text
<output-root>/<数据集>/train<规模>/<时间戳>/
```

主要产物：

| 文件 | 内容 |
| --- | --- |
| `probe_univariate_results.csv` | 单变量探测结果 |
| `probe_knowledge.md` | 知识探测整理结果 |
| `heuristic_system.py` | 包含各轮版本的演化规则 |
| `evolution_results.txt` | 各规则版本的验证结果 |
| `iteration_log.json` | 每轮迭代记录 |
| `final_heuristic_model.py` | 最终白盒规则，可调用 `predict(features)` |
| `final_comparison.txt` | 最终版本比较 |
| `heldout_test_summary.json` | 测试指标、划分和抽样元数据 |
| `heldout_test_summary.txt` | 人类可读的测试摘要 |

每个 HL 任务创建新的时间戳目录，不要把新的运行写入已有目录。任务失败不会阻止其他组合继续执行，失败原因会进入汇总表的 `error` 字段。

## 结果解读与注意事项

- 比较训练规模时，应固定模型、数据集和种子；跨种子结果应再汇总均值与离散程度。
- 测试集固定为平衡集，因此这些结果用于控制条件下的规模敏感性比较，不等同于原始患病率下的部署表现。
- 小样本训练集可能包含重复源行；应结合 `split_manifest.json` 中的 replacement 与唯一源行数解释结果。
- 基线不做自动类别加权，HL 与基线都由相同的 1:1 抽样控制训练分布。
- HL 完整矩阵涉及多次 LLM 调用，运行前先用基线最小命令检查数据路径，再确认 API 密钥和费用。
