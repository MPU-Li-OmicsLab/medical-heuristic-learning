# 持续学习实验：SIRS 到 SOFA 的特征漂移

本实验在 MIMIC 医学表格二分类数据上构造两个完全分离的时间阶段，研究模型在训练样本极少且特征定义发生变化时，继承第一阶段状态是否优于只用第二阶段数据从头训练。当前实现同时覆盖 6 个机器学习/深度学习基线和 Heuristic Learning（HL）白盒规则。

核心漂移是：第一阶段提供 `SIRS`，第二阶段删除 `SIRS` 并加入 `SOFA`。两者是不同临床指标，代码不会把 `SIRS` 重命名或复制为 `SOFA`。

所有命令都应在仓库根目录执行。

## 当前文件与入口

| 文件 | 作用 | 是否直接运行 |
| --- | --- | --- |
| `experiment/continuous_learning/run_continuous_learning_baselines_v2.py` | 运行 6 个基线的三分支实验 | 是 |
| `experiment/continuous_learning/run_continuous_learning_hl_v2.py` | 运行 HL 的三个阶段端点 | 是 |
| `experiment/continuous_learning/verify_continuous_baselines_v2.py` | 校验结果表、冻结划分、预测、指标和迁移产物 | 是 |
| `experiment/continuous_learning/continuous_learning_experiment_common.py` | 统一定义数据集、特征、种子、划分和结果结构 | 否 |
| `experiment/continuous_learning/continuous_baseline_v2.py` | 实现各基线的训练、直接训练和状态迁移 | 否 |
| `experiment/continuous_learning/README.md` | 本实验说明 | 否 |

只需直接调用前三个入口；另外两个文件是共享实现，不是独立命令。

## 运行前准备

### Python 与依赖

项目要求 Python 3.11 及以上，仓库的 `.python-version` 当前指定 3.11。当前持续学习入口会加载 PyTorch Lightning、DeepTab、XGBoost、LightGBM 和 InterpretML 等实现，因此无论运行基线还是 HL，都应安装完整开发依赖：

```bash
uv sync
```

项目不维护锁文件，所以不使用 `--locked` 或 `--frozen`。FT-Transformer 和 ResNet 可使用 GPU；如需限制可见设备，可在命令前设置：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
  experiment/continuous_learning/run_continuous_learning_baselines_v2.py
```

是否实际使用 GPU 由当前 PyTorch/Lightning 环境和设备可用性决定。

### 数据文件与标签

| 项目 | 值 |
| --- | --- |
| 数据集名 | `MIMIC` |
| CSV | `data/merged_by_subject_id_complete_rows_without_unit_cols_renamed.csv` |
| 标签列 | `death_within_hosp_28days` |
| 标签取值 | 可转换为整数 `0/1` |
| 默认种子 | `36 40 42` |

数据中不能预先存在保留字段 `__continuous_row_id__`。脚本会按原始行顺序创建该字段，用于验证各阶段源行不重叠。

### HL 密钥

```bash
export DEEPSEEK_API_KEY="你的密钥"
```

当前 HL 入口固定使用 `https://api.deepseek.com/v1`、模型 `deepseek-v4-pro`、温度 `0.0` 和密钥变量 `DEEPSEEK_API_KEY`。

## 两阶段数据协议

### 特征变化

两个阶段共享 30 个特征：

```text
Age
Sex (M-0, F-1)
White Blood Cell Count
Red Blood Cell Count
Platelet Count
Hemoglobin
Red Cell Distribution Width
Hematocrit
Albumin
Sodium
Potassium
Total Calcium
Chloride
Blood Glucose
Anion Gap
pH
Partial Pressure of Carbon Dioxide
Partial Pressure of Oxygen
Blood Lactate
Total Carbon Dioxide
Ionized Calcium
Prothrombin Time
Activated Partial Thromboplastin Time
International Normalized Ratio
Bilirubin
Alanine Aminotransferase
Aspartate Aminotransferase
Blood Urea Nitrogen
Creatinine
Lactate Dehydrogenase
```

阶段特有特征：

| 阶段 | 第 31 个特征 | 变化 |
| --- | --- | --- |
| Stage 1 | `SIRS` | 初始特征集 |
| Stage 2 | `SOFA` | 删除 `SIRS`，加入 `SOFA`，无重命名映射 |

### 样本数量与隔离

每个种子都构造六个两两不重叠的平衡子集：

| 阶段 | 训练集 | 验证集 | 测试集 |
| --- | ---: | ---: | ---: |
| Stage 1 | 1,000（500/500） | 500（250/250） | 800（400/400） |
| Stage 2 | 40（20/20） | 500（250/250） | 800（400/400） |

抽样不允许替换，因此原始数据每个类别至少需要 `500 + 250 + 400 + 20 + 250 + 400 = 1820` 条。Stage 1 的任何训练、验证、测试行都不会出现在 Stage 2，Stage 2 内部三部分也互不重叠。

Stage 1 与 Stage 2 的测试集不同；两个 Stage 2 分支共享同一份 Stage 2 测试集。

## 三个评估端点

| 结果中的阶段名 | 含义 | 可用训练数据/状态 |
| --- | --- | --- |
| `stage1_direct_train1000` | 第一阶段直接训练 | Stage 1 的 1,000 条训练数据 |
| `stage2_continual_from_stage1_train40` | 从 Stage 1 状态持续学习 | Stage 1 已训练状态 + Stage 2 的 40 条训练数据 |
| `stage2_direct_train40` | 第二阶段从头训练 | 仅 Stage 2 的 40 条训练数据 |

持续学习分支不会回放任何 Stage 1 训练行。它继承的是模型或规则状态，而不是把两阶段训练数据合并。Stage 2 直接分支不能读取 Stage 1 模型、预处理统计或训练行，因此它是严格的 40 样本从头训练对照。

## 基线模型与迁移方式

当前基线集合：

1. `MLP`
2. `XGBoost`
3. `LightGBM`
4. `EBM`
5. `FT-Transformer`
6. `ResNet`

所有模型都生成 Stage 1、Stage 2 持续学习和 Stage 2 直接训练三个端点。一次“模型 × 种子”作为一个整体训练；任一环节抛出异常时，该模型三个端点都会标记为错误。

### 统一特征模式

MLP、XGBoost、LightGBM、FT-Transformer 和 ResNet 使用包含全部公共特征、`SIRS` 与 `SOFA` 的联合模式：

- Stage 1 不存在 `SOFA`，填充值为 0。
- 持续学习的 Stage 2 不存在 `SIRS`，使用 Stage 1 训练数据的 `SIRS` 中位数填充。
- Stage 2 直接训练不允许读取 Stage 1 中位数，因此 `SIRS` 填 0。
- `SIRS` 和 `SOFA` 始终占据不同特征位置，绝不互相替代。

EBM 使用自身的原始表格处理；持续分支中的 Stage 2 缺失 `SIRS` 也使用 Stage 1 中位数填充，直接分支则独立从头拟合。

### 各模型的持续学习语义

| 模型 | Stage 2 持续学习方式 | Stage 2 直接对照 |
| --- | --- | --- |
| MLP | 深拷贝 Stage 1 estimator，校验参数哈希后用 `partial_fit` 继续；只适配新增 SOFA 的预处理统计，SOFA 对应休眠权重从 0 开始；最多 200 epoch，patience 20 | 使用 40 条 Stage 2 数据从头训练 |
| XGBoost | 将选定的 Stage 1 booster 通过 `xgb_model` 传入继续生长，patience 20 | 从头训练，容量为 400 estimators |
| LightGBM | 通过 `init_model` 继承 Stage 1 booster，使用安全特征名，patience 20 | 从头训练，容量为 500 estimators |
| EBM | 用 Stage 1 模型对 Stage 2 生成原始决策分数，作为 `init_score` 拟合目标残差，并在预测时继续使用该初始分数 | 不读取 Stage 1 分数，完全从头拟合 |
| FT-Transformer | 在内存中精确迁移 Stage 1 参数，重新创建 trainer/optimizer；Stage 2 学习率为原来的 0.1，SOFA 对应休眠参数从 0 开始；最多 100 epoch，patience 15 | 使用相同架构从头训练 |
| ResNet | 与 FT-Transformer 相同：内存参数迁移、新 optimizer、学习率乘 0.1、SOFA 参数从 0 开始、最多 100 epoch、patience 15 | 使用相同架构从头训练 |

DeepTab 两个模型的迁移不依赖磁盘 checkpoint；最佳验证状态在内存中选择。每个端点保存后，执行器会重新加载模型并核对预测概率，以确认持久化产物可复现。

## 基线入口运行方式

### 完整运行

```bash
uv run python \
  experiment/continuous_learning/run_continuous_learning_baselines_v2.py
```

完整基线共有：

```text
6 个模型 × 3 个种子 × 3 个端点 = 54 行结果
```

### 最小可运行检查

```bash
uv run python \
  experiment/continuous_learning/run_continuous_learning_baselines_v2.py \
  --models MLP \
  --seeds 42
```

### 筛选模型与种子

`--models` 使用逗号分隔，`--seeds` 使用空格分隔：

```bash
uv run python \
  experiment/continuous_learning/run_continuous_learning_baselines_v2.py \
  --models MLP,XGBoost,EBM \
  --seeds 36 42
```

### 续跑与错误重试

```bash
uv run python \
  experiment/continuous_learning/run_continuous_learning_baselines_v2.py \
  --resume

uv run python \
  experiment/continuous_learning/run_continuous_learning_baselines_v2.py \
  --resume --retry-errors
```

一个模型仅在三个端点都有 `ok/continued` 状态，且每个端点都存在 `predictions.csv` 和 `metrics.json` 时才视为完整并跳过。已有错误且不加 `--retry-errors` 时会保留错误；加上后会重新训练该模型的整个三端点组合。

不使用 `--resume` 时，汇总表中的 HL 行会保留，但已有基线行会从汇总表中清除，然后只写回本次选中的模型和种子；磁盘上的基线产物不会因此删除。因此在已有完整结果上做局部补跑时，应使用 `--resume`。

### 基线命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--models` | `all` | 全部 6 个模型，或逗号分隔的精确模型名 |
| `--seeds` | `36 40 42` | 一个或多个种子 |
| `--resume` | 关闭 | 跳过三个端点均完整的模型 |
| `--retry-errors` | 关闭 | 与 `--resume` 配合重新训练错误模型 |

## HL 持续学习语义

HL 同样生成三个端点：

- Stage 1：在 1,000 条 Stage 1 训练数据上执行完整单变量探测、知识探测、初始规则生成和 10 轮规则迭代。
- Stage 2 持续学习：读取同一种子的 Stage 1 最终规则、单变量探测和知识探测产物，显式传入“删除 SIRS、加入 SOFA”的漂移上下文，在 40 条 Stage 2 训练数据上执行持续规则学习和 10 轮迭代。
- Stage 2 直接训练：不读取 Stage 1 的任何状态，在 40 条 Stage 2 数据上独立执行完整 HL 流程和 10 轮迭代。

持续分支使用 Stage 1 最终规则作为蓝图，并保存上一阶段探测信息的过滤副本；它同样不回放 Stage 1 训练行。直接分支用于隔离“继承规则知识”带来的效果。

HL 的 `predictions.csv` 中 `positive_probability` 当前是硬预测标签转换成的 `0.0/1.0`，不是经过校准的概率；指标仍由固定的二分类预测计算。

## HL 入口运行方式

### 完整运行三个端点和全部种子

```bash
uv run python \
  experiment/continuous_learning/run_continuous_learning_hl_v2.py
```

默认 `--stages all`、`--seeds 36 40 42`。同一种子内按 Stage 1、Stage 2 持续学习、Stage 2 直接训练的顺序运行。

### 最小独立端点检查

Stage 2 直接分支不依赖 Stage 1：

```bash
uv run python \
  experiment/continuous_learning/run_continuous_learning_hl_v2.py \
  --stages stage2 \
  --seeds 42
```

### 只运行 Stage 1 和持续学习分支

```bash
uv run python \
  experiment/continuous_learning/run_continuous_learning_hl_v2.py \
  --stages stage1 stage1-to-stage2 \
  --seeds 42
```

### 单独运行持续学习分支

```bash
uv run python \
  experiment/continuous_learning/run_continuous_learning_hl_v2.py \
  --stages stage1-to-stage2 \
  --seeds 42
```

这种调用要求汇总表中同一种子的 Stage 1 行状态为 `ok`，并且 Stage 1 目录中至少存在最终规则、预测、指标、运行清单、单变量探测和知识探测产物。脚本会在发出任何 LLM 请求前完成依赖校验。

### 阶段参数及别名

| 目标端点 | 可用参数值 |
| --- | --- |
| 全部 | `all` |
| Stage 1 | `stage1`、`stage1-direct`、完整阶段名 |
| Stage 2 持续学习 | `stage1-to-stage2`、`stage1->2`、`continual`、完整阶段名 |
| Stage 2 直接训练 | `stage2`、`stage2-direct`、完整阶段名 |

多个值可用空格传入，也可在单个参数值中用逗号分隔。

### 续跑、重试与运行标识

```bash
uv run python \
  experiment/continuous_learning/run_continuous_learning_hl_v2.py \
  --resume

uv run python \
  experiment/continuous_learning/run_continuous_learning_hl_v2.py \
  --resume --retry-errors
```

`--resume` 跳过结果和必要产物完整的端点，并保留已有错误；`--retry-errors` 只在配合 `--resume` 时重试错误端点。如果所有选中端点均已完整跳过，则无需 API 密钥。

默认 `run_id` 是带微秒的时间戳。也可显式指定便于追踪的叶目录名：

```bash
uv run python \
  experiment/continuous_learning/run_continuous_learning_hl_v2.py \
  --stages stage2 --seeds 42 \
  --run-id sofa_direct_seed42_trial1
```

脚本绝不会覆盖非空的同名目录；若目标 `run_id` 已被占用，会直接报错。新运行应使用新的标识。

### HL 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--stages` | `all` | 一个或多个阶段/别名 |
| `--seeds` | `36 40 42` | 一个或多个种子 |
| `--resume` | 关闭 | 跳过产物完整的选中端点 |
| `--retry-errors` | 关闭 | 与 `--resume` 配合重试错误端点 |
| `--run-id` | 唯一时间戳 | 本次新产物的叶目录名 |

## 推荐的完整运行顺序

```bash
# 1. 安装完整依赖
uv sync

# 2. 运行 6 个基线
uv run python \
  experiment/continuous_learning/run_continuous_learning_baselines_v2.py

# 3. 设置密钥并运行 HL
export DEEPSEEK_API_KEY="你的密钥"
uv run python \
  experiment/continuous_learning/run_continuous_learning_hl_v2.py

# 4. 验证完整结果
uv run python \
  experiment/continuous_learning/verify_continuous_baselines_v2.py
```

## 输出目录与文件

### 共享结果表

基线与 HL 共用同一张 CSV：

```text
experiment/continuous_learning/continuous_baselines_v2_results.csv
```

字段为模型、数据集、seed、阶段、ACC、F1、Sensitivity、Specificity、`status`、`error` 和 `out_dir`。直接训练端点成功状态为 `ok`，持续学习端点成功状态为 `continued`。脚本在每个模型或端点结束后更新该表，便于中断恢复。

### 冻结数据清单

每个种子写入：

```text
experiment/outputs_rerun/continuous_learning_v2/
└── seed<seed>/MIMIC/stage_data_manifest.json
```

其中记录两个阶段六个子集的源行标识、哈希、类别计数、特征漂移和交集校验。HL 会检查现有清单与本次重建结果完全一致，不允许静默改变冻结划分。

### 基线端点目录

```text
experiment/outputs_rerun/continuous_learning_v2/
└── seed<seed>/MIMIC/<模型>/<阶段名>/
```

主要文件：

| 文件 | 内容 |
| --- | --- |
| `predictions.csv` | 800 条测试预测、真实标签、硬预测和正类概率 |
| `metrics.json` | 四项测试指标 |
| `run_manifest.json` | Stage 1 或 Stage 2 直接训练协议 |
| `continuation_manifest.json` | Stage 2 持续学习的源模型、迁移和防回放证据 |
| `resolved_config.json` | 实际训练参数和预处理配置 |
| `environment.json` | 依赖与运行环境 |
| 模型文件 | 按模型类型保存的可重新加载产物 |
| `elapsed.json` | 一个模型三端点的总耗时，位于模型根目录 |
| `error.txt` | 模型组合失败时的完整堆栈，位于模型根目录 |

### HL 端点目录

```text
experiment/outputs_rerun/continuous_learning_v2/
└── seed<seed>/MIMIC/HL/<阶段名>/<run_id>/
```

所有成功端点包含：

| 文件 | 内容 |
| --- | --- |
| `probe_univariate_results.csv` | 当前阶段单变量探测 |
| `probe_knowledge.md` | 当前阶段知识表 |
| `heuristic_system.py` | 各轮规则版本 |
| `evolution_results.txt` | 版本验证结果 |
| `iteration_log.json` | 10 轮迭代记录 |
| `final_heuristic_model.py` | 最终白盒规则，可调用 `predict(features)` |
| `final_comparison.txt` | 最终版本比较 |
| `predictions.csv` | 800 条保留测试预测 |
| `metrics.json` | 测试指标 |
| `run_manifest.json` / `continuation_manifest.json` | 直接或持续端点协议 |
| `elapsed.json` | 端点耗时 |
| `error.txt` | 失败时的完整堆栈 |

Stage 2 持续学习还包含：

| 文件 | 内容 |
| --- | --- |
| `continuous_learning_context.json` | 删除/新增特征、Stage 1 目录、迭代次数和指标优先级 |
| `probe_univariate_results_prev.csv` | 过滤后的 Stage 1 单变量探测副本 |
| `probe_knowledge_prev.md` | Stage 1 知识表副本 |

## 结果验证入口

### 验证完整矩阵

```bash
uv run python \
  experiment/continuous_learning/verify_continuous_baselines_v2.py
```

默认验证完整的：

```text
(6 个基线 + HL) × 3 个种子 × 3 个端点 = 63 行
```

验证器会检查：

- 汇总 CSV 字段、组合完整性、唯一性、状态和指标范围。
- 每个端点恰有 800 条预测。
- 预测源行等于冻结测试清单，Stage 1 与 Stage 2 各子集无交集。
- 正类概率位于 `[0, 1]`，0.5 阈值生成的标签与文件一致。
- 从预测重新计算的四项指标与 `metrics.json`、汇总表一致。
- 运行清单或持续学习清单存在且阶段语义一致。
- Stage 2 直接分支没有读取 Stage 1 状态。
- 持续分支的源模型、参数迁移、模型重载和防数据回放证据一致。

### 验证局部运行

只运行部分模型、种子或端点时：

```bash
uv run python \
  experiment/continuous_learning/verify_continuous_baselines_v2.py \
  --allow-partial
```

`--allow-partial` 只放宽“必须包含完整 63 个组合”的要求；对当前存在结果的文件结构、预测、指标和清单检查仍会执行。

## 结果解读与运行注意事项

- 关键比较是同一模型、同一种子下 `stage2_continual_from_stage1_train40` 与 `stage2_direct_train40` 的差异。
- Stage 1 与 Stage 2 测试集不同，不应直接把 Stage 1 指标变化解释为同一测试人群上的退化。
- 三个种子应分别完成并汇总，避免单一 40 样本训练集造成偶然结论。
- 基线和 HL 共用结果 CSV；局部补跑优先使用 `--resume`，完成后再运行验证器。
- HL 需要大量远程调用；可以先用 Stage 2 直接端点单种子检查环境，但持续学习比较必须同时具备同种子的完整 Stage 1 依赖。
- 新 HL 任务必须使用唯一 `run_id`；产物目录是结果可追溯性的一部分，不应手工复用或覆盖。
