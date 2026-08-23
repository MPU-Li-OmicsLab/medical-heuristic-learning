# Contrast2：类别不平衡对比实验

本实验在固定训练总量下系统改变正负样本比例，用于评估医学表格二分类模型和 Heuristic Learning（HL）白盒规则对类别不平衡的敏感性。当前代码提供基线入口、HL 入口和 HL 混淆矩阵补全工具。

所有命令都应在仓库根目录执行。

## 当前文件与入口

| 文件 | 作用 | 是否直接运行 |
| --- | --- | --- |
| `experiment/contrast2/run_contrast2.py` | 运行 10 个机器学习/深度学习基线 | 是 |
| `experiment/contrast2/run_contrast2_hl.py` | 运行完整 HL 类别比例矩阵 | 是 |
| `experiment/contrast2/fill_contrast2_hl_confusion.py` | 根据 HL 最终规则和冻结测试划分补写 TP/FP/FN/TN | 是 |
| `experiment/contrast2/README.md` | 本实验说明 | 否 |

## 运行前准备

### Python 与依赖

项目要求 Python 3.11 及以上，仓库的 `.python-version` 当前指定 3.11。运行 10 个基线需要完整开发依赖：

```bash
uv sync
```

只运行 HL 入口时可使用 `uv sync --no-dev`。项目不维护锁文件，因此不使用 `--locked` 或 `--frozen`。

CORELS 1.1.29 从源码构建需要 C++ 工具链。若 Python 3.11、NumPy 2 环境下生成的扩展代码无法编译，应使用开发依赖中的 Cython 3 从官方 `corels/_corels.pyx` 重新生成 C++ 文件后构建官方源码包。

### 数据文件

| 数据集 | CSV 路径 | 标签列 |
| --- | --- | --- |
| UKB | `data/UKB.csv` | `label` |
| YHD | `data/YHD_bicarbonate.csv` | `hospital_expire_flag` |

标签必须能转换为 `0/1`。测试集与验证集均要求每类至少 500 条，扣除两者后训练池还必须保留正、负样本。

### HL 密钥

```bash
export DEEPSEEK_API_KEY="你的密钥"
```

当前 HL 入口固定使用：

| 配置 | 值 |
| --- | --- |
| API 地址 | `https://api.deepseek.com/v1` |
| 密钥变量 | `DEEPSEEK_API_KEY` |
| 模型 | `deepseek-v4-pro` |
| 温度 | `0.0` |

## 实验设计

### 数据划分与比例矩阵

每个数据集和种子先构造互不重叠的固定保留集：

- 验证集：1,000 条，正负各 500 条。
- 测试集：1,000 条，正负各 500 条。
- 训练池：扣除验证集和测试集后的所有行。

随后分别生成两种训练总量：

```text
1000, 3000
```

每种总量覆盖九个“正类:负类”比例：

```text
1:1, 1:2, 2:1, 1:5, 5:1, 1:10, 10:1, 1:50, 50:1
```

目标正类数按比例四舍五入并限制为至少 1 条，负类数为总量减去正类数。例如总量 1,000、比例 `1:50` 时约为 20 个正例和 980 个负例。某一类别的训练池数量不足时，该类别使用有放回抽样。验证集和测试集始终保持 1:1，不随训练比例改变。

### 基线模型与公平性约束

当前模型集合：

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

固定模型参数与预处理来自 `experiment/modeling/`。本实验不启用任何自动类别修正：不使用 `class_weight`、`sample_weight`、加权采样、`scale_pos_weight` 或框架自带类别平衡。这样，模型看到的不平衡仅由显式训练比例决定。

基线以正类概率 `>= 0.5` 判定为 1，在平衡测试集上报告 Accuracy、F1、Sensitivity、Specificity 以及 TP、FP、FN、TN。

默认完整基线矩阵共有：

```text
10 个模型 × 2 个数据集 × 2 个训练总量
× 9 个比例 × 3 个种子 = 1080 个任务
```

### HL 配置

HL 对每个“数据集 × 训练总量 × 比例”执行完整 `U1_K1` 流程：单变量探测、知识探测、`v0` 生成和规则迭代全部启用。一次命令只处理一个种子，共：

```text
2 个数据集 × 2 个训练总量 × 9 个比例 = 36 个任务
```

最终 `final_heuristic_model.py` 在同一冻结测试集上评估。

## 基线入口运行方式

### 完整运行

```bash
uv run python experiment/contrast2/run_contrast2.py
```

1080 个任务耗时较长，建议先进行最小检查。

### 最小可运行检查

```bash
uv run python experiment/contrast2/run_contrast2.py \
  --models LogisticRegression \
  --datasets UKB \
  --train-totals 1000 \
  --ratios 1:1 \
  --seeds 42
```

比例参数必须使用脚本定义的精确字符串；包含冒号的值可直接传给 shell。

### 运行筛选后的矩阵

```bash
uv run python experiment/contrast2/run_contrast2.py \
  --models LogisticRegression,LightGBM,EBM \
  --datasets UKB YHD \
  --train-totals 1000 3000 \
  --ratios 1:1 1:10 10:1 \
  --seeds 36 40 42
```

### 续跑、错误重试和强制重跑

```bash
uv run python experiment/contrast2/run_contrast2.py --resume
uv run python experiment/contrast2/run_contrast2.py --resume --retry-errors
uv run python experiment/contrast2/run_contrast2.py --rerun-existing
```

单独使用 `--resume` 会跳过汇总表中所有已有项，包括错误项；与 `--retry-errors` 同时使用会保留成功项并重跑错误项。`--rerun-existing` 会强制重跑当前筛选范围内已有任务。

### 基线命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--models` | `all` | 全部模型，或逗号分隔的精确模型名 |
| `--seeds` | `36 40 42` | 一个或多个种子 |
| `--seed` | 无 | 单种子快捷覆盖；设置后覆盖 `--seeds` |
| `--train-totals` | `1000 3000` | 一个或两个训练总量 |
| `--ratios` | 全部九个比例 | 一个或多个精确比例字符串 |
| `--datasets` | `UKB YHD` | 一个或两个数据集 |
| `--workers` | `1` | 当前接受该参数，但模型任务仍顺序执行 |
| `--resume` | 关闭 | 跳过已有结果 |
| `--retry-errors` | 关闭 | 与 `--resume` 配合重跑错误项 |
| `--rerun-existing` | 关闭 | 强制重跑已有项 |

## HL 入口运行方式

### 运行一个种子的完整矩阵

```bash
uv run python experiment/contrast2/run_contrast2_hl.py --seed 42
```

默认写入 `experiment/contrast2/outputs_hl`。

### 指定独立输出根目录

建议每个种子使用独立目录：

```bash
uv run python experiment/contrast2/run_contrast2_hl.py \
  --seed 36 \
  --output-root experiment/contrast2/outputs_hl_seed36

uv run python experiment/contrast2/run_contrast2_hl.py \
  --seed 40 \
  --output-root experiment/contrast2/outputs_hl_seed40

uv run python experiment/contrast2/run_contrast2_hl.py \
  --seed 42 \
  --output-root experiment/contrast2/outputs_hl_seed42
```

### 并行 HL 任务

```bash
uv run python experiment/contrast2/run_contrast2_hl.py \
  --seed 42 --workers 2 \
  --output-root experiment/contrast2/outputs_hl_seed42
```

`--workers` 使用多进程执行独立 LLM 任务。提高并发会同时增加 API 请求速率、费用压力与本机资源占用。

### HL 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--seed` | `42` | 本次 36 个任务共用的种子 |
| `--workers` | `1` | LLM 任务并发进程数 |
| `--output-root` | `experiment/contrast2/outputs_hl` | HL 产物根目录 |

HL 入口没有数据集、训练总量、比例筛选或续跑参数。它每次都会执行完整 36 项矩阵，并重写固定汇总文件 `experiment/contrast2/contrast2_hl.csv`。连续运行多个种子时，应在下一次运行前保存当前汇总表。

## 输出目录与文件

### 基线结果

每个种子的汇总表：

```text
experiment/contrast2/contrast2_rerun_seed<seed>.csv
```

每个模型任务目录：

```text
experiment/outputs_rerun/contrast2/
└── seed<seed>/<数据集>/train<总量>/ratio<正>_<负>/<模型>/
```

汇总 CSV 包含模型、数据集、训练总量、类别比例、四项指标、TP、FP、FN、TN、`status` 和 `error`。任务目录包含：

| 文件 | 内容 |
| --- | --- |
| `predictions.csv` | 源行标识、真实标签、预测标签和正类概率 |
| `metrics.json` | 测试指标 |
| `split_manifest.json` | 冻结划分、比例、抽样替换和源行哈希 |
| `run_manifest.json` | 任务配置与状态 |
| `resolved_config.json` | 实际模型参数 |
| `environment.json` | 关键环境信息 |
| `model.joblib` / `model.deeptab` / `model.corels` | 拟合模型 |
| CORELS/EBM 附加文件 | 规则、谓词、预处理器或可解释性摘要 |
| `error.txt` | 失败任务的详细错误 |

### HL 结果

固定汇总表：

```text
experiment/contrast2/contrast2_hl.csv
```

它当前包含模型标识、数据集、训练总量和四项指标，不包含状态、错误或混淆矩阵字段。单任务目录：

```text
<output-root>/<数据集>/train<总量>/ratio<正>_<负>/<时间戳>/
```

其中包括 `probe_univariate_results.csv`、`probe_knowledge.md`、`heuristic_system.py`、`evolution_results.txt`、`iteration_log.json`、`final_comparison.txt`、`final_heuristic_model.py` 和 `heldout_test_summary.txt`。最终模型可直接调用 `predict(features)`。

每次任务创建新的时间戳目录；不要把新运行写进已有时间戳目录。

## 为 HL 汇总补全混淆矩阵

补全工具只扫描每个 `--output-roots` 根目录第一层、文件名符合 `contrast2_hl_<seed>.csv` 的汇总表。由于 HL 入口写出的固定文件名是 `experiment/contrast2/contrast2_hl.csv`，运行一个种子后需要将它复制到相应产物根目录并带上种子：

```bash
cp experiment/contrast2/contrast2_hl.csv \
  experiment/contrast2/outputs_hl_seed42/contrast2_hl_42.csv

uv run python experiment/contrast2/fill_contrast2_hl_confusion.py \
  --output-roots experiment/contrast2/outputs_hl_seed42
```

多个根目录使用逗号分隔：

```bash
uv run python experiment/contrast2/fill_contrast2_hl_confusion.py \
  --output-roots experiment/contrast2/outputs_hl_seed36,experiment/contrast2/outputs_hl_seed40,experiment/contrast2/outputs_hl_seed42
```

不传 `--output-roots` 时，工具自动检查 `experiment/contrast2/outputs_hl` 和匹配 `outputs_hl_*` 的目录。它会：

1. 从文件名解析种子。
2. 根据数据集、训练总量和比例找到最新时间戳产物。
3. 加载 `final_heuristic_model.py`。
4. 使用相同种子重建冻结测试集并重新预测。
5. 在 Specificity 后插入 TP、FP、FN、TN。
6. 原子更新 CSV；已有这四列的文件会跳过。

`--repo-root` 通常不需要设置，只有从非标准目录调用或仓库位置无法自动解析时才指定。

## 结果解读与注意事项

- 类别不平衡仅作用于训练集；验证集和测试集保持平衡，便于比较 Sensitivity 与 Specificity 的偏移。
- 应在同一模型、数据集、总量和种子内比较不同比例，再跨三个种子汇总。
- 极端比例可能触发训练有放回抽样；应结合 `split_manifest.json` 的唯一源行数解释结果。
- TP、FP、FN、TN 之和应为 1,000，与测试集大小一致。
- HL 汇总不记录失败状态；若行缺失，应检查控制台输出和对应产物目录是否生成完整的最终模型。
