# contrast1：训练集规模对比实验

本目录用于研究训练集规模变化对模型性能的影响。当前包含 3 个脚本：

- `run_contrast1.py`：常规模型对比，训练集从剩余样本中随机抽样，不强制 1:1
- `run_contrast1_balance.py`：常规模型对比，训练集强制 1:1 平衡采样
- `run_contrast1_balance_hl.py`：启发式学习系统 `HL` 的平衡训练集版本

## 实验目标

在两个数据集上，固定平衡的验证集与测试集，系统性改变训练集规模，比较：

- 常规监督学习模型在不同训练样本量下的表现
- 启发式学习系统在平衡训练集下随样本量变化的表现

其中：

- `contrast1` 关注的是训练集规模
- `contrast2` 关注的是训练集正负比

## 数据与切分

### 数据集

- `UKB`：`./data/UKB.csv`，标签列 `label`
- `YHD`：`./data/YHD_bicarbonate.csv`，标签列 `hospital_expire_flag`

要求：

- 标签必须是二分类 `0/1`
- 正负样本数必须足够支撑平衡的 `val/test` 抽样

### 随机种子

- 默认 `seed=42`

### 固定切分规则

- `val`：1000 条，按 `1:1` 抽样，即 `500` 正 + `500` 负
- `test`：1000 条，按 `1:1` 抽样，即 `500` 正 + `500` 负
- `train_pool`：去掉 `val/test` 后剩余的样本

### 训练集规模

三个脚本都使用同一组训练集规模：

- `3000`
- `1000`
- `500`
- `100`
- `50`
- `10`

## 三个脚本的差异

### 1. `run_contrast1.py`

标准常规模型对比脚本：

- 训练集从 `train_pool` 中随机无放回抽样
- 不强制训练集类别平衡
- 但部分模型会启用类别不平衡修正：
  - `LogisticRegression` 使用 `class_weight="balanced"`
  - `DecisionTree` 使用 `class_weight="balanced"`
  - `XGBoost` 使用基于训练标签计算的 `scale_pos_weight`
  - `LightGBM` 使用 `is_unbalance=True`

输出文件：

- `contrast1/contrast1.csv`
- `contrast1/checkpoints/...`（仅 FT-Transformer）

### 2. `run_contrast1_balance.py`

平衡训练集版本：

- 训练集按 `1:1` 正负类采样
- 若某一类样本不足，会对该类启用有放回采样
- 其它整体实验框架与 `run_contrast1.py` 类似

输出文件：

- `contrast1/contrast1_balance.csv`
- `contrast1/checkpoints_balance/...`（仅 FT-Transformer）

### 3. `run_contrast1_balance_hl.py`

启发式学习系统 `HL` 的平衡训练集版本：

- 训练集同样按 `1:1` 采样，必要时允许有放回抽样
- 不训练 sklearn / xgboost / lightgbm 模型
- 调用项目主流程 `run_heuristic_learning(...)`
- 固定使用完整的 `U1_K1` 启发式学习流程，而不是做 probe 消融 <mccoremem id="project_memory" />

输出文件：

- `contrast1/contrast1_balance_hl.csv`
- 默认实验目录：`contrast1/outputs_balance_hl_seed<SEED>/...`

## 常规模型列表

`run_contrast1.py` 与 `run_contrast1_balance.py` 会比较以下模型：

- `LogisticRegression`
- `DecisionTree`
- `MLP`
- `XGBoost`
- `LightGBM`
- `FT-Transformer`

依赖说明：

- `XGBoost` 依赖 `xgboost`
- `LightGBM` 依赖 `lightgbm`
- `FT-Transformer` 依赖 `torch`
- 若依赖缺失，脚本不会整体退出，而是在 CSV 中写入 `status=missing_dependency`

## 预处理与训练细节

### 传统模型预处理

在 sklearn / xgboost / lightgbm 模型中：

- 数值列：中位数填补 + 标准化
- 类别列：众数填补 + One-Hot 编码

### FT-Transformer

`FT-Transformer` 使用脚本内置的单独特征处理逻辑：

- 数值列按训练集统计量填补
- 类别列在训练集上建立整数映射，未知值映射到 `__UNK__`
- 使用 `val` 做 early stopping 与 best checkpoint 选择
- 最终在 held-out `test` 上评估

### MLP 的小样本行为

实际代码行为如下：

- `run_contrast1.py` 中，`MLP` 在 `train_size >= 100` 时开启 `early_stopping`
- 若小样本下触发 sklearn 的类别不足报错，脚本会自动回退为 `early_stopping=False` 再尝试一次
- `run_contrast1_balance.py` 中，`MLP` 也是仅在 `train_size >= 100` 时开启 `early_stopping`
- 同时会把 `batch_size` 设为 `min(256, train_size)`，因此小训练集时会自动缩小 batch size

## 指标

所有汇总表都使用 held-out test 指标，主要包括：

- `ACC`
- `F1`
- `Sensitivity`
- `Specificity`

对于常规模型脚本，结果表还会记录：

- `best_epoch`
- `checkpoint`
- `status`
- `error`

其中：

- `best_epoch/checkpoint` 主要对 `FT-Transformer` 有意义
- 其它模型通常留空

## 安装依赖

如果需要完整运行所有常规模型与 HL 流程，建议在仓库根目录执行：

```bash
uv sync --group dev
```

额外说明：

- `xgboost`、`lightgbm`、`torch` 主要影响常规模型脚本
- `run_contrast1_balance_hl.py` 还依赖项目主流程和 LLM 配置

## 运行方式

### 1. 标准常规模型对比

```bash
uv run python contrast1/run_contrast1.py --seed 42 --workers 8
```

参数：

- `--seed`：控制 `val/test` 划分与训练集抽样
- `--workers`：CPU 任务的并行进程数

说明：

- 并行执行的是 CPU 模型任务块
- `FT-Transformer` 在后续单独串行训练

### 2. 平衡训练集常规模型对比

```bash
uv run python contrast1/run_contrast1_balance.py --seed 42 --workers 8
```

参数与标准版相同，但训练集采样方式改为 `1:1` 平衡采样。

### 3. 平衡训练集 HL 对比

运行前需要配置大模型环境变量，例如：

```bash
export DEEPSEEK_API_KEY="你的 key"
uv run python contrast1/run_contrast1_balance_hl.py --seed 42 --workers 1
```

参数：

- `--seed`：控制 `val/test` 划分与训练集采样
- `--workers`：HL 实验并发数
- `--output-root`：可选，自定义 HL 单次实验输出根目录

可覆盖的 LLM 环境变量：

- `CONTRAST1_HL_BASE_URL`
- `CONTRAST1_HL_KEY_ENV`
- `CONTRAST1_HL_MODEL`
- `CONTRAST1_HL_TEMPERATURE`

注意：

- `workers` 过大时更容易遇到 API 限流
- HL 任务建议从 `1` 开始逐步提升并发

## 输出说明

### `run_contrast1.py`

生成：

- `contrast1/contrast1.csv`
- `contrast1/checkpoints/<DATASET>/train<TRAIN_SIZE>/seed<SEED>_best.pth`

`contrast1.csv` 当前列为：

- `模型`
- `数据集`
- `训练集数据量`
- `ACC`
- `F1`
- `Sensitivity`
- `Specificity`
- `best_epoch`
- `checkpoint`
- `status`
- `error`

### `run_contrast1_balance.py`

生成：

- `contrast1/contrast1_balance.csv`
- `contrast1/checkpoints_balance/<DATASET>/train<TRAIN_SIZE>/seed<SEED>_best.pth`

CSV 列与 `contrast1.csv` 相同。

### `run_contrast1_balance_hl.py`

生成：

- `contrast1/contrast1_balance_hl.csv`
- `contrast1/outputs_balance_hl_seed<SEED>/<DATASET>/train<TRAIN_SIZE>/<TIMESTAMP>/...`

若显式传入 `--output-root`，则实验目录会写到指定位置。

`contrast1_balance_hl.csv` 的列为：

- `模型`
- `数据集`
- `训练集数据量`
- `ACC`
- `F1`
- `Sensitivity`
- `Specificity`
- `best_epoch`
- `checkpoint`
- `status`
- `error`

其中：

- `模型` 固定为 `HL`
- `best_epoch/checkpoint` 当前固定为空

单个 HL 实验目录通常包含：

- `final_heuristic_model.py`
- `final_comparison.txt`
- `evolution_results.txt`
- `iteration_log.json`
- `heldout_test_summary.json`
- `heldout_test_summary.txt`

## 复现性说明

- 数据切分与训练集抽样由 `--seed` 控制，可复现
- `FT-Transformer` 尝试启用尽量确定性的设置，但在不同硬件、驱动或算子路径下仍可能有细微差异
- HL 脚本也会把 `seed` 传入 `RunConfig.random_seed`

## 常见现象

- Pandas 可能提示 `is_categorical_dtype is deprecated`，属于兼容性警告
- 小训练集下，`MLP` 的 batch size 可能被自动裁剪，这通常不影响结果
- LightGBM 可能提示特征名相关 warning，一般不影响结果
- torch 可能提示 `enable_nested_tensor ... False because norm_first was True`，这是性能路径提示
