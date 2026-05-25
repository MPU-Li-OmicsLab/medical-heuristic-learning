# contrast2：训练集正负比对比实验

本目录用于研究训练集类别分布变化对模型表现的影响。和 `contrast1` 不同，`contrast2` 固定验证集与测试集为平衡划分，系统性改变训练集正负样本比例，并比较：

- 常规监督学习基线模型
- 启发式学习系统 `HL`

当前目录下包含 3 个脚本：

- `run_contrast2.py`：运行常规模型对比，输出 `contrast2.csv`
- `run_contrast2_hl.py`：运行启发式学习系统对比，输出 `contrast2_hl.csv`
- `fill_contrast2_hl_confusion.py`：为已有的 HL 结果 CSV 回填 `TP/FP/FN/TN`

## 实验设置

### 数据集

- `UKB`：`./data/UKB.csv`，标签列 `label`
- `YHD`：`./data/YHD_bicarbonate.csv`，标签列 `hospital_expire_flag`

要求：

- 标签必须是二分类 `0/1`
- 正负样本数必须足够支撑平衡的验证集与测试集抽样

### 随机种子与切分

- 默认 `seed=42`
- `val`：1000 条，按 `1:1` 抽样，即 `500` 正 + `500` 负
- `test`：1000 条，按 `1:1` 抽样，即 `500` 正 + `500` 负
- `train_pool`：去掉 `val/test` 后剩余的样本

### 训练集设置

- 训练集总数：`1000`、`3000`
- 训练集正负比（正:负）：
  - `1:1`
  - `1:2`
  - `2:1`
  - `1:5`
  - `5:1`
  - `1:10`
  - `10:1`
  - `1:50`
  - `50:1`

对每个 `(数据集, 训练集总数, 正负比)` 组合，都会根据目标比例计算正负样本目标数，再从 `train_pool` 中采样：

- 若样本足够，则无放回采样
- 若某一类样本不足，则该类自动改为有放回采样，以满足目标数量

## 常规模型对比

`run_contrast2.py` 会比较以下模型：

- `LogisticRegression`
- `DecisionTree`
- `MLP`
- `XGBoost`
- `LightGBM`
- `FT-Transformer`

说明：

- `XGBoost` 依赖 `xgboost`
- `LightGBM` 依赖 `lightgbm`
- `FT-Transformer` 依赖 `torch`
- 如果依赖缺失，脚本不会报错退出，而是在结果表中写入 `status=missing_dependency`

### 预处理方式

在传统模型中，特征预处理由脚本自动完成：

- 数值列：中位数填补 + 标准化
- 类别列：众数填补 + One-Hot 编码

`FT-Transformer` 使用脚本内置的单独特征处理逻辑：

- 数值列：转为数值并用训练集统计量填补
- 类别列：在训练集上建立类别到整数的映射，未知值映射到 `__UNK__`
- 验证集用于 early stopping 和 best checkpoint 选择

### 运行方式

从仓库根目录运行：

```bash
uv run python experiment/contrast2/run_contrast2.py --seed 42 --workers 8
```

参数：

- `--seed`：控制 `val/test` 划分与训练集采样
- `--workers`：CPU 任务的并行进程数

说明：

- 并行执行的是按 `(数据集, 训练集总数, 正负比)` 划分的 CPU 模型任务块
- `FT-Transformer` 不走该并行块，而是在后续串行训练，避免资源竞争更严重

### 输出文件

运行后会在 `experiment/contrast2/` 下生成：

- `contrast2.csv`
- `checkpoints/`（仅当 `FT-Transformer` 成功训练时产生）

`contrast2.csv` 的列为：

- `模型`
- `数据集`
- `训练集数据量`
- `训练集正负比`
- `ACC`
- `F1`
- `Sensitivity`
- `Specificity`
- `TP`
- `FP`
- `FN`
- `TN`
- `status`
- `error`

排序顺序为：

- `模型`
- `数据集`
- `训练集数据量`（`1000 -> 3000`）
- `训练集正负比`（按上面比例列表顺序）

`FT-Transformer` checkpoint 路径示例：

`experiment/contrast2/checkpoints/YHD/train3000/ratio1_10/seed42_best.pth`

## 启发式学习对比（HL）

`run_contrast2_hl.py` 使用项目主流程 `run_heuristic_learning(...)` 运行启发式学习系统，并在本阶段固定使用：

- `run_univariate_probe=True`
- `run_knowledge_probe=True`
- `run_v0_generation=True`
- `run_iterations=True`

也就是说，这里比较的是完整的 `U1_K1` 启发式学习流程，而不是 probe 消融。<mccoremem id="project_memory" />

### 运行方式

运行前需要配置大模型 API Key，例如：

```bash
export DEEPSEEK_API_KEY="你的 key"
uv run python experiment/contrast2/run_contrast2_hl.py --seed 42 --workers 1
```

参数：

- `--seed`：控制 `val/test` 划分与训练集采样
- `--workers`：HL 实验并发进程数
- `--output-root`：单次 HL 实验目录的输出根目录，默认位于 `./experiment/contrast2/outputs_hl`

注意：

- `workers` 增大后会并发调用 LLM 接口，可能触发限流或显著增加成本
- 建议先从 `1` 开始，再逐步提高

### HL 输出

脚本会产出两类结果：

- 汇总表：`experiment/contrast2/contrast2_hl.csv`
- 每个实验的独立输出目录：`<output-root>/<DATASET>/train<TRAIN_TOTAL>/ratio<POS_NEG>/<TIMESTAMP>/`

例如：

- `experiment/contrast2/outputs_hl/UKB/train1000/ratio1_5/20260525_123456/`

单个实验目录通常包含：

- `final_heuristic_model.py`
- `final_comparison.txt`
- `evolution_results.txt`
- `iteration_log.json`
- `heldout_test_summary.txt`

`contrast2_hl.csv` 当前写出的列为：

- `模型`
- `数据集`
- `训练集数据量`
- `ACC`
- `F1`
- `Sensitivity`
- `Specificity`

其中：

- `模型` 字段形如 `HL(1:5)`，训练集正负比被编码在这里，而不是单独一列
- 排序顺序为 `数据集 -> 训练集数据量 -> 正负比`

需要注意：

- 脚本内部虽然生成了更完整的结果字典，包括 `status`、`out_dir`、`error` 等字段，但最终写入 `contrast2_hl.csv` 时只保留上面 7 列
- 如果某次 HL 运行失败，该行对应指标通常会为空

## 辅助脚本：回填 HL 混淆矩阵

`fill_contrast2_hl_confusion.py` 用于给已有的 HL 结果 CSV 补充以下列：

- `TP`
- `FP`
- `FN`
- `TN`

它的工作方式是：

- 从文件名 `contrast2_hl_<seed>.csv` 中解析随机种子
- 根据每行的 `数据集`、`训练集数据量` 与 `模型=HL(x:y)` 推断对应实验配置
- 在输出目录中查找最新一次对应的 HL 实验目录
- 读取其中的 `final_heuristic_model.py`
- 在同一 seed 对应的 held-out test 上重新预测并回填混淆矩阵

运行方式：

```bash
uv run python experiment/contrast2/fill_contrast2_hl_confusion.py --output-roots ./experiment/contrast2/outputs_hl_42
```

参数：

- `--repo-root`：仓库根目录，默认自动推断
- `--output-roots`：逗号分隔的多个 HL 输出根目录；若留空，则自动扫描 `./experiment/contrast2/outputs_hl` 与 `./experiment/contrast2/outputs_hl_*`

注意：

- 该脚本面向的是命名为 `contrast2_hl_<seed>.csv` 的结果文件
- 当前 `run_contrast2_hl.py` 默认输出的汇总文件名是固定的 `experiment/contrast2/contrast2_hl.csv`，如果要配合这个补全脚本使用，通常需要你自己按 seed 另存为对应命名格式

## 依赖安装

如果需要完整运行所有常规模型和 HL 流程，建议在仓库根目录执行：

```bash
uv sync --group dev
```

额外说明：

- `xgboost`、`lightgbm`、`torch` 主要影响 `run_contrast2.py`
- HL 相关脚本还依赖项目主流程所需的 LLM 配置与运行环境
