# contrast1：对比模型实验（UKB + YHD）

## 实验目标

在两个数据集（UKB、YHD）上，使用 6 个对比模型进行二分类对比，输出统一的指标表 `contrast1.csv`，用于与启发式规则系统的结果对照。

## 数据与切分

- 数据集
  - UKB：`./data/UKB.csv`，标签列 `label`
  - YHD：`./data/YHD_bicarbonate.csv`，标签列 `hospital_expire_flag`
- 随机种子：默认 `seed=42`（命令行可改）
- 切分规则（固定可复现）
  - 验证集 `val`：1000 条，按 1:1 抽样（500 正 + 500 负）
  - 测试集 `test`：1000 条，按 1:1 抽样（500 正 + 500 负）
  - 训练集 `train`：从剩余样本池随机抽取（不强制 1:1）
- 训练集规模：`3000, 1000, 500, 100, 50, 10`

## 对比模型

- XGBoost
- LightGBM
- 决策树（DecisionTree）
- MLP（sklearn MLPClassifier）
- FT-Transformer（torch）
- 逻辑回归（LogisticRegression）

## 指标

输出指标为：

- ACC
- F1
- Sensitivity（TPR）
- Specificity（TNR）

## 安装依赖（uv）

本实验依赖 `xgboost/lightgbm/torch`（已加入项目 dev 依赖组）。

```bash
cd /home/yk/medical-heuristic-learning
uv sync --group dev
```

## 运行方式

从仓库根目录运行：

```bash
uv run python contrast1/run_contrast1.py --seed 42 --workers 8
```

- `--seed`：控制 val/test 平衡抽样、以及 train 抽样的随机种子
- `--workers`：用于 CPU 模型的多进程并行度
  - 并行范围：XGBoost / LightGBM / 决策树 / MLP / 逻辑回归
  - FT-Transformer：串行训练（避免 GPU/资源竞争与不必要的进程间开销）

## 输出说明

运行完成后，主要产物在 `contrast1/` 下：

- `contrast1.csv`：对比实验总表
  - 列：`模型, 数据集, 训练集数据量, ACC, F1, Sensitivity, Specificity, best_epoch, checkpoint, status, error`
  - `best_epoch/checkpoint`：仅 FT-Transformer 使用（其它模型为空）
- `checkpoints/`：FT-Transformer 的最优 checkpoint（按 val 选择 best）
  - 路径示例：`contrast1/checkpoints/YHD/train1000/seed42_best.pth`
  - `.pth` 内包含：模型权重 `state_dict`、特征处理映射（列名、数值填充中位数、类别映射等）、`seed/best_epoch/best_val_metrics`
- `warnings.log`：若使用 `2> contrast1/warnings.log` 重定向 stderr，可查看运行警告信息

## 复现性说明

- 数据切分与训练集抽样是可复现的（由 `--seed` 控制）。
- FT-Transformer 启用了尽量确定性的设置（manual_seed、cuDNN deterministic 等），但在不同硬件/驱动/算子路径下仍可能存在极小差异。

## 常见警告含义（参考）

- Pandas：`is_categorical_dtype is deprecated`
  - 兼容性提示，不影响结果
- sklearn MLP：`batch_size ... larger than sample size. It is going to be clipped`
  - 小训练集下 batch_size 自动裁剪，不影响正确性
- LightGBM：`X does not have valid feature names...`
  - Pipeline 里特征名提示，一般不影响结果
- torch：`enable_nested_tensor ... False because norm_first was True`
  - 性能路径提示，不影响结果

## 常见问题

- MLP 在 `train_size=10` 可能触发 sklearn 的 early_stopping 内部划分报错（验证集样本过少导致类别不足）。此时建议关闭 MLP 的 early_stopping 或增大 train_size。
