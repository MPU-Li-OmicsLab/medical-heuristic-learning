# contrast2：对比模型实验（二阶段：训练集正负比）

## 实验目标

在两个数据集（UKB、YHD）上，对 6 个对比模型进行二分类对比实验。与 contrast1 的区别是：本阶段系统性改变训练集的正负样本比例（允许重采样补足），考察在类别不平衡下的性能。

## 数据与切分

- 数据集
  - UKB：`./data/UKB.csv`，标签列 `label`
  - YHD：`./data/YHD_bicarbonate.csv`，标签列 `hospital_expire_flag`
- 随机种子：默认 `seed=42`
- 验证集/测试集切分（固定可复现）
  - `val`：1000 条，按 1:1 抽样（500 正 + 500 负）
  - `test`：1000 条，按 1:1 抽样（500 正 + 500 负）

## 训练集设置

- 训练集总数：`1000`、`3000`
- 训练集正负比（正:负）：
  - `1:1、1:2、2:1、1:5、5:1、1:10、10:1、1:50、50:1`
- 若某一类样本不足以达到目标数量，会在训练集中对该类 **with replacement 重采样**，以满足目标正负数量。

## 对比模型（6 个）

- XGBoost
- LightGBM
- 决策树（DecisionTree）
- MLP（sklearn MLPClassifier）
- FT-Transformer（torch）
- 逻辑回归（LogisticRegression）

## 指标

输出指标：

- ACC
- F1
- Sensitivity（TPR）
- Specificity（TNR）

## 安装依赖（uv）

需要 `xgboost/lightgbm/torch`（已加入项目 dev 依赖组）：

```bash
cd /home/yk/medical-heuristic-learning
uv sync --group dev
```

## 运行方式

从仓库根目录运行：

```bash
uv run python contrast2/run_contrast2.py --seed 42 --workers 8
```

- `--seed`：控制 val/test 抽样与训练集抽样的随机种子
- `--workers`：CPU 模型的多进程并行度
  - 并行范围：XGBoost / LightGBM / 决策树 / MLP / 逻辑回归
  - FT-Transformer：串行训练（避免 GPU/资源竞争）

## 输出说明

运行完成后在 `contrast2/` 目录下生成：

- `contrast2.csv`：对比实验总表
  - 列：`模型, 数据集, 训练集数据量, 训练集正负比, ACC, F1, Sensitivity, Specificity, status, error`
  - 行顺序：按 `模型 → 数据集 → 训练集数据量（1000→3000）→ 正负比（按 README 列表顺序）` 排序
- `checkpoints/`：FT-Transformer 的最优 checkpoint（按 val 选择 best）
  - 路径示例：`contrast2/checkpoints/YHD/train3000/ratio1_10/seed42_best.pth`

