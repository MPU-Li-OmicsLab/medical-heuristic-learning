# 对比实验扩展实施与训练报告

更新时间：2026-08-17（Asia/Shanghai）

## 1. 实施结论

已按 `EXPERIMENT_EXTENSION_PLAN.md` 完成环境配置、统一建模层、三个实验入口、
断点续跑、模型产物和文档修改。正式随机种子为 `36、40、42`，模型层自动类别
平衡均关闭；HL 未重跑。

当前共完成 **1090 / 1500** 行普通模型实验，所有已写入行均成功，没有 error 行：

| 实验 | 已完成 | 计划总数 | 状态 |
|---|---:|---:|---|
| contrast1 | 271 | 360 | 七个快速模型完整；EBM/DeepTab 部分完成 |
| contrast2 | 759 | 1080 | 七个快速模型完整；三个耗时模型各完成 1 格 |
| continuous_learning | 60 | 60 | 完整 |
| 合计 | 1090 | 1500 | 410 行待断点续跑 |

这里的“七个快速模型”指 LogisticRegression、DecisionTree、MLP、XGBoost、
LightGBM、APLR 和 CORELS。未完成行全部属于 EBM、FT-Transformer 和 ResNet
的 CPU 长任务；没有把未训练条件写成成功结果。

## 2. 环境与依赖验收

`uv sync --group dev` 已成功执行，当前环境版本为：

| 包 | 版本 |
|---|---|
| DeepTab | 2.0.0 |
| InterpretML core | 0.7.8 |
| APLR | 10.23.0 |
| CORELS | 1.1.29 |
| PyTorch | 2.9.1 |
| scikit-learn | 1.8.0 |
| XGBoost | 3.2.0 |
| LightGBM | 4.6.0 |

CORELS 官方包已在当前 Python 3.11 环境完成源码编译，并通过官方风格 toy-data
拟合/预测闸门。FT-Transformer 与 ResNet 均由 DeepTab 2.0.0 的公开 estimator
API 训练、预测和保存，不再使用实验脚本内的自实现版本。

安装说明：官方 CORELS 1.1.29 sdist 自带的旧版生成 C++ 与当前 Python 3.11 / NumPy
2 组合不兼容。本环境没有切换 fork，而是对官方 sdist 的 `_corels.pyx` 使用 Cython
3 重新生成 C++ 后构建并安装本机 wheel；GMP 不可用时按官方 `setup.py` 自动退回
非 GMP 构建。当前环境再次执行 `uv sync --group dev` 可通过。若在全新机器重建，
仍需准备 C++ 编译器并执行同一 Cython 重生成步骤，不能静默替换成其它 CORELS 实现。

## 3. 训练完整度明细

### contrast1

| seed | 已完成/120 | 说明 |
|---:|---:|---|
| 36 | 98/120 | 七个快速模型 84 格；EBM 12 格；FT/ResNet 各 1 格 |
| 40 | 89/120 | 七个快速模型 84 格；EBM 5 格 |
| 42 | 84/120 | 七个快速模型 84 格 |

七个完整模型在两个数据集、三个 seed 上的平均 F1：

| 模型 | n=10 | n=50 | n=100 | n=500 | n=1000 | n=3000 |
|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.570 | 0.581 | 0.621 | 0.662 | 0.672 | 0.682 |
| DecisionTree | 0.470 | 0.569 | 0.616 | 0.590 | 0.580 | 0.584 |
| MLP | 0.551 | 0.582 | 0.608 | 0.654 | 0.660 | 0.656 |
| XGBoost | 0.550 | 0.608 | 0.634 | 0.673 | 0.668 | 0.680 |
| LightGBM | 0.667 | 0.568 | 0.623 | 0.664 | 0.660 | 0.680 |
| APLR | 0.529 | 0.562 | 0.642 | 0.677 | 0.654 | 0.654 |
| CORELS | 0.485 | 0.610 | 0.616 | 0.655 | 0.634 | 0.601 |

主要现象：LogisticRegression 和 XGBoost 随训练量增加整体改善，n=3000 的平均
F1 分别为 0.682 和 0.680；DecisionTree 在 n=100 后没有继续稳定获益。n=10 的
LightGBM 平均 F1 较高，但其部分极小样本条件会退化为偏向单一类别的预测，不能只看
F1 下结论。

### contrast2

| seed | 已完成/360 | 说明 |
|---:|---:|---|
| 36 | 255/360 | 七个快速模型 252 格；EBM/FT/ResNet 各 1 格 |
| 40 | 252/360 | 七个快速模型完整 |
| 42 | 252/360 | 七个快速模型完整 |

七个完整模型在 1:1 训练比例下，跨数据集、训练量和 seed 的均值：

| 模型 | ACC | F1 | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| LogisticRegression | 0.688 | 0.676 | 0.646 | 0.730 |
| DecisionTree | 0.620 | 0.587 | 0.536 | 0.704 |
| MLP | 0.675 | 0.654 | 0.610 | 0.740 |
| XGBoost | 0.692 | 0.665 | 0.607 | 0.778 |
| LightGBM | 0.697 | 0.670 | 0.612 | 0.782 |
| APLR | 0.683 | 0.664 | 0.626 | 0.741 |
| CORELS | 0.668 | 0.646 | 0.602 | 0.734 |

类别比例极端时出现清晰的预测偏置：

- 正:负为 1:50 时，七个模型平均 Sensitivity 均不高于 0.134；APLR、CORELS
  等模型几乎全部预测为负类。
- 正:负为 50:1 时，除本配置下的 CORELS 外，其余模型平均 Sensitivity 为
  0.899–0.997，而 Specificity 仅为 0.012–0.185。
- 这正是关闭模型层自动平衡后的预期实验信号：训练前类别分布直接影响决策倾向。

所有已完成 contrast2 行均通过 `TP + FP + FN + TN == test_size` 校验。

### continuous_learning

持续学习已完整产生 60 行：Stage 1 的 30 行均为 `ok`，Stage 2 的 30 行均为
`continued`。六个数据集合源行两两不重叠；Stage 1 有 SIRS、无 SOFA，Stage 2
有 SOFA、无 SIRS。五个新增模型的 manifest 均记录
`true_stage2_SIRS_accessed=false`。

三 seed 的 ACC/F1 均值 ± 样本标准差：

| 模型 | Stage 1 ACC | Stage 1 F1 | Stage 2 ACC | Stage 2 F1 |
|---|---:|---:|---:|---:|
| LogisticRegression | 0.659±0.007 | 0.658±0.010 | 0.565±0.015 | 0.557±0.038 |
| DecisionTree | 0.582±0.008 | 0.580±0.008 | 0.534±0.031 | 0.539±0.008 |
| MLP | 0.637±0.012 | 0.632±0.014 | 0.597±0.010 | 0.602±0.020 |
| XGBoost | 0.659±0.018 | 0.665±0.014 | 0.637±0.023 | 0.650±0.006 |
| LightGBM | 0.659±0.003 | 0.658±0.005 | 0.641±0.009 | 0.661±0.006 |
| FT-Transformer | 0.652±0.006 | 0.671±0.017 | 0.604±0.015 | 0.592±0.017 |
| ResNet | 0.653±0.008 | 0.644±0.044 | 0.615±0.001 | 0.614±0.018 |
| EBM | 0.666±0.017 | 0.666±0.016 | 0.614±0.019 | 0.611±0.023 |
| APLR | 0.639±0.011 | 0.641±0.012 | 0.601±0.008 | 0.609±0.038 |
| CORELS | 0.590±0.022 | 0.577±0.021 | 0.547±0.014 | 0.569±0.043 |

Stage 2 只有 40 条训练数据并发生 SIRS→SOFA 漂移。在当前配置下，LightGBM 的
Stage 2 平均 F1 最高（0.661），其次为 XGBoost（0.650）；五个 prior-cascade
新增模型中 ResNet 的 ACC 最高（0.615），APLR 的 F1 为 0.609，EBM 的 F1 为
0.611。这里仅有三个 seed，报告描述性统计，不进行显著性推断。

## 4. 输出位置

- contrast1：`experiment/contrast1/contrast1_balance_rerun_seed{36,40,42}.csv`
- contrast2：`experiment/contrast2/contrast2_rerun_seed{36,40,42}.csv`
- continuous：`experiment/continuous_learning/continuous_baselines_rerun_results.csv`
- 模型、预测、指标和 manifest：`experiment/outputs_rerun/`

旧 CSV 与 HL 结果均未覆盖。DeepTab 任务保存 `.deeptab` 和最佳 checkpoint；
CORELS 保存官方模型、规则列表和谓词清单；普通模型保存 joblib 产物。

## 5. 待续跑任务

当前缺少 410 行，全部可由原命令安全断点续跑：

```bash
uv run python experiment/contrast1/run_contrast1_balance.py \
  --models EBM,FT-Transformer,ResNet --seeds 36 40 42 --resume

uv run python experiment/contrast2/run_contrast2.py \
  --models EBM,FT-Transformer,ResNet --seeds 36 40 42 --resume
```

当前机器无可用 GPU。实测一个 YHD/1000 的 FT-Transformer 任务约 401 秒，完整
DeepTab/EBM 剩余矩阵预计还需数小时到十余小时。必须等待上述命令实际完成并重新
校验 360/1080 行后，才能生成包含十个模型的 contrast1/contrast2 最终三 seed
均值/标准差；本报告没有对未完成条件进行插值或伪造。
