# Continuous Learning（持续学习启发式学习框架）

本文件夹在不修改 `./hl` 任何已有逻辑的前提下，基于现有启发式学习流程搭建一个“两阶段持续学习”框架，用于模拟特征漂移（特征删除/恢复/新增/重命名）并在第二阶段基于第一阶段的输出继续迭代。

入口脚本：`continuous_learning/run_continuous_learning.py`

## 目标与核心约束

- 持续学习含义：当出现特征漂移时，在“已有启发式系统”的基础上做修改，适配新数据并继续迭代优化。
- 不覆盖旧系统：旧系统输出目录只作为输入读取与记录；所有新产物写入 `continuous_learning/outputs/...` 的新目录。
- 两阶段设置（固定）：
  - Stage1：训练集 1000（1:1）
  - Stage2：训练集 10（1:1），在 Stage1 的输出目录基础上继续迭代
  - 两阶段的验证集/测试集：各 500（1:1）
- 多 seed：默认 `36,40,42`
- 同时对比 baseline：XGBoost、LightGBM、决策树、MLP、FT-Transformer、逻辑回归（与 HL 同样记录两阶段结果）

## 数据集（默认 MIMIC）

- 数据路径：`./data/MIMIC.csv`
- 标签列：`death_within_hosp_28days`（0/1）

可以通过参数切换为 UKB/YHD（若你提供对应数据与标签列）。

## 入口参数（关键）

### 数据与标签

- `--datasets`：默认 `MIMIC`，可填 `MIMIC` / `UKB` / `YHD`（逗号分隔）
- `--mimic-csv`：默认 `./data/MIMIC.csv`
- `--mimic-label-col`：默认 `death_within_hosp_28days`
- `--mimic-prev-hl-outdir`：可选。上一套 HL 输出目录（若为空则 Stage1 从零开始）

### 两阶段特征漂移（必须写清楚）

Stage1：
- `--stage1-drop-cols`：Stage1 删除列（逗号分隔）
- `--stage1-add-cols`：Stage1 新增列（逗号分隔，可选）
- `--stage1-rename-cols`：Stage1 重命名列（`old:new`，逗号分隔，可选）
- `--stage1-change-note`：Stage1 变更说明（必须提供一句话/一段话）

Stage2：
- `--stage2-drop-cols`：Stage2 删除列（逗号分隔）
- `--stage2-add-cols`：Stage2 新增列（逗号分隔，可选）
- `--stage2-rename-cols`：Stage2 重命名列（`old:new`，逗号分隔，可选）
- `--stage2-change-note`：Stage2 变更说明（必须提供一句话/一段话）

重要规则（恢复列）：

- Stage2 会自动把「Stage1 删除但 Stage2 不再删除」的列视为“恢复列”，加入 Stage2 的 `added_cols`，并在输出文件中记录。

### 输出与 LLM

- `--output-root`：默认 `./continuous_learning/outputs`
- `--seeds`：默认 `36,40,42`
- `--llm-base-url`：默认 `https://api.deepseek.com/v1`
- `--llm-key-env`：默认 `DEEPSEEK_API_KEY`
- `--llm-model`：默认 `deepseek-v4-pro`
- `--llm-temperature`：默认 `0.0`

## 输出结构与追溯

每次 HL 的输出目录（每个 seed × 每个阶段一份）：

`continuous_learning/outputs/<DATASET>/seed<SEED>/<STAGE>/HL/<TIMESTAMP>/`

其中会写入关键追溯文件：

- `adaptation_spec.json`
  - 记录：删除列/新增列/重命名列/变更说明/上一套 HL 输出目录（prev_hl_out_dir）/采样信息等
- `heldout_test_summary.json` / `heldout_test_summary.txt`
  - 记录：阶段指标（ACC/F1/Sensitivity/Specificity）与本阶段漂移信息
- `probe_univariate_results_prev.csv`、`probe_knowledge_prev.md`
  - 从“上一套 HL 输出目录（或 Stage1 输出目录）”复制一份，不覆盖原文件
- `probe_univariate_results.csv`、`probe_knowledge.md`
  - 在“复制旧表”的基础上，同步删除列、追加新增/恢复列信息，形成新表
- `v0_prompt.txt` / `v0_error_analysis.txt` / `v0_attempt_*.txt`
  - 记录“持续学习 V0 生成”的提示词与失败重试

baseline 输出目录（每个 seed 一份，内含两阶段结果）：

`continuous_learning/outputs/<DATASET>/seed<SEED>/baselines/<TIMESTAMP>/`

总表：

- `continuous_learning/continuous_results.csv`
  - 列：`模型, 数据集, seed, 阶段, ACC, F1, Sensitivity, Specificity, status, error, out_dir`
  - HL 与 baseline 都会写两阶段结果

## 探针与 V0 的持续学习逻辑（概述）

Stage1：

- 无上一套目录：探针直接基于 Stage1 的训练数据生成（并写入新目录）。
- 有上一套目录：先复制旧 probe，再删除 dropped、追加 added/rename 对应条目，生成新 probe。

Stage2：

- prev_hl_out_dir 固定为 Stage1 的 HL 输出目录（继续迭代的“上一套系统”）。
- 探针同样遵循“复制旧表→同步删除→追加新增/恢复”。
- V0 生成会把 Stage1 的 `final_heuristic_model.py` 作为蓝本写入 prompt，要求生成新的 `predict_v0`，并明确说明删/增/恢复/重命名与变更原因。

迭代步骤与 `./hl` 中普通启发式学习一致：由 `hl.orchestrator.run_heuristic_learning(...)` 执行。

## baseline 的两阶段训练/迁移说明（实现口径）

- LogisticRegression：`warm_start=True`，Stage2 在 Stage1 基础上继续拟合
- MLP：`warm_start=True`，Stage2 在 Stage1 基础上继续拟合
- DecisionTree：不做权重迁移，Stage2 记录为 `retrain`
- XGBoost：若依赖可用，Stage2 使用 `xgb_model` 从 Stage1 booster 继续训练
- LightGBM：若依赖可用，Stage2 使用 `init_model` 从 Stage1 booster 继续训练
- FT-Transformer：若 `torch` 可用，Stage2 从 Stage1 的权重继续训练（并保存 `seed*_best.pt`）

注意：baseline 会使用各阶段各自的特征集合，内部会先对齐 feature columns（缺失列补 NaN），再做简单数值化以便训练。

## 运行示例（MIMIC，两阶段删除/恢复）

示例：Stage1 删除 `Blood Lactate`，Stage2 删除 `SIRS`，并恢复 `Blood Lactate`：

```bash
cd /home/yk/medical-heuristic-learning
export DEEPSEEK_API_KEY="你的key"

uv run python continuous_learning/run_continuous_learning.py \
  --datasets MIMIC \
  --mimic-csv "/home/yk/medical-heuristic-learning/data/MIMIC.csv" \
  --mimic-label-col "death_within_hosp_28days" \
  --seeds "36,40,42" \
  --stage1-drop-cols "Blood Lactate" \
  --stage1-change-note "阶段1：模拟该检验在新环境中采集缺失/不可用，先移除以减少噪声并让规则适应。" \
  --stage2-drop-cols "SIRS" \
  --stage2-change-note "阶段2：SIRS 定义发生变化导致分布漂移，移除该列；同时恢复 Blood Lactate，因为该检验已恢复采集，需要重新纳入规则并在阶段1基础上继续迭代。" \
  --output-root "./continuous_learning/outputs"
```

