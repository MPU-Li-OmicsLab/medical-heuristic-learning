# Continuous Learning

本目录现在位于 `experiment/continuous_learning/`，用于保存持续学习相关文档与实验输出；持续学习启发式学习的代码主干位于 `hl/continuous_learning/`。实验脚本已拆分为共享数据流模块、HL 持续学习脚本、Baseline 对比脚本和总控入口脚本。

## 新结构
- 持续学习主干包：`hl/continuous_learning/`
- 主入口函数：`hl.continuous_learning.run_continuous_learning(...)`
- 共享数据流模块：`experiment/continuous_learning/continuous_learning_experiment_common.py`
- HL 持续学习脚本：`experiment/continuous_learning/run_continuous_learning_hl.py`
- Baseline 对比脚本：`experiment/continuous_learning/run_continuous_learning_baselines.py`
- 总控入口：`experiment/continuous_learning/run_continuous_learning_experiment.py`
- 实验输出目录：`experiment/continuous_learning/outputs/`
- HL 结果总表：`experiment/continuous_learning/continuous_hl_results.csv`
- Baseline 结果总表：`experiment/continuous_learning/continuous_baseline_results.csv`
- 合并结果总表：`experiment/continuous_learning/continuous_results.csv`

## 设计目标
- 将“持续学习启发式学习主干”和“外部 baseline 对比实验”彻底拆开。
- 持续学习主干沿用 `hl/orchestrator` 的模块化风格，拆成 probe1、probe2、v0 生成、主编排器等独立文件。
- 在发生特征漂移时，基于旧策略与新数据共同生成新策略，而不是从零开始重训规则。
- 第二阶段在生成新 `v0` 后，直接调用 `hl.orchestrator.iteration_step.run_iterations_task(...)` 做迭代，不再绕回普通 `hl.orchestrator.run_heuristic_learning(...)`。

## 代码布局
- `hl/continuous_learning/config.py`
  持续学习专用配置对象：`DriftConfig`、`ContinuousLearningConfig`、`ContinuousLearningResult`。
- `hl/continuous_learning/univariate_probe_step.py`
  负责 Probe 1 的持续学习更新：复制旧单变量分析、同步删除和重命名、追加新增或恢复变量的统计结果。
- `hl/continuous_learning/knowledge_probe_step.py`
  负责 Probe 2 的持续学习更新：复制旧知识表、同步删除和重命名、为新增或恢复变量补充知识条目。
- `hl/agent/continuous_prompts.py`
  集中维护持续学习专用 prompt，包括知识探测、连续版 `v0` 生成与迭代优化提示词。
- `hl/continuous_learning/v0_generation_step.py`
  负责持续学习版 `v0` 的提示词构造与生成，会把旧 `final_heuristic_model.py` 作为蓝本。
- `hl/continuous_learning/iteration_step.py`
  持续学习专用迭代优化逻辑，复用标准 HL 的评估/校验策略，但使用独立 prompt 构造。
- `hl/continuous_learning/main_orchestrator.py`
  持续学习主编排器，统一串联 probe 更新、连续版 `v0` 生成、迭代优化、最终模型导出。
- `experiment/continuous_learning/continuous_learning_experiment_common.py`
  共享实验数据流，统一处理固定实验配置、两阶段漂移配置、平衡切分、训练采样以及结果 CSV 写入。
- `experiment/continuous_learning/run_continuous_learning_hl.py`
  只运行两阶段 HL 持续学习，并保存 HL 专属结果表。
- `experiment/continuous_learning/run_continuous_learning_baselines.py`
  只运行 Baseline 对比模型，并复用同一份两阶段数据流。
- `experiment/continuous_learning/run_continuous_learning_experiment.py`
  总控入口，依次调用 HL 与 Baseline 两个脚本逻辑，并写出合并总表。

## 持续学习主干接口

持续学习主干入口为：

```python
from hl.config import LLMConfig
from hl.continuous_learning import ContinuousLearningConfig, DriftConfig, run_continuous_learning
```

调用形式为：

```python
result = run_continuous_learning(
    train_df=train_df,
    test_df=val_df,
    label_col=label_col,
    llm_cfg=llm_cfg,
    continuous_cfg=ContinuousLearningConfig(
        task_description="Continuous learning example.",
        drift=DriftConfig(
            dropped_cols=("old_feature",),
            added_cols=("new_feature",),
            renamed_cols=(("old_name", "new_name"),),
            change_note="Describe the drift clearly.",
            prev_hl_out_dir=prev_out_dir,
        )
    ),
)
```

返回值 `result` 中包含：
- `out_dir`
- `heuristic_path`
- `final_model_path`

## 默认输出目录规则
- 普通 HL 在 `RunConfig.output_dir is None` 时，会写到 `out/<timestamp>/`。
- 持续学习 HL 在 `ContinuousLearningConfig.output_dir is None` 时，会写到 `out/<timestamp>_continuous_learning/`。
- 如果显式传入 `output_dir`，则完全使用传入路径。

## 持续学习主干的执行顺序
- 校验 `train_df`、`test_df` 与 `label_col`。
- 根据 `ContinuousLearningConfig` 中的漂移与运行配置记录当前上下文。
- 更新 Probe 1：
  读取旧 `probe_univariate_results.csv`，删除失效特征，同步 rename，并对新增或恢复特征补充新分析。
- 更新 Probe 2：
  读取旧 `probe_knowledge.md`，删除失效特征，同步 rename，并对新增或恢复特征补充新知识。
- 生成连续学习版 `v0`：
  读取旧 `final_heuristic_model.py` 作为 blueprint，结合漂移信息、新 probe 结果和任务描述构造 prompt。
- 直接调用 `run_iterations_task(...)` 做版本迭代。
- 从所有版本中选择最佳版本并导出新的 `final_heuristic_model.py`。

## 提示词与追溯文件

持续学习输出目录中会写入以下关键文件：
- `continuous_learning_context.json`
  记录本次持续学习的 `task_description`、漂移配置与旧输出目录。
- `probe_univariate_results_prev.csv`
  上一阶段或上一套系统的 Probe 1 快照。
- `probe_univariate_results.csv`
  本轮更新后的 Probe 1 结果。
- `probe_knowledge_prev.md`
  上一阶段或上一套系统的 Probe 2 快照。
- `probe_knowledge.md`
  本轮更新后的 Probe 2 结果。
- `iteration_log.json`
  逐轮迭代日志。
- `final_comparison.txt`
  `v0`、最佳版本、最后版本的指标对比。

## 实验入口职责

- 共享模块 `continuous_learning_experiment_common.py` 负责：
- 写死实验配置，不再解析命令行参数。
- 固定读取 `data/merged_by_subject_id_complete_rows_without_unit_cols_renamed.csv`。
- 固定数据集为 `MIMIC`，标签列为 `death_within_hosp_28days`。
- 固定随机种子为 `36`、`40`、`42`。
- 固定两阶段特征集合：
  Stage1 使用 `SIRS`，Stage2 删除 `SIRS` 并增加 `SOFA`。
- 应用两阶段特征漂移。
- 做平衡抽样：
  Stage1 训练集 1000，Stage2 训练集 10，验证集和测试集固定为 500。
- `run_continuous_learning_hl.py` 负责：
- 调用 `hl.continuous_learning.run_continuous_learning(...)` 完成两个阶段的 HL 持续学习。
- 在 held-out test 上评估 HL。
- 汇总到 `experiment/continuous_learning/continuous_hl_results.csv`。
- `run_continuous_learning_baselines.py` 负责：
- 训练并评估 baseline：
  `LogisticRegression`、`MLP`、`DecisionTree`、`XGBoost`、`LightGBM`、`FT-Transformer`。
- 汇总到 `experiment/continuous_learning/continuous_baseline_results.csv`。
- `run_continuous_learning_experiment.py` 负责：
- 依次运行 HL 与 Baseline 两条链路。
- 汇总到 `experiment/continuous_learning/continuous_results.csv`。

## 数据流一致性
- HL 与 Baseline 不再各自实现一套取数逻辑，而是统一复用 `continuous_learning_experiment_common.py`。
- 两边都使用同一套：
  平衡划分、训练采样、阶段定义、漂移应用和随机种子规则。
- 共享模块会在阶段 manifest 中记录 source row ids，因此可以追溯并核对两个脚本在相同 seed 下得到的是同一批样本。

## 两阶段漂移规则
- Stage1 使用原始 ICU 基线特征集合，其中包含 `SIRS`。
- Stage2 删除 `SIRS`，并增加 `SOFA`。
- Stage2 的 `prev_hl_out_dir` 固定指向 Stage1 的 HL 输出目录，因此第二阶段总是在第一阶段规则基础上继续适配。

## 运行实验示例

```bash
cd /home/xw/medical-heuristic-learning
export DEEPSEEK_API_KEY="你的key"

uv run python experiment/continuous_learning/run_continuous_learning_hl.py
uv run python experiment/continuous_learning/run_continuous_learning_baselines.py
uv run python experiment/continuous_learning/run_continuous_learning_experiment.py
```

## 最小代码示例

如果你只想调用持续学习主干，而不跑 baseline，可以直接在 Python 中调用：

```python
from pathlib import Path

import pandas as pd

from hl.config import LLMConfig
from hl.continuous_learning import ContinuousLearningConfig, DriftConfig, run_continuous_learning


data = pd.read_csv("./data/MIMIC.csv")
label_col = "death_within_hosp_28days"

train_df = data.iloc[:1000].copy()
val_df = data.iloc[1000:1500].copy()

llm_cfg = LLMConfig(
    base_url="https://api.deepseek.com/v1",
    api_key_env="DEEPSEEK_API_KEY",
    model_name="deepseek-v4-pro",
    temperature=0.0,
)

continuous_cfg = ContinuousLearningConfig(
    output_dir=None,
    run_univariate_probe=True,
    run_knowledge_probe=True,
    run_v0_generation=True,
    run_iterations=True,
    task_description="Continuous learning for binary clinical risk prediction.",
    drift=DriftConfig(
        dropped_cols=("old_feature",),
        added_cols=("new_feature",),
        renamed_cols=(),
        change_note="Example drift note.",
        prev_hl_out_dir=Path("./path/to/previous_hl_output"),
    ),
)

result = run_continuous_learning(
    train_df=train_df,
    test_df=val_df,
    label_col=label_col,
    llm_cfg=llm_cfg,
    continuous_cfg=continuous_cfg,
)

print(result.out_dir)
print(result.final_model_path)
```
