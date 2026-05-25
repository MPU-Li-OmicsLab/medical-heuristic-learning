# Medical Heuristic Learning

参考引用：[Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/)

[English README](./README.md)

## 项目简介

Medical Heuristic Learning 是一个面向临床表格数据的轻量级启发式学习框架。
这个项目的核心产物不是神经网络权重，而是通过统计探针、LLM 辅助知识提取和规则代码迭代编辑，
生成可读、可执行、可版本化的 Python 启发式规则系统，例如 `predict_v0`、`predict_v1`，
以及最终导出的 `predict(features) -> int`。

当前仓库主要包含：

- `hl/orchestrator/` 中的标准启发式学习主流程
- `hl/continuous_learning/` 中的持续学习与特征漂移适配流程
- 仓库根目录下的训练、推理、持续学习示例脚本
- `experiment/` 下的各类实验目录

## 核心工作流

标准启发式学习流程如下：

1. 对训练数据运行单变量统计探针。
2. 可选地调用 LLM 获取特征相关的医学先验知识。
3. 生成初始规则函数 `predict_v0`。
4. 基于训练错误样本和回归退化反馈，迭代更新规则系统。
5. 导出最佳规则版本为 `final_heuristic_model.py`。

持续学习流程是在特征漂移场景下对上述流程的扩展：

1. 读取上一阶段的 HL 输出目录。
2. 在新特征空间下更新单变量探针和知识探针结果。
3. 以旧的最终模型为蓝本，生成新的漂移感知 `predict_v0`。
4. 复用同一套迭代优化逻辑继续适配规则系统。
5. 导出新环境下的最终启发式模型。

## 仓库结构

- `hl/config.py`
  核心配置对象：`LLMConfig` 与 `RunConfig`。
- `hl/orchestrator/`
  标准启发式学习主编排入口：`run_heuristic_learning(...)`。
- `hl/continuous_learning/`
  持续学习主入口及漂移配置：`run_continuous_learning(...)`。
- `hl/probes/`
  单变量统计探针与知识探针实现。
- `hl/evolution/`
  规则迭代、退化检测和错误分析相关工具。
- `example_training.py`
  基于 `./data/YHD_bicarbonate.csv` 的端到端训练示例，输出到 `./example_out`。
- `example_inference.py`
  从 `./example_out/final_heuristic_model.py` 加载模型并执行推理。
- `example_continuous_learning.py`
  持续学习示例，模拟特征删除，并输出到 `./example_out_continuous_learning`。
- `experiment/`
  消融实验、模型对比实验和持续学习实验目录。

## 安装

环境要求：

- Python `>=3.11`
- 推荐包管理器：`uv`

安装运行时依赖：

```bash
uv sync
```

如果你要运行实验脚本或基线模型，请安装开发依赖组：

```bash
uv sync --group dev
```

当前依赖分组如下：

- 运行时依赖：`numpy`、`openai`、`pandas`、`scipy`
- 开发/实验依赖：`scikit-learn`、`lightgbm`、`torch`、`xgboost`

## 快速开始

先设置 API Key：

```bash
export DEEPSEEK_API_KEY="your-api-key"
```

运行基础训练示例：

```bash
uv run python example_training.py
```

运行推理示例：

```bash
uv run python example_inference.py
```

运行持续学习示例：

```bash
uv run python example_continuous_learning.py
```

直接调用标准主流程的最小示例：

```python
from hl.config import LLMConfig, RunConfig
from hl.orchestrator import run_heuristic_learning

run_cfg = RunConfig()
llm_cfg = LLMConfig(
    api_key="your-api-key",  # 可选；不传时回退到 api_key_env
)

run_heuristic_learning(
    train_df=train_df,
    test_df=test_df,
    label_col="hospital_expire_flag",
    run_cfg=run_cfg,
    llm_cfg=llm_cfg,
)
```

直接调用持续学习主流程的最小示例：

```python
from pathlib import Path

from hl.config import LLMConfig
from hl.continuous_learning import ContinuousLearningConfig, DriftConfig, run_continuous_learning

llm_cfg = LLMConfig(api_key="your-api-key")
continuous_cfg = ContinuousLearningConfig(
    drift=DriftConfig(
        dropped_cols=("old_feature",),
        added_cols=("new_feature",),
        renamed_cols=(("old_name", "new_name"),),
        change_note="在这里描述特征漂移。",
        prev_hl_out_dir=Path("./example_out"),
    )
)

result = run_continuous_learning(
    train_df=train_df,
    test_df=test_df,
    label_col="hospital_expire_flag",
    llm_cfg=llm_cfg,
    continuous_cfg=continuous_cfg,
)

print(result.out_dir)
print(result.final_model_path)
```

## 示例输出

仓库根目录下的示例脚本当前默认输出到：

- `example_training.py` -> `./example_out`
- `example_continuous_learning.py` -> `./example_out_continuous_learning`

标准启发式学习通常会产出：

- `probe_univariate_results.csv`
- `probe_knowledge.md`
- `heuristic_system.py`
- `evolution_results.txt`
- `iteration_log.json`
- `final_heuristic_model.py`
- `final_comparison.txt`

持续学习运行还可能额外产出：

- `continuous_learning_context.json`
- `probe_univariate_results_prev.csv`
- `probe_knowledge_prev.md`
- `probe_knowledge_prompt.txt`
- `v0_prompt.txt`
- `v0_error_analysis.txt`
- `v0_attempt_*.txt`

## 运行时进度输出

现在 `hl/` 主干在运行时会把关键阶段进度打印到标准输出。
典型信息包括：

- 整体运行开始与结束
- 输出目录解析结果
- 单变量探针、知识探针、`v0` 生成、迭代优化各阶段的开始与完成
- 每轮迭代中的重试失败原因、接受的新版本、以及检测到的回归样本提示

这使得长时间运行的 LLM 驱动流程更容易在终端中追踪。

## API 概览

### `LLMConfig`

用于配置 OpenAI 兼容的 LLM 后端。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `base_url` | `str` | `"https://api.deepseek.com/v1"` | API 服务基础地址。 |
| `api_key` | `str | None` | `None` | 直接传入的 API Key；如果提供则优先使用。 |
| `api_key_env` | `str` | `"DEEPSEEK_API_KEY"` | 未显式提供 `api_key` 时读取的环境变量名。 |
| `model_name` | `str` | `"deepseek-v4-pro"` | 模型名称。 |
| `temperature` | `float` | `0.3` | 采样温度。 |
| `extra_body` | `dict | None` | `None` | 后端特定能力所需的额外请求体。 |

Key 解析逻辑：

- 如果传入了 `api_key`，则直接使用。
- 否则读取 `api_key_env` 指定的环境变量。
- 如果启用了 LLM 且两种方式都拿不到 Key，则初始化时报错。

### `RunConfig`

用于配置标准启发式学习流程。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `output_dir` | `Path | None` | `None` | 输出目录；为 `None` 时会在当前工作目录下创建 `./out/<时间戳>/`。 |
| `iterations` | `int` | `10` | 最大迭代轮数。 |
| `metric_priority` | `tuple[str, ...]` | `("F1", "ACC")` | 最终版本选择所用的指标优先级。 |
| `train_baselines` | `bool` | `False` | 预留字段；主编排器当前不使用。 |
| `run_univariate_probe` | `bool` | `True` | 是否运行单变量探针。 |
| `run_knowledge_probe` | `bool` | `True` | 是否运行知识探针。 |
| `run_v0_generation` | `bool` | `True` | 当规则文件不存在时，是否生成 `predict_v0`。 |
| `run_iterations` | `bool` | `True` | 是否执行规则迭代优化。 |
| `max_error_samples` | `int` | `100` | 每轮 prompt 中最多采样的错误样本数。 |
| `max_error_details` | `int` | `40` | 每轮 prompt 中最多展开的详细错误样本数。 |
| `degradation_threshold` | `int` | `10` | 预留字段；当前主流程未直接使用。 |
| `degradation_rate` | `float` | `0.05` | 预留字段；当前主流程未直接使用。 |
| `degradation_max_examples` | `int` | `30` | 退化样本最多保留多少条写入上下文。 |
| `max_llm_attempts` | `int` | `4` | LLM 输出解析或校验失败时的最大重试次数。 |
| `task_description` | `str` | `""` | 写入 prompt 的任务自然语言描述。 |
| `enable_auto_patch` | `bool` | `False` | 预留字段，用于未来自动 patch 流程。 |
| `max_specificity_drop` | `float` | `1.0` | 预留的验收约束字段。 |
| `max_acc_drop` | `float` | `1.0` | 预留的验收约束字段。 |
| `univariate_top_k` | `int` | `30` | 进入摘要和 prompt 的 top-k 单变量特征数。 |
| `knowledge_top_k` | `int` | `20` | 预留字段；主编排器当前未直接使用。 |
| `random_seed` | `int` | `42` | 用于采样错误样本和退化样本的随机种子。 |
| `llm_enabled` | `bool` | `True` | 是否初始化 LLM client 并执行依赖 LLM 的步骤。 |

### `run_heuristic_learning`

```python
def run_heuristic_learning(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    run_cfg: RunConfig,
    llm_cfg: LLMConfig,
) -> None:
```

行为概述：

- 校验标签列和训练/测试特征集合是否一致。
- 解析输出目录。
- 按“探针 -> v0 生成 -> 迭代优化”执行主流程。
- 将所有产物写入磁盘。
- 按 `metric_priority` 选择最佳版本并导出 `final_heuristic_model.py`。

### `DriftConfig`

用于描述持续学习中的特征漂移。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `dropped_cols` | `tuple[str, ...]` | `()` | 新环境中被删除的特征。 |
| `added_cols` | `tuple[str, ...]` | `()` | 新增或恢复的特征。 |
| `renamed_cols` | `tuple[tuple[str, str], ...]` | `()` | 重命名映射 `(old_name, new_name)`。 |
| `change_note` | `str` | `""` | 对漂移的自然语言描述。 |
| `prev_hl_out_dir` | `Path | None` | `None` | 之前的 HL 输出目录，用于提供适配上下文。 |

### `ContinuousLearningConfig`

用于配置漂移感知的持续学习流程。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `drift` | `DriftConfig` | `DriftConfig()` | 漂移配置。 |
| `output_dir` | `Path | None` | `None` | 输出目录；为 `None` 时会创建 `./out/<时间戳>_continuous_learning/`。 |
| `iterations` | `int` | `10` | 最大迭代轮数。 |
| `metric_priority` | `tuple[str, ...]` | `("F1", "ACC")` | 最终版本选择所用指标优先级。 |
| `run_univariate_probe` | `bool` | `True` | 是否更新单变量探针。 |
| `run_knowledge_probe` | `bool` | `True` | 是否更新知识探针。 |
| `run_v0_generation` | `bool` | `True` | 是否生成新的漂移感知 `v0`。 |
| `run_iterations` | `bool` | `True` | 是否继续执行适配迭代。 |
| `max_error_samples` | `int` | `100` | 每轮最多采样的错误样本数。 |
| `max_error_details` | `int` | `40` | 每轮最多展开的详细错误样本数。 |
| `degradation_max_examples` | `int` | `30` | 最多保留多少退化样本进入上下文。 |
| `max_llm_attempts` | `int` | `4` | LLM 输出校验失败时的最大重试次数。 |
| `task_description` | `str` | `""` | 写入 prompt 的任务描述。 |
| `univariate_top_k` | `int` | `30` | 进入摘要的 top-k 单变量特征数。 |
| `random_seed` | `int` | `42` | 适配流程中的随机种子。 |
| `llm_enabled` | `bool` | `True` | 是否初始化 LLM client。 |
| `write_prompt_artifacts` | `bool` | `True` | 当前 dataclass 中存在的配置字段。 |

### `ContinuousLearningResult`

`run_continuous_learning(...)` 的返回对象。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `out_dir` | `Path` | 持续学习本次运行的输出目录。 |
| `heuristic_path` | `Path` | 适配后的 `heuristic_system.py` 路径。 |
| `final_model_path` | `Path` | 导出的最终模型路径。 |

### `run_continuous_learning`

```python
def run_continuous_learning(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    llm_cfg: LLMConfig,
    continuous_cfg: ContinuousLearningConfig,
) -> ContinuousLearningResult:
```

行为概述：

- 校验新的训练/测试数据和标签列。
- 将漂移上下文写入输出目录。
- 在新特征空间下更新 probe 结果。
- 以旧最终模型为蓝本生成新的漂移感知 `predict_v0`。
- 复用标准迭代逻辑继续适配。
- 返回本次运行生成的关键路径。

## 实验目录

`experiment/` 下的脚本属于实验层，不属于可复用的 `hl/` 主干：

- `experiment/ablation/`
  `UKB` 和 `YHD` 上的探针消融实验。
- `experiment/contrast0/`
  不同 LLM 后端下的 HL 对比实验。
- `experiment/contrast1/`
  以训练集规模为重点的对比实验。
- `experiment/contrast2/`
  以训练集类别比例为重点的对比实验。
- `experiment/continuous_learning/`
  两阶段持续学习与 baseline 对比实验。

每个子目录下都有独立 README，说明数据要求、运行命令与输出结构。

## 说明

- 如果你希望输出目录稳定可复现，请显式传入 `output_dir`。
- 当 `llm_enabled=False` 时，依赖 LLM 的步骤无法新生成规则，除非所需中间产物已经存在于磁盘上。
- 大多数实验脚本需要 `dev` 依赖组。 
