# HL Medical

[English README](./README.md)

## 项目简介

HL Medical 是一个面向临床表格数据的轻量级启发式学习框架。
这个项目的主要产物不是神经网络参数，而是通过统计探针、LLM 辅助知识提取和规则迭代编辑，
逐步构建出来的 Python 启发式规则系统，适用于院内死亡预测这类二分类医学任务。

默认工作流如下：

1. 对训练数据运行单变量统计探针。
2. 可选地调用 LLM 获取特征相关的医学先验知识。
3. 生成初始规则函数 `predict_v0`。
4. 基于训练错误样本和回归退化反馈，迭代更新规则系统。
5. 导出最佳规则版本，作为最终预测入口。

仓库当前提供了一个医学表格训练示例 `example_training.py`，主要编排入口是 `run_heuristic_learning(...)`。

## 安装

环境要求：

- Python `>=3.11`
- 推荐包管理器：`uv`

安装依赖：

```bash
uv sync
```

运行时依赖包括：

- `numpy`
- `openai`
- `pandas`
- `scipy`

开发依赖包括：

- `scikit-learn`

## 快速开始

示例调用：

```python
from hl.config import LLMConfig, RunConfig
from hl.orchestrator import run_heuristic_learning

run_cfg = RunConfig()
llm_cfg = LLMConfig(
    api_key="your-api-key",  # 可选；如果不传，会回退到 api_key_env
)

run_heuristic_learning(
    train_df=train_df,
    test_df=test_df,
    label_col="hospital_expire_flag",
    run_cfg=run_cfg,
    llm_cfg=llm_cfg,
)
```

你也可以不直接传 key，而是使用环境变量：

```bash
export DEEPSEEK_API_KEY="your-api-key"
uv run python example_training.py
```

## 示例文件

- `example_training.py`：使用 `./data/YHD_bicarbonate.csv` 运行完整训练流程，并将生成产物写入 `./example_out`。
- `example_inference.py`：从 `./example_out/final_heuristic_model.py` 加载已训练模型，读取 `./data/YHD_bicarbonate.csv` 的最后 5 行，并输出预测结果。
- `example_out`：训练脚本生成的示例输出目录，包含 `heuristic_system.py`、`final_heuristic_model.py` 等文件。

## API 文档

### `LLMConfig`

用于配置 LLM 后端。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `base_url` | `str` | `"https://api.deepseek.com/v1"` | OpenAI 兼容 API 服务的基础地址。 |
| `api_key` | `str | None` | `None` | 直接传入的 API key。只要它不是 `None` 且不是空字符串，就优先使用它，不再读取环境变量。 |
| `api_key_env` | `str` | `"DEEPSEEK_API_KEY"` | 当 `api_key` 未提供时，用于读取 key 的环境变量名。 |
| `model_name` | `str` | `"deepseek-v4-pro"` | 调用 OpenAI 兼容 chat completion API 时传入的模型名。 |
| `temperature` | `float` | `0.3` | LLM 采样温度。 |

Key 解析逻辑：

- 如果传入了 `api_key`，框架直接使用它。
- 否则，框架读取 `api_key_env` 指定的环境变量。
- 如果两种方式都没有拿到 key，且 `llm_enabled=True`，初始化会报错。

### `RunConfig`

用于配置启发式学习运行流程。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `output_dir` | `Path | None` | `None` | 所有输出产物的目录。当为 `None` 时，框架会在当前工作目录下自动创建 `./out/<时间戳>/`。 |
| `iterations` | `int` | `10` | 规则迭代优化的最大轮数。 |
| `metric_priority` | `tuple[str, ...]` | `("F1", "ACC")` | 选择最终最佳版本时使用的指标优先级。 |
| `train_baselines` | `bool` | `False` | 预留的 baseline 训练开关；当前主编排流程没有实际使用这个字段。 |
| `run_univariate_probe` | `bool` | `True` | 是否运行单变量统计探针。 |
| `run_knowledge_probe` | `bool` | `True` | 是否运行基于 LLM 的医学知识探针。 |
| `run_v0_generation` | `bool` | `True` | 当启发式规则文件不存在时，是否生成初始规则 `predict_v0`。 |
| `run_iterations` | `bool` | `True` | 是否在 `v0` 基础上继续做迭代优化。 |
| `max_error_samples` | `int` | `100` | 每轮迭代中，最多采样多少个训练集错分样本写入 prompt。 |
| `max_error_details` | `int` | `40` | 每轮 prompt 中最多展开多少个详细错误样本。 |
| `degradation_threshold` | `int` | `10` | 预留的退化阈值字段；当前主编排逻辑没有直接使用。 |
| `degradation_rate` | `float` | `0.05` | 预留的退化比例字段；当前主编排逻辑没有直接使用。 |
| `degradation_max_examples` | `int` | `30` | 检测到退化后，最多采样多少个回归样本写入下一轮上下文。 |
| `max_llm_attempts` | `int` | `4` | 每轮迭代中 LLM 输出解析或校验失败时的最大重试次数。 |
| `task_description` | `str` | `""` | 任务的自然语言描述，会传入 prompt。 |
| `enable_auto_patch` | `bool` | `False` | 预留的自动 patch 开关。 |
| `max_specificity_drop` | `float` | `1.0` | 预留的约束字段，用于未来更严格的验收策略。 |
| `max_acc_drop` | `float` | `1.0` | 预留的约束字段，用于未来更严格的验收策略。 |
| `univariate_top_k` | `int` | `30` | 单变量探针中会进入摘要和报告的 top-k 特征数。 |
| `knowledge_top_k` | `int` | `20` | 预留的知识探针相关字段；当前主入口没有直接使用。 |
| `random_seed` | `int` | `42` | 用于采样错误样本和退化样本的随机种子。 |
| `llm_enabled` | `bool` | `True` | 是否初始化 LLM client 并执行依赖 LLM 的步骤。 |

### `run_heuristic_learning`

接口签名：

```python
def run_heuristic_learning(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    run_cfg: RunConfig,
    llm_cfg: LLMConfig,
) -> None:
```

参数说明：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `train_df` | `pd.DataFrame` | 训练集数据，包含特征列和标签列。 |
| `test_df` | `pd.DataFrame` | 测试集数据，必须包含与训练集一致的特征列和标签列。 |
| `label_col` | `str` | 标签列名称，必须同时存在于训练集和测试集中。 |
| `run_cfg` | `RunConfig` | 运行流程配置。 |
| `llm_cfg` | `LLMConfig` | LLM 后端配置。 |

接口行为：

- 重置训练集和测试集索引。
- 校验 `label_col` 是否存在于两个数据集中。
- 校验 `train_df` 和 `test_df` 的特征列集合是否一致。
- 解析 `output_dir`；当其为 `None` 时自动创建 `./out/<时间戳>/`。
- 运行单变量探针和知识探针。
- 在需要时生成包含 `predict_v0` 的 `heuristic_system.py`。
- 执行规则迭代更新与评估。
- 写入日志和总结产物。
- 按 `metric_priority` 选择最佳版本。
- 导出统一入口 `predict(features) -> int` 的 `final_heuristic_model.py`。

返回值：

- 返回 `None`。
- 所有结果都会写入解析后的输出目录中。

可能抛出的错误：

- 如果 `label_col` 缺失，或训练集与测试集的特征列不一致，会抛出 `ValueError`。
- 如果没有产生任何版本记录，或在需要 LLM 时没有有效 API key，会抛出 `RuntimeError`。

## 输出文件

流程运行后，通常会在 `output_dir` 下生成以下文件：

- `probe_univariate_results.csv`
- `probe_knowledge.md`
- `heuristic_system.py`
- `evolution_results.txt`
- `iteration_log.json`
- `final_heuristic_model.py`
- `final_comparison.txt`

## 说明

- 当 `output_dir=None` 时，系统会自动在 `./out/` 下创建时间戳目录。
- 如果你想基于已有实验结果继续迭代，应显式传入已有的 `output_dir`。
- 当 `llm_enabled=False` 时，依赖 LLM 的步骤无法新生成规则，除非所需中间产物已经存在。
