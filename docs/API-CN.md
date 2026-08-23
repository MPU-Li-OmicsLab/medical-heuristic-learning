# Medical Heuristic Learning API 文档

[返回中文 README](../README-CN.md) · [English API Documentation](./API.md)

本文档对应当前 `medical-heuristic-learning` 的 `src/hl` 实现。

## 公共入口

以下名称可直接从 `hl` 导入：

```python
from hl import (
    __version__,
    BatchPredictFunction,
    ContinuousLearningConfig,
    ContinuousLearningResult,
    DriftConfig,
    LLMConfig,
    PredictFunction,
    RunConfig,
    load_batch_model,
    load_model,
    run_continuous_learning,
    run_heuristic_learning,
)
```

## 配置对象

所有配置对象均为冻结的 dataclass；创建后不能原地修改字段。需要调整配置时，请构造新对象或使用 `dataclasses.replace(...)`。

### `LLMConfig`

配置 OpenAI 兼容的 LLM 后端。

```python
LLMConfig(
    base_url="https://api.deepseek.com/v1",
    api_key=None,
    api_key_env="DEEPSEEK_API_KEY",
    model_name="deepseek-v4-pro",
    temperature=0.3,
    extra_body=None,
    thinking_mode=None,
    thinking_strength=None,
)
```

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `base_url` | `str` | `"https://api.deepseek.com/v1"` | OpenAI 兼容 API 的基础地址。 |
| `api_key` | `str \| None` | `None` | 直接提供的 API Key；非空时优先于环境变量。 |
| `api_key_env` | `str` | `"DEEPSEEK_API_KEY"` | `api_key` 为空时读取的环境变量名。 |
| `model_name` | `str` | `"deepseek-v4-pro"` | 发送给后端的模型名称。 |
| `temperature` | `float` | `0.3` | 采样温度。 |
| `extra_body` | `dict \| None` | `None` | 后端专用的附加请求体。 |
| `thinking_mode` | `bool \| None` | `None` | `True`/`False` 显式启用或禁用思考模式；`None` 不发送该开关。 |
| `thinking_strength` | `str \| None` | `None` | 可选值：`low`、`medium`、`high`、`xhigh`、`max`。设置强度且模式为 `None` 时会自动启用思考模式。 |

补充规则：

- `extra_body` 中已有的 `thinking` 键优先于 `thinking_mode`。
- `thinking_mode=False` 不能与非空 `thinking_strength` 同时使用，否则抛出 `ValueError`。
- 启用 LLM 时，如果 `api_key` 和 `api_key_env` 对应的环境变量均为空，客户端初始化会抛出 `RuntimeError`。

### `RunConfig`

控制标准 MHL 流程。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `output_dir` | `Path \| None` | `None` | 输出目录；为空时使用 `./out/{时间戳}/`。 |
| `iterations` | `int` | `10` | `v0` 之后最多进行的规则迭代轮数；负数按 `0` 处理。 |
| `metric_priority` | `tuple[str, ...]` | `("F1", "ACC", "Sensitivity", "Specificity")` | 最佳版本的字典序指标优先级。 |
| `run_univariate_probe` | `bool` | `True` | 是否重新运行统计探针；关闭时尝试复用输出目录中的缓存。 |
| `run_knowledge_probe` | `bool` | `True` | 是否调用 LLM 运行知识探针；关闭时尝试复用缓存。 |
| `run_v0_generation` | `bool` | `True` | `heuristic_system.py` 不存在时是否生成 `v0`。已有文件始终优先复用。 |
| `run_iterations` | `bool` | `True` | 是否执行 `v1...vN` 迭代；关闭时仍评估并导出 `v0`。 |
| `max_error_samples` | `int` | `100` | 每轮从训练集误分类病例中最多采样的数量。 |
| `max_error_details` | `int` | `40` | 写入 LLM 错误报告的最多病例详情数。 |
| `degradation_max_examples` | `int` | `30` | 每轮最多保留的退化病例示例数。 |
| `max_llm_attempts` | `int` | `4` | 每次规则生成或修订的最大校验重试次数；小于 `1` 时仍至少尝试一次。 |
| `task_description` | `str` | `""` | 提供给知识探针、规则生成和规则迭代的任务说明。 |
| `univariate_top_k` | `int` | `30` | 传递给下游规则生成和错误报告的统计探针特征数。 |
| `random_seed` | `int` | `42` | 错误病例与退化病例采样的随机种子。 |
| `llm_enabled` | `bool` | `True` | 是否创建 LLM 客户端；关闭后只能复用已有规则或探针产物。 |
| `train_baselines` | `bool` | `False` | 预留字段；当前标准编排器未使用。 |
| `degradation_threshold` | `int` | `10` | 预留字段；当前退化检测逻辑未读取。 |
| `degradation_rate` | `float` | `0.05` | 预留字段；当前退化检测逻辑未读取。 |
| `enable_auto_patch` | `bool` | `False` | 预留字段；当前标准编排器未实现自动补丁路径。 |
| `max_specificity_drop` | `float` | `1.0` | 预留字段；当前候选接受逻辑未读取。 |
| `max_acc_drop` | `float` | `1.0` | 预留字段；当前候选接受逻辑未读取。 |
| `knowledge_top_k` | `int` | `20` | 预留字段；当前知识探针调用未截断特征列表。 |

### `DriftConfig`

描述持续学习阶段的特征空间变化。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `dropped_cols` | `tuple[str, ...]` | `()` | 新阶段已删除的特征。 |
| `added_cols` | `tuple[str, ...]` | `()` | 新阶段新增的特征；增量知识探针仅查询这些特征。 |
| `renamed_cols` | `tuple[tuple[str, str], ...]` | `()` | `(旧名称, 新名称)` 映射。 |
| `change_note` | `str` | `""` | 提供给漂移感知规则生成的变化说明。 |
| `prev_hl_out_dir` | `Path \| None` | `None` | 上一阶段 MHL 输出目录，用于读取最终模型和探针产物。 |

### `ContinuousLearningConfig`

控制持续学习流程。

```python
ContinuousLearningConfig(
    drift=DriftConfig(),
    output_dir=None,
    iterations=10,
    metric_priority=("F1", "ACC", "Sensitivity", "Specificity"),
    run_univariate_probe=True,
    run_knowledge_probe=True,
    run_v0_generation=True,
    run_iterations=True,
    max_error_samples=100,
    max_error_details=40,
    degradation_max_examples=30,
    max_llm_attempts=4,
    task_description="",
    univariate_top_k=30,
    random_seed=42,
    llm_enabled=True,
)
```

除 `drift` 外，各字段语义与 `RunConfig` 中的同名字段一致。`output_dir=None` 时，输出目录为 `./out/{时间戳}_continuous_learning/`。

## 运行入口

### `run_heuristic_learning`

```python
def run_heuristic_learning(
    train_df: pandas.DataFrame,
    val_df: pandas.DataFrame,
    label_col: str,
    run_cfg: RunConfig,
    llm_cfg: LLMConfig,
) -> RunResult:
    ...
```

执行统计探针、医学知识探针、`v0` 生成和规则迭代，并导出最佳版本。

参数约束：

- `label_col` 必须同时存在于 `train_df` 与 `val_df`；否则抛出 `ValueError`。
- 移除标签后，两者的特征名称集合必须一致；否则抛出 `ValueError`。
- 输入索引会在内部重置；特征列顺序可以不同，但建议保持一致以便审计。
- 若关闭某阶段，应保证输出目录中已有其下游所需缓存。尤其当 `heuristic_system.py` 不存在且无法生成 `v0` 时会抛出 `RuntimeError`。

```python
from pathlib import Path

from hl import LLMConfig, RunConfig, run_heuristic_learning

result = run_heuristic_learning(
    train_df=train_df,
    val_df=val_df,
    label_col="target",
    run_cfg=RunConfig(output_dir=Path("./out/example"), iterations=5),
    llm_cfg=LLMConfig(api_key_env="DEEPSEEK_API_KEY"),
)
```

### `run_continuous_learning`

```python
def run_continuous_learning(
    *,
    train_df: pandas.DataFrame,
    val_df: pandas.DataFrame,
    label_col: str,
    llm_cfg: LLMConfig,
    continuous_cfg: ContinuousLearningConfig,
) -> ContinuousLearningResult:
    ...
```

该函数的参数均为仅限关键字参数。它会：

1. 写入 `continuous_learning_context.json`；
2. 读取并快照上一阶段探针产物；
3. 根据 `DriftConfig` 过滤删除项、应用重命名并补充新增特征；
4. 以前一阶段最终模型为蓝本生成新的 `v0`；
5. 执行与标准流程相同的版本迭代与最佳版本导出。

`prev_hl_out_dir` 为空或相关文件缺失时，流程可以退化为使用当前数据和空的历史上下文；若目标是继承旧模型，应显式提供一个完整、可信的上一阶段输出目录。

## 模型加载

### `load_model`

```python
def load_model(model_path: str | Path) -> PredictFunction:
    ...
```

加载导出文件中的 `predict(features)`，返回单行预测函数：

```python
predict_one = load_model("./out/example/final_heuristic_model.py")
label = predict_one({"age": 65, "marker": 1})
```

异常：

- 文件不存在：`FileNotFoundError`；
- 无法创建或执行模块：`RuntimeError`；
- 文件中没有可调用的 `predict`：`RuntimeError`。

### `load_batch_model`

```python
def load_batch_model(model_path: str | Path) -> BatchPredictFunction:
    ...
```

模型文件只加载一次，返回接收 `pandas.DataFrame` 的批量函数。结果为按输入行顺序排列的 `list[int]`。

```python
predict_batch = load_batch_model("./out/example/final_heuristic_model.py")
labels = predict_batch(feature_dataframe)
```

批量输入不是 `DataFrame` 时抛出 `TypeError`；列名不唯一时抛出 `ValueError`。函数不会自动剔除标签、补齐缺失列或执行特征预处理。

> [!WARNING]
> `load_model` 和 `load_batch_model` 会执行模型文件中的 Python 代码，只能加载可信来源的产物。

## 返回类型与类型别名

### `RunResult`

`RunResult` 的唯一公共定义位于：

```python
from hl.result import RunResult
```

标准流程返回的冻结 dataclass：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `out_dir` | `Path` | 本次运行的输出目录。 |
| `heuristic_path` | `Path` | 版本化规则文件 `heuristic_system.py`。 |
| `final_model_path` | `Path` | 最终导出文件 `final_heuristic_model.py`。 |

### `ContinuousLearningResult`

继承 `RunResult`，当前不增加额外字段。

### 预测函数类型

```python
PredictFunction = Callable[[dict[str, Any]], int]
BatchPredictFunction = Callable[[pandas.DataFrame], list[int]]
```

### `__version__`

`hl.__version__` 从已安装分发包 `medical-heuristic-learning` 读取版本；源码未安装为分发包时回退为 `"0+unknown"`。

## 产物契约

### `heuristic_system.py`

- 初始包含 `CURRENT_VERSION = 'v0'` 与 `predict_v0(features: dict) -> int`；
- 接受的后续版本以 `predict_v1`、`predict_v2` 等名称追加，不覆盖旧代码；
- 可包含 `ERROR_ANALYSIS_predict_vX`，记录相应版本的修改依据。

### `final_heuristic_model.py`

- 包含 `FINAL_VERSION = "vX"`；
- 内嵌累计规则代码；
- 暴露稳定的 `predict(features: dict) -> int`，并转发到最佳 `predict_vX`。

最佳版本按 `metric_priority` 的顺序进行字典序比较。例如 `("F1", "ACC")` 会先选择 F1 较高的版本，仅在 F1 相同时比较 ACC。

## 日志

库使用名为 `hl` 的标准 Python logger，但不主动安装 handler。调用方可自行启用：

```python
import logging

logging.basicConfig(level=logging.INFO)
```

## 完整示例

- [训练](../example_training.py)
- [单行与批量推理](../example_inference.py)
- [持续学习](../example_continuous_learning.py)
