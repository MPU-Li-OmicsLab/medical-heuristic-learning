# Medical Heuristic Learning API Documentation

[Back to the English README](../README.md) · [中文 API 文档](./API-CN.md)

This document describes the current `src/hl` implementation of `medical-heuristic-learning`.

## Public API

The following names can be imported directly from `hl`:

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

## Configuration Objects

All configuration objects are frozen dataclasses and therefore cannot be mutated after construction. To modify a configuration, construct a new instance or use `dataclasses.replace(...)`.

### `LLMConfig`

Configures an OpenAI-compatible LLM backend.

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

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `base_url` | `str` | `"https://api.deepseek.com/v1"` | Base URL of the OpenAI-compatible API. |
| `api_key` | `str \| None` | `None` | Explicit API key. A nonempty value takes precedence over the environment variable. |
| `api_key_env` | `str` | `"DEEPSEEK_API_KEY"` | Name of the environment variable consulted when `api_key` is empty. |
| `model_name` | `str` | `"deepseek-v4-pro"` | Model identifier sent to the backend. |
| `temperature` | `float` | `0.3` | Sampling temperature. |
| `extra_body` | `dict \| None` | `None` | Optional backend-specific request payload. |
| `thinking_mode` | `bool \| None` | `None` | `True` or `False` explicitly enables or disables thinking mode; `None` omits the switch. |
| `thinking_strength` | `str \| None` | `None` | One of `low`, `medium`, `high`, `xhigh`, or `max`. Specifying a strength while `thinking_mode=None` implicitly enables thinking mode. |

Additional rules:

- An existing `thinking` key in `extra_body` takes precedence over `thinking_mode`.
- Combining `thinking_mode=False` with a nonempty `thinking_strength` raises `ValueError`.
- When LLM use is enabled, client initialization raises `RuntimeError` if neither `api_key` nor the environment variable specified by `api_key_env` provides a key.

### `RunConfig`

Controls the standard MHL workflow.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `output_dir` | `Path \| None` | `None` | Output directory. If `None`, the workflow uses `./out/{timestamp}/`. |
| `iterations` | `int` | `10` | Maximum number of rule-refinement rounds after `v0`; negative values are treated as `0`. |
| `metric_priority` | `tuple[str, ...]` | `("F1", "ACC", "Sensitivity", "Specificity")` | Lexicographically ordered metric priority used to select the best version. |
| `run_univariate_probe` | `bool` | `True` | Whether to recompute the statistical probe. If disabled, the workflow attempts to reuse a cached artifact from the output directory. |
| `run_knowledge_probe` | `bool` | `True` | Whether to query the LLM-based medical knowledge probe. If disabled, the workflow attempts to reuse a cached artifact. |
| `run_v0_generation` | `bool` | `True` | Whether to generate `v0` when `heuristic_system.py` does not exist. An existing file is always reused. |
| `run_iterations` | `bool` | `True` | Whether to generate `v1...vN`. If disabled, `v0` is still evaluated and exported. |
| `max_error_samples` | `int` | `100` | Maximum number of misclassified training cases sampled per iteration. |
| `max_error_details` | `int` | `40` | Maximum number of case-level details included in the LLM error report. |
| `degradation_max_examples` | `int` | `30` | Maximum number of degradation examples retained per iteration. |
| `max_llm_attempts` | `int` | `4` | Maximum number of validation retries for each rule-generation or revision request; at least one request is made even when the value is less than `1`. |
| `task_description` | `str` | `""` | Task description supplied to the knowledge probe, initial rule generation, and rule refinement. |
| `univariate_top_k` | `int` | `30` | Number of top-ranked statistical-probe features passed to downstream rule generation and error reporting. |
| `random_seed` | `int` | `42` | Random seed used to sample error and degradation cases. |
| `llm_enabled` | `bool` | `True` | Whether to initialize the LLM client. If disabled, the workflow can only reuse existing rule and probe artifacts. |
| `train_baselines` | `bool` | `False` | Reserved; the current standard orchestrator does not use this field. |
| `degradation_threshold` | `int` | `10` | Reserved; the current degradation-detection logic does not read this field. |
| `degradation_rate` | `float` | `0.05` | Reserved; the current degradation-detection logic does not read this field. |
| `enable_auto_patch` | `bool` | `False` | Reserved; the current standard orchestrator does not implement an automatic patching path. |
| `max_specificity_drop` | `float` | `1.0` | Reserved; the current candidate-acceptance logic does not read this field. |
| `max_acc_drop` | `float` | `1.0` | Reserved; the current candidate-acceptance logic does not read this field. |
| `knowledge_top_k` | `int` | `20` | Reserved; the current knowledge-probe call does not truncate the feature list. |

### `DriftConfig`

Describes changes in the feature space at a continual-learning stage.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `dropped_cols` | `tuple[str, ...]` | `()` | Features removed in the new stage. |
| `added_cols` | `tuple[str, ...]` | `()` | Features introduced in the new stage; the incremental knowledge probe queries only these features. |
| `renamed_cols` | `tuple[tuple[str, str], ...]` | `()` | `(old_name, new_name)` mappings. |
| `change_note` | `str` | `""` | Description of the feature-space change supplied to drift-aware rule generation. |
| `prev_hl_out_dir` | `Path \| None` | `None` | Output directory from the preceding MHL stage, used to retrieve the final model and probe artifacts. |

### `ContinuousLearningConfig`

Controls the continual-learning workflow.

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

Except for `drift`, fields with the same names have the semantics documented for `RunConfig`. If `output_dir=None`, the workflow uses `./out/{timestamp}_continuous_learning/`.

## Workflow Entry Points

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

Executes the statistical probe, medical knowledge probe, `v0` generation, and rule-refinement stages, and then exports the selected version.

Input constraints:

- `label_col` must be present in both `train_df` and `val_df`; otherwise, the function raises `ValueError`.
- After removal of the label column, the two data frames must contain identical sets of feature names; otherwise, the function raises `ValueError`.
- Input indices are reset internally. Feature-column order may differ, although consistent ordering is recommended for auditability.
- If a stage is disabled, its downstream dependencies must already be available in the output directory. In particular, `RuntimeError` is raised when `heuristic_system.py` is absent and `v0` cannot be generated.

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

All arguments are keyword-only. The function performs the following operations:

1. Writes `continuous_learning_context.json`.
2. Loads and snapshots the probe artifacts from the preceding stage.
3. Uses `DriftConfig` to remove dropped features, apply renamings, and incorporate newly introduced features.
4. Generates a new `v0` using the preceding stage's final model as an explicit blueprint.
5. Applies the same version-refinement and best-version export procedure used by the standard workflow.

If `prev_hl_out_dir` is `None` or the corresponding artifacts are unavailable, the workflow can proceed using the current data and an empty historical context. To inherit a previous model, provide an explicit, complete, and trusted output directory from the preceding stage.

## Model Loading

### `load_model`

```python
def load_model(model_path: str | Path) -> PredictFunction:
    ...
```

Loads `predict(features)` from an exported artifact and returns a single-row prediction function:

```python
predict_one = load_model("./out/example/final_heuristic_model.py")
label = predict_one({"age": 65, "marker": 1})
```

Exceptions:

- If the file does not exist: `FileNotFoundError`.
- If the module cannot be created or executed: `RuntimeError`.
- If the file does not define a callable `predict`: `RuntimeError`.

### `load_batch_model`

```python
def load_batch_model(model_path: str | Path) -> BatchPredictFunction:
    ...
```

Loads the model artifact once and returns a batch predictor that accepts a `pandas.DataFrame`. The result is a `list[int]` whose order matches the input rows.

```python
predict_batch = load_batch_model("./out/example/final_heuristic_model.py")
labels = predict_batch(feature_dataframe)
```

The batch predictor raises `TypeError` when its input is not a `DataFrame` and `ValueError` when column names are not unique. It does not remove labels, supply missing columns, or perform feature preprocessing.

> [!WARNING]
> `load_model` and `load_batch_model` execute the Python code contained in the model artifact. Load artifacts only from trusted sources.

## Return Types and Type Aliases

### `RunResult`

The sole public definition of `RunResult` is located at:

```python
from hl.result import RunResult
```

This frozen dataclass is returned by the standard workflow:

| Field | Type | Description |
| --- | --- | --- |
| `out_dir` | `Path` | Output directory for the run. |
| `heuristic_path` | `Path` | Path to the versioned `heuristic_system.py` artifact. |
| `final_model_path` | `Path` | Path to the exported `final_heuristic_model.py` artifact. |

### `ContinuousLearningResult`

Inherits from `RunResult` and currently introduces no additional fields.

### Prediction Function Types

```python
PredictFunction = Callable[[dict[str, Any]], int]
BatchPredictFunction = Callable[[pandas.DataFrame], list[int]]
```

### `__version__`

`hl.__version__` is obtained from the installed `medical-heuristic-learning` distribution. When the source tree is not installed as a distribution, it falls back to `"0+unknown"`.

## Artifact Contracts

### `heuristic_system.py`

- Initially contains `CURRENT_VERSION = 'v0'` and `predict_v0(features: dict) -> int`.
- Accepted subsequent versions are appended as `predict_v1`, `predict_v2`, and so forth; prior code is not overwritten.
- May contain `ERROR_ANALYSIS_predict_vX` values that record the rationale for the corresponding versions.

### `final_heuristic_model.py`

- Contains `FINAL_VERSION = "vX"`.
- Embeds the accumulated rule code.
- Exposes the stable `predict(features: dict) -> int` entry point, which delegates to the selected `predict_vX`.

The best version is selected by lexicographically comparing metrics in the order specified by `metric_priority`. For example, `("F1", "ACC")` prioritizes the higher-F1 version and compares ACC only when F1 is tied.

## Logging

The library uses the standard Python logger named `hl` but does not install a handler. Applications can enable logging explicitly:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

## Complete Examples

- [Training](../example_training.py)
- [Single-row and batch inference](../example_inference.py)
- [Continual learning](../example_continuous_learning.py)
