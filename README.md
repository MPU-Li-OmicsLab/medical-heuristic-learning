# Medical Heuristic Learning

Reference: [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/)

[中文文档](./README-CN.md)

## Overview

Medical Heuristic Learning is a lightweight framework for building executable Python rule systems on clinical tabular data.
Instead of treating neural network weights as the main artifact, the project combines statistical probes, LLM-assisted
knowledge extraction, and iterative code editing to produce versioned heuristic functions such as `predict_v0`,
`predict_v1`, and a final exported `predict(features) -> int`.

The repository currently contains:

- A main heuristic learning pipeline in `hl/orchestrator/`
- A continuous learning pipeline for feature drift adaptation in `hl/continuous_learning/`
- End-to-end example scripts in the repository root
- Experiment suites in `experiment/`

## Core Workflow

The default heuristic learning workflow is:

1. Run a univariate statistical probe on the training data.
2. Optionally query an LLM for medical prior knowledge about the features.
3. Generate an initial rule function `predict_v0`.
4. Iteratively refine the rule system using training errors and regression feedback.
5. Export the best rule version as `final_heuristic_model.py`.

The continuous learning workflow extends this process to feature drift settings:

1. Load the previous HL output directory.
2. Update univariate and knowledge probe artifacts under the new feature space.
3. Generate a drift-aware new `predict_v0` using the previous final model as blueprint.
4. Reuse the same iterative optimization loop to adapt the rule system.
5. Export a new final heuristic model for the shifted environment.

## Repository Layout

- `hl/config.py`
  Core runtime configuration objects: `LLMConfig` and `RunConfig`.
- `hl/orchestrator/`
  Main heuristic learning orchestration entrypoint: `run_heuristic_learning(...)`.
- `hl/continuous_learning/`
  Continuous learning entrypoint and drift-aware configuration: `run_continuous_learning(...)`.
- `hl/probes/`
  Univariate statistical probe and knowledge probe implementations.
- `hl/evolution/`
  Iteration utilities, degradation detection, and error analysis helpers.
- `example_training.py`
  End-to-end training example on `./data/YHD_bicarbonate.csv`, writing outputs to `./example_out`.
- `example_inference.py`
  Inference example that loads `./example_out/final_heuristic_model.py`.
- `example_continuous_learning.py`
  Continuous learning example that simulates feature removal and writes outputs to `./example_out_continuous_learning`.
- `experiment/`
  Experiment suites for ablation studies, model comparisons, and continuous learning experiments.

## Installation

Requirements:

- Python `>=3.11`
- Recommended package manager: `uv`

Install runtime dependencies:

```bash
uv sync
```

Install development and experiment dependencies as well:

```bash
uv sync --group dev
```

Current dependency groups:

- Runtime: `numpy`, `openai`, `pandas`, `scipy`
- Dev/experiments: `scikit-learn`, `lightgbm`, `torch`, `xgboost`

## Quick Start

Set the API key through an environment variable:

```bash
export DEEPSEEK_API_KEY="your-api-key"
```

Run the basic training example:

```bash
uv run python example_training.py
```

Run the inference example:

```bash
uv run python example_inference.py
```

Run the continuous learning example:

```bash
uv run python example_continuous_learning.py
```

Minimal direct usage:

```python
from hl.config import LLMConfig, RunConfig
from hl.orchestrator import run_heuristic_learning

run_cfg = RunConfig()
llm_cfg = LLMConfig(
    api_key="your-api-key",  # optional; falls back to api_key_env when omitted
)

run_heuristic_learning(
    train_df=train_df,
    test_df=test_df,
    label_col="hospital_expire_flag",
    run_cfg=run_cfg,
    llm_cfg=llm_cfg,
)
```

Minimal continuous learning usage:

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
        change_note="Describe the feature drift here.",
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

## Example Outputs

The root example scripts currently write to:

- `example_training.py` -> `./example_out`
- `example_continuous_learning.py` -> `./example_out_continuous_learning`

Typical heuristic learning artifacts include:

- `probe_univariate_results.csv`
- `probe_knowledge.md`
- `heuristic_system.py`
- `evolution_results.txt`
- `iteration_log.json`
- `final_heuristic_model.py`
- `final_comparison.txt`

Continuous learning runs may additionally write:

- `continuous_learning_context.json`
- `probe_univariate_results_prev.csv`
- `probe_knowledge_prev.md`

## Runtime Progress Output

The main `hl/` pipeline now emits stage progress to stdout while running.
Typical messages include:

- Start and finish of the overall run
- Output directory resolution
- Univariate probe, knowledge probe, v0 generation, and iteration stage boundaries
- Iteration-level progress, retry failures, accepted versions, and regression warnings

This makes long LLM-driven runs easier to monitor from the terminal.

## API Summary

### `LLMConfig`

Configuration for the OpenAI-compatible LLM backend.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `base_url` | `str` | `"https://api.deepseek.com/v1"` | Base URL for the API service. |
| `api_key` | `str | None` | `None` | Directly provided API key. If set, it takes priority. |
| `api_key_env` | `str` | `"DEEPSEEK_API_KEY"` | Environment variable name used when `api_key` is absent. |
| `model_name` | `str` | `"deepseek-v4-pro"` | Chat model name. |
| `temperature` | `float` | `0.3` | Sampling temperature. |
| `extra_body` | `dict | None` | `None` | Optional extra request payload for backend-specific features. |

Key resolution behavior:

- If `api_key` is provided, it is used directly.
- Otherwise, the framework reads the environment variable named by `api_key_env`.
- If LLM is enabled and no key is available, client initialization raises an error.

### `RunConfig`

Configuration for the standard heuristic learning pipeline.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `output_dir` | `Path | None` | `None` | Output directory. If `None`, the framework creates `./out/<timestamp>/` from the current working directory. |
| `iterations` | `int` | `10` | Maximum number of refinement rounds. |
| `metric_priority` | `tuple[str, ...]` | `("F1", "ACC")` | Ordered metric priority for final model selection. |
| `train_baselines` | `bool` | `False` | Reserved field; not used by the main orchestrator. |
| `run_univariate_probe` | `bool` | `True` | Whether to run the univariate probe. |
| `run_knowledge_probe` | `bool` | `True` | Whether to run the LLM-based knowledge probe. |
| `run_v0_generation` | `bool` | `True` | Whether to generate `predict_v0` when no heuristic file exists. |
| `run_iterations` | `bool` | `True` | Whether to run iterative rule refinement. |
| `max_error_samples` | `int` | `100` | Maximum number of sampled training errors for prompts. |
| `max_error_details` | `int` | `40` | Maximum number of detailed error examples in prompts. |
| `degradation_threshold` | `int` | `10` | Reserved field; not currently enforced in the main path. |
| `degradation_rate` | `float` | `0.05` | Reserved field; not currently enforced in the main path. |
| `degradation_max_examples` | `int` | `30` | Maximum number of regression examples added to the context. |
| `max_llm_attempts` | `int` | `4` | Maximum retry count for parse or validation failures. |
| `task_description` | `str` | `""` | Free-form task description injected into prompts. |
| `enable_auto_patch` | `bool` | `False` | Reserved field for future patch workflows. |
| `max_specificity_drop` | `float` | `1.0` | Reserved acceptance-policy field. |
| `max_acc_drop` | `float` | `1.0` | Reserved acceptance-policy field. |
| `univariate_top_k` | `int` | `30` | Number of top univariate features summarized into prompts. |
| `knowledge_top_k` | `int` | `20` | Reserved field; not currently enforced directly in the main orchestrator. |
| `random_seed` | `int` | `42` | Random seed for sampling error and degradation examples. |
| `llm_enabled` | `bool` | `True` | Whether to initialize the LLM client and run LLM-dependent steps. |

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

Behavior summary:

- Validates the label column and feature-set consistency.
- Resolves the output directory.
- Runs probe -> v0 generation -> iterative optimization.
- Writes all artifacts to disk.
- Exports `final_heuristic_model.py` using the best version under `metric_priority`.

### `DriftConfig`

Configuration describing schema or feature drift for continuous learning.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `dropped_cols` | `tuple[str, ...]` | `()` | Features removed in the new environment. |
| `added_cols` | `tuple[str, ...]` | `()` | New or restored features. |
| `renamed_cols` | `tuple[tuple[str, str], ...]` | `()` | Feature rename mapping `(old_name, new_name)`. |
| `change_note` | `str` | `""` | Natural-language drift description. |
| `prev_hl_out_dir` | `Path | None` | `None` | Previous HL output directory used as adaptation context. |

### `ContinuousLearningConfig`

Configuration for drift-aware continuous learning.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `drift` | `DriftConfig` | `DriftConfig()` | Drift specification. |
| `output_dir` | `Path | None` | `None` | Output directory. If `None`, the framework creates `./out/<timestamp>_continuous_learning/`. |
| `iterations` | `int` | `10` | Maximum refinement rounds. |
| `metric_priority` | `tuple[str, ...]` | `("F1", "ACC")` | Metric priority used for final selection. |
| `run_univariate_probe` | `bool` | `True` | Whether to update the univariate probe. |
| `run_knowledge_probe` | `bool` | `True` | Whether to update the knowledge probe. |
| `run_v0_generation` | `bool` | `True` | Whether to generate a new drift-aware `v0`. |
| `run_iterations` | `bool` | `True` | Whether to run iterative adaptation. |
| `max_error_samples` | `int` | `100` | Maximum sampled training errors per iteration. |
| `max_error_details` | `int` | `40` | Maximum detailed error examples in prompts. |
| `degradation_max_examples` | `int` | `30` | Maximum regression examples retained for prompts. |
| `max_llm_attempts` | `int` | `4` | Maximum retry count for LLM output validation. |
| `task_description` | `str` | `""` | Task description for prompts. |
| `univariate_top_k` | `int` | `30` | Number of top univariate features summarized. |
| `random_seed` | `int` | `42` | Random seed used by the adaptation loop. |
| `llm_enabled` | `bool` | `True` | Whether to initialize the LLM client. |

### `ContinuousLearningResult`

Return object from `run_continuous_learning(...)`.

| Field | Type | Description |
| --- | --- | --- |
| `out_dir` | `Path` | Output directory of the continuous learning run. |
| `heuristic_path` | `Path` | Path to the adapted `heuristic_system.py`. |
| `final_model_path` | `Path` | Path to the exported final model. |

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

Behavior summary:

- Validates the new train/test dataframes and label column.
- Records drift context to disk.
- Updates probe artifacts under the new feature space.
- Builds a new drift-aware `predict_v0` using the previous final model as blueprint.
- Reuses the standard iteration loop for subsequent refinement.
- Returns paths to the generated artifacts.

## Experiments

The `experiment/` directory contains experiment suites that are separate from the reusable `hl/` core:

- `experiment/ablation/`
  Probe ablation studies on `UKB` and `YHD`.
- `experiment/contrast0/`
  HL comparison across different LLM backends.
- `experiment/contrast1/`
  Comparisons focused on training-set size.
- `experiment/contrast2/`
  Comparisons focused on training-set class ratio.
- `experiment/continuous_learning/`
  Two-stage continuous learning experiments with baseline comparisons.

Each subdirectory contains its own README with dataset assumptions, commands, and output structure.

## Notes

- If you want reproducible output locations, pass an explicit `output_dir`.
- If `llm_enabled=False`, LLM-dependent steps cannot generate new rules unless the required artifacts already exist on disk.
- The experiment scripts often require the `dev` dependency group.
