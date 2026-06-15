# Medical Heuristic Learning

Reference: [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/)

[中文文档](./README-CN.md)

## Abstract

Predictive modeling for clinical tabular data is central to clinical decision support and therefore requires not only strong predictive performance but also transparent decision logic. Although deep learning and tree-based ensemble methods can achieve high accuracy, their black-box nature remains a major obstacle to clinical deployment. This challenge is further compounded by common characteristics of medical data, including limited sample sizes, severe class imbalance, and feature evolution arising from changes in diagnostic criteria and clinical documentation. To address these issues, we propose Medical Heuristic Learning (MHL), an instantiation of the learning-beyond-gradients paradigm for clinical tabular prediction. Instead of relying on neural network weight updates, MHL uses a large language model (LLM)-driven workflow that integrates statistical probes, medical knowledge probes, rule synthesis, and code-level iterative refinement to optimize a deterministic and executable decision system. The resulting model is expressed not as opaque parameters, but as versioned pure-Python decision rules that are explicitly interpretable, fully auditable, and clinically grounded. MHL also supports continual learning by starting from previously validated rules and iteratively revising them using updated feature information under data drift or feature evolution. Comprehensive experiments on medical datasets show that MHL achieves performance comparable to state-of-the-art methods while maintaining strong behavior in small-sample and highly imbalanced settings. The results further indicate that this explicit rule update mechanism can help alleviate catastrophic forgetting under feature evolution. Overall, these findings suggest that non-gradient-based heuristic systems offer a transparent and adaptable alternative for high-stakes clinical decision support.

![Medical Heuristic Learning Overview](./supporting_files/fig1.jpg)

## What This Repository Implements

Medical Heuristic Learning (MHL) is a lightweight framework for clinical tabular prediction whose main artifact is executable rule code rather than learned weights. The core `hl/` package combines:

- univariate statistical probes over the training set,
- optional LLM-based medical knowledge probes,
- LLM synthesis of an initial `predict_v0(features: dict) -> int`,
- iterative code-level refinement using sampled training errors and regression warnings,
- export of a stable `predict(features: dict) -> int` entrypoint.

The framework also includes a continuous learning path for feature drift. Instead of rebuilding from scratch, it starts from a previous HL output directory, carries forward prior rule logic, and adapts the rule system under dropped, added, or renamed features.

## Experimental Findings

Based on comprehensive evaluations across multiple medical datasets—including UK Biobank (UKB), Critical Care Information Database (CCID), and Medical Information Mart for Intensive Care (MIMIC)—and compared against representative baselines (Logistic Regression, Decision Tree, XGBoost, LightGBM, MLP, FT-Transformer), MHL demonstrated several key advantages:

- **Robustness in Small-Sample Settings**: MHL consistently outperforms black-box baselines in low-resource regimes (e.g., $n < 100$). When labeled data are scarce, medically informed priors and explicit rule structures compensate for the fragility of purely statistical learners.
- **Resilience to Extreme Class Imbalance**: Under highly skewed distributions (e.g., 50:1 or 1:50), many black-box models collapse into near-one-sided predictions. MHL maintains a workable balance between minority-class detection and majority-class control, guided by explicit error analysis and degradation warnings.
- **Continual Learning without Catastrophic Forgetting**: When the feature space evolves (e.g., transitioning from SIRS to SOFA criteria in sepsis assessment), traditional models suffer severe performance degradation. MHL adapts by explicitly identifying obsolete features and incorporating new signals through code-level rule revisions, avoiding the catastrophic forgetting associated with overwriting hidden parameters.
- **Probe Complementarity**: Ablation studies confirm that combining the statistical probe (for empirical signals) and the medical knowledge probe (for clinical priors and thresholds) yields the most stable and least failure-prone performance.
- **LLM Backend Portability**: MHL remains highly effective across different foundation models (e.g., DeepSeek, Gemini, GPT, Qwen). The structured workflow limits hallucinations and ensures that the synthesized rules are deterministic and usable regardless of the specific backend.

## Core Workflows

### Standard Heuristic Learning

`hl.orchestrator.run_heuristic_learning(...)` executes four stages:

1. Run the univariate statistical probe on `train_df`.
2. Run the knowledge probe if LLM usage is enabled.
3. Generate `predict_v0` into `heuristic_system.py`.
4. Iteratively append `predict_v1`, `predict_v2`, ... and export the best version as `final_heuristic_model.py`.

The orchestrator validates that:

- `label_col` exists in both `train_df` and `test_df`;
- the train/test feature sets are identical after removing the label column.

### Continuous Learning Under Drift

`hl.continuous_learning.run_continuous_learning(...)` reuses the same four-stage structure, but with drift-aware semantics:

1. Load the previous HL output directory described by `DriftConfig.prev_hl_out_dir`.
2. Update univariate probe artifacts under the new feature space.
3. Update the knowledge probe and preserve previous knowledge when possible.
4. Generate a new drift-aware `predict_v0` from the previous final model blueprint and continue iterative refinement.

Continuous learning additionally writes the drift context and previous probe snapshots to disk.

## Repository Layout

- `hl/config.py`
  Standard runtime configuration dataclasses: `LLMConfig` and `RunConfig`.
- `hl/orchestrator/`
  Standard heuristic learning entrypoint and four stage implementations.
- `hl/continuous_learning/`
  Drift-aware configuration, entrypoint, and continuous learning stages.
- `hl/probes/`
  Statistical and knowledge probe implementations.
- `hl/agent/`
  OpenAI-compatible client and prompt templates for standard and continuous workflows.
- `hl/evolution/`
  Error sampling, degradation detection, rule parsing, and syntax validation helpers.
- `hl/metrics.py`
  Metric computation and metric-priority prompt description generation.
- `hl/utils/`
  Thin wrappers for file output and terminal progress logging.
- `example_training.py`
  End-to-end training example on `./data/YHD_bicarbonate.csv`, writing to `./example_out`.
- `example_inference.py`
  Inference example that loads `./example_out/final_heuristic_model.py`.
- `example_continuous_learning.py`
  Continuous learning example that drops the `wbc` feature and writes to `./example_out_continuous_learning`.
- `experiment/`
  Experiment suites separated from the reusable `hl/` core.

## Artifact Contracts

The generated rule files are not arbitrary outputs; downstream scripts assume specific conventions.

### `heuristic_system.py`

- starts with `CURRENT_VERSION = 'v0'` when first created;
- contains versioned rule functions such as `predict_v0`, `predict_v1`, `predict_v2`;
- may contain `ERROR_ANALYSIS_predict_vX` strings for each accepted version;
- is updated incrementally by appending new versions rather than rewriting history.

### `final_heuristic_model.py`

- contains `FINAL_VERSION = "vX"`;
- embeds the accumulated rule code;
- exposes a stable `predict(features: dict) -> int` entrypoint that forwards to `predict_vX`.

## Installation

Requirements:

- Python `>=3.11`
- recommended package manager: `uv`

Install the base dependency set:

```bash
uv sync
```

Install the full dependency set used by examples and experiments:

```bash
uv sync --group dev
```

Current dependency groups in `pyproject.toml` are:

- runtime: `numpy`, `openai`, `pandas`, `scipy`
- dev: `scikit-learn`, `lightgbm`, `torch`, `xgboost`

Practical note: the current core metric implementation in `hl/metrics.py` imports `scikit-learn`, so full end-to-end training and continuous learning runs require the `dev` group as the code exists today.

## Quick Start

Set the API key:

```bash
export DEEPSEEK_API_KEY="your-api-key"
```

Run the standard example:

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

### What The Root Examples Do

- `example_training.py`
  loads `./data/YHD_bicarbonate.csv`, uses `hospital_expire_flag` as the label, takes rows `0:500` as train and `500:1000` as test, and writes to `./example_out`.
- `example_inference.py`
  loads `./example_out/final_heuristic_model.py` and runs inference on the last 5 rows of `./data/YHD_bicarbonate.csv`.
- `example_continuous_learning.py`
  reuses `./example_out`, removes the `wbc` feature to simulate drift, and writes results to `./example_out_continuous_learning`.

## Minimal Usage

### Standard Workflow

```python
from hl.config import LLMConfig, RunConfig
from hl.orchestrator import run_heuristic_learning

run_cfg = RunConfig()
llm_cfg = LLMConfig(
    api_key="your-api-key",  # optional; otherwise read from api_key_env
)

run_heuristic_learning(
    train_df=train_df,
    test_df=test_df,
    label_col="hospital_expire_flag",
    run_cfg=run_cfg,
    llm_cfg=llm_cfg,
)
```

### Continuous Learning Workflow

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

## Outputs

When `RunConfig.output_dir is None`, the standard workflow writes to:

- `./out/<timestamp>/`

When `ContinuousLearningConfig.output_dir is None`, the continuous workflow writes to:

- `./out/<timestamp>_continuous_learning/`

Typical standard HL artifacts are:

- `probe_univariate_results.csv`
- `probe_knowledge.md`
- `heuristic_system.py`
- `evolution_results.txt`
- `iteration_log.json`
- `final_heuristic_model.py`
- `final_comparison.txt`

Continuous learning additionally writes:

- `continuous_learning_context.json`
- `probe_univariate_results_prev.csv`
- `probe_knowledge_prev.md`

## Runtime Behavior

The main `hl/` pipeline prints stage progress to stdout. Typical messages include:

- run start and finish;
- resolved output directory;
- stage boundaries for univariate probe, knowledge probe, v0 generation, and iterations;
- retry failures, accepted versions, and detected regression examples.

This is implemented by `hl/utils/progress.py` and is enabled by default.

## Prompt And Rule Constraints

The current prompt templates in `hl/agent/prompts.py` and `hl/agent/continuous_prompts.py` enforce the following constraints:

- LLM output text must be in English.
- v0 generation and iterative refinement both return strict JSON.
- generated rules must be self-contained pure Python;
- generated rules may use only the Python standard library;
- every `if`/`elif`/`else` branch must include an English comment explaining the medical rationale or design intent;
- iterative updates should be minimal rather than full rewrites.

The code validates JSON structure, Python syntax, and required function names before accepting a proposal.

## API Summary

### `LLMConfig`

Configuration for the OpenAI-compatible backend.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `base_url` | `str` | `"https://api.deepseek.com/v1"` | API base URL. |
| `api_key` | `str \| None` | `None` | Directly provided API key. If present, it takes priority. |
| `api_key_env` | `str` | `"DEEPSEEK_API_KEY"` | Environment variable name used when `api_key` is absent. |
| `model_name` | `str` | `"deepseek-v4-pro"` | Model name passed to the OpenAI-compatible client. |
| `temperature` | `float` | `0.3` | Sampling temperature. |
| `extra_body` | `dict \| None` | `None` | Optional backend-specific extra request payload. |

Key resolution behavior:

- use `api_key` if it is provided;
- otherwise read the environment variable named by `api_key_env`;
- if LLM usage is enabled and neither is available, client construction raises an error.

### `RunConfig`

Configuration for standard heuristic learning.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `output_dir` | `Path \| None` | `None` | If `None`, writes to `./out/<timestamp>/`. |
| `iterations` | `int` | `10` | Maximum number of iterative refinement rounds. |
| `metric_priority` | `tuple[str, ...]` | `("F1", "ACC", "Sensitivity", "Specificity")` | Ordered metric priority for final selection and prompt guidance. |
| `train_baselines` | `bool` | `False` | Reserved field; not used by the main orchestrator. |
| `run_univariate_probe` | `bool` | `True` | Whether to compute the univariate probe. |
| `run_knowledge_probe` | `bool` | `True` | Whether to query the LLM knowledge probe. |
| `run_v0_generation` | `bool` | `True` | Whether to generate `predict_v0` if no heuristic file exists. |
| `run_iterations` | `bool` | `True` | Whether to run iterative refinement. |
| `max_error_samples` | `int` | `100` | Maximum sampled training errors per iteration. |
| `max_error_details` | `int` | `40` | Maximum detailed error samples included in the prompt. |
| `degradation_threshold` | `int` | `10` | Reserved field; currently not enforced directly. |
| `degradation_rate` | `float` | `0.05` | Reserved field; currently not enforced directly. |
| `degradation_max_examples` | `int` | `30` | Maximum regression examples written into the next prompt context. |
| `max_llm_attempts` | `int` | `4` | Maximum retry count for parsing or validation failures. |
| `task_description` | `str` | `""` | Free-form task description inserted into prompts. |
| `enable_auto_patch` | `bool` | `False` | Reserved field for future patch workflows. |
| `max_specificity_drop` | `float` | `1.0` | Reserved field. |
| `max_acc_drop` | `float` | `1.0` | Reserved field. |
| `univariate_top_k` | `int` | `30` | Number of top univariate rows summarized into prompts. |
| `knowledge_top_k` | `int` | `20` | Reserved field; currently not enforced directly. |
| `random_seed` | `int` | `42` | Random seed used for error and degradation sampling. |
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

- validates label and feature-set consistency;
- resolves the output directory;
- runs univariate probe, knowledge probe, v0 generation, and iterations;
- writes iteration logs and artifact files;
- exports `final_heuristic_model.py` using the best recorded version under `metric_priority`.

### `DriftConfig`

Configuration for schema or feature drift.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `dropped_cols` | `tuple[str, ...]` | `()` | Features removed in the new environment. |
| `added_cols` | `tuple[str, ...]` | `()` | New or restored features. |
| `renamed_cols` | `tuple[tuple[str, str], ...]` | `()` | Feature rename mapping `(old_name, new_name)`. |
| `change_note` | `str` | `""` | Free-form natural-language drift description. |
| `prev_hl_out_dir` | `Path \| None` | `None` | Previous HL output directory used as adaptation context. |

### `ContinuousLearningConfig`

Configuration for drift-aware continuous learning.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `drift` | `DriftConfig` | `DriftConfig()` | Drift specification. |
| `output_dir` | `Path \| None` | `None` | If `None`, writes to `./out/<timestamp>_continuous_learning/`. |
| `iterations` | `int` | `10` | Maximum refinement rounds. |
| `metric_priority` | `tuple[str, ...]` | `("F1", "ACC", "Sensitivity", "Specificity")` | Ordered metric priority for final selection and prompt guidance. |
| `run_univariate_probe` | `bool` | `True` | Whether to update the univariate probe. |
| `run_knowledge_probe` | `bool` | `True` | Whether to update the knowledge probe. |
| `run_v0_generation` | `bool` | `True` | Whether to generate a new drift-aware `predict_v0`. |
| `run_iterations` | `bool` | `True` | Whether to run iterative adaptation. |
| `max_error_samples` | `int` | `100` | Maximum sampled training errors per iteration. |
| `max_error_details` | `int` | `40` | Maximum detailed error samples included in the prompt. |
| `degradation_max_examples` | `int` | `30` | Maximum regression examples retained for prompts. |
| `max_llm_attempts` | `int` | `4` | Maximum retry count for LLM output validation. |
| `task_description` | `str` | `""` | Task description injected into prompts. |
| `univariate_top_k` | `int` | `30` | Number of top updated univariate rows summarized. |
| `random_seed` | `int` | `42` | Random seed used by the adaptation loop. |
| `llm_enabled` | `bool` | `True` | Whether to initialize the LLM client. |

### `ContinuousLearningResult`

Return object from `run_continuous_learning(...)`.

| Field | Type | Description |
| --- | --- | --- |
| `out_dir` | `Path` | Output directory for the continuous learning run. |
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

- validates label and feature-set consistency in the new environment;
- writes `continuous_learning_context.json`;
- updates univariate and knowledge probe artifacts under drift;
- builds a new drift-aware `predict_v0` using the previous final model blueprint;
- reuses the same iterative optimization pattern and returns the generated paths.

## Probe Behavior

### Univariate Probe

`hl/probes/univariate.py` currently:

- treats non-binary numeric features as continuous;
- evaluates continuous features with point-biserial correlation and Mann-Whitney U, retaining the better p-value;
- evaluates binary/categorical features with chi-square statistics when applicable;
- records missing rate, summary statistics, and level counts;
- sorts the final table by `p_value` and then `missing_rate`.

### Knowledge Probe

`hl/probes/knowledge.py` asks the LLM to return a Markdown table with exactly these columns:

| Feature | Univariate signal (summary) | Clinical rationale | Suggested threshold | Evidence confidence (high/medium/low) |
| --- | --- | --- | --- | --- |

## Experiments

The `experiment/` directory is separate from the reusable `hl/` core. Current subdirectories are:

- `experiment/ablation/`
  probe and workflow ablation studies.
- `experiment/contrast0/`
  comparisons across LLM backends.
- `experiment/contrast1/`
  comparisons focused on training-set size.
- `experiment/contrast2/`
  comparisons focused on class ratio.
- `experiment/continuous_learning/`
  multi-stage continuous learning experiments and baseline comparisons.

Each experiment subdirectory contains its own README with dataset requirements and commands.

## Notes

- Pass an explicit `output_dir` if you need stable artifact locations.
- If `llm_enabled=False`, LLM-dependent steps can only proceed by reusing artifacts already present on disk.
- If `run_univariate_probe=False` or `run_knowledge_probe=False`, the pipeline will try to reuse the corresponding cached files from the output directory.
- Continuous learning preserves previous probe snapshots in `probe_univariate_results_prev.csv` and `probe_knowledge_prev.md`.
