# Continual-learning experiment

This directory contains the current three-endpoint experiment for MIMIC:

1. `stage1_direct_train1000`
2. `stage2_continual_from_stage1_train40`
3. `stage2_direct_train40`

All endpoints use seeds 36, 40, and 42. The frozen data partition is defined
in `continuous_learning_experiment_common.py`: Stage 1 uses 1,000/500/800
train/validation/test rows and SIRS, while Stage 2 uses 40/500/800 disjoint
rows and SOFA.

## Models

The retained comparison contains MLP, XGBoost, LightGBM, EBM,
FT-Transformer, ResNet, and HL.

Baseline continuation is implemented as follows:

- MLP: continued parameters through epoch-wise `partial_fit`.
- XGBoost: continued booster through `xgb_model`.
- LightGBM: continued booster through `init_model`.
- EBM: Stage 1 raw score supplied as Stage 2 `init_score` at fitting and
  prediction.
- FT-Transformer and ResNet: exact in-memory Stage 1 state transfer, a fresh
  optimizer, and a lower Stage 2 learning rate.

HL Stage 2 direct training calls the standard `hl.orchestrator` from scratch.
It does not read a Stage 1 model or consume Stage 1 training rows. HL exposes
hard labels rather than probabilities, so its `positive_probability` artifact
column contains the hard label cast to 0.0/1.0; this is explicitly recorded in
each run manifest.

## Code layout

| File | Role |
|---|---|
| `continuous_learning_experiment_common.py` | Shared data flow: frozen two-stage partition, result CSV schema, stage/dataset dataclasses |
| `continuous_baseline_v2.py` | Per-model transfer implementations for the six baselines |
| `run_continuous_learning_baselines_v2.py` | Baseline runner (6 models x 3 seeds x 3 endpoints) |
| `run_continuous_learning_hl_v2.py` | HL runner for independently selected Stage 1, Stage 1 -> 2 continual adaptation, and direct Stage 2 endpoints |
| `verify_continuous_baselines_v2.py` | Artifact verification: row keys, frozen test rows, metric recomputation, transfer consistency |

The earlier gated A->B hyperparameter-search runner was removed from this
directory. Its full attempt audit (per-attempt temperatures, priorities and
validation metrics for every trial) is archived at
`experiment/outputs_rerun/continuous_learning_v2_hl_ab_search_20260821_archive.tar.gz`.

## Run

Run or rerun all baselines:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python \
  experiment/continuous_learning/run_continuous_learning_baselines_v2.py \
  --models all --seeds 36 40 42 --resume
```

Run or resume all three HL endpoints:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python \
  experiment/continuous_learning/run_continuous_learning_hl_v2.py \
  --seeds 36 40 42 --resume
```

`--resume` checks each selected endpoint independently. For example, if the
Stage 1 and Stage 1 -> 2 HL rows/artifacts are missing but direct Stage 2 is
complete, the command above runs the first two endpoints and leaves direct
Stage 2 untouched.

Run endpoints separately:

```bash
# Stage 1 from scratch
CUDA_VISIBLE_DEVICES=1 uv run python \
  experiment/continuous_learning/run_continuous_learning_hl_v2.py \
  --stages stage1 --seeds 36 40 42

# Stage 1 -> 2 continual adaptation; requires a complete Stage 1 result and artifacts
CUDA_VISIBLE_DEVICES=1 uv run python \
  experiment/continuous_learning/run_continuous_learning_hl_v2.py \
  --stages stage1-to-stage2 --seeds 36 40 42

# Direct Stage 2 from scratch, without Stage 1 access
CUDA_VISIBLE_DEVICES=1 uv run python \
  experiment/continuous_learning/run_continuous_learning_hl_v2.py \
  --stages stage2 --seeds 36 40 42
```

Multiple stages can be selected with, for example,
`--stages stage1 stage1-to-stage2`. `--stages all` is the default. A standalone
Stage 1 -> 2 invocation validates the successful Stage 1 CSV row, final model,
metrics, predictions, manifest, and probe artifacts before making an LLM call.
New executions write to a unique timestamp leaf below the selected HL endpoint,
so earlier model artifacts are not overwritten. The combined results CSV keeps
all baseline and unselected endpoint rows; only the selected `(seed, HL, stage)`
row is updated after a successful rerun (or recorded error).

The HL command sends derived medical-data context to the configured external
DeepSeek API and therefore requires explicit authorization and
`DEEPSEEK_API_KEY`.

Verify the combined experiment:

```bash
uv run python experiment/continuous_learning/verify_continuous_baselines_v2.py
```

## Outputs

- Combined results: `continuous_baselines_v2_results.csv`
- Models and endpoint artifacts:
  `experiment/outputs_rerun/continuous_learning_v2/`

The complete combined CSV has 63 rows: seven models, three seeds, and three
endpoints. All three HL endpoints now write the current V2 prediction, metrics,
and manifest contracts and can be verified together with the baselines.
