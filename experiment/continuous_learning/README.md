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
| `run_continuous_learning_hl_v2.py` | HL Stage 2 direct-training runner; guards the previously completed HL rows |
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

Run or resume direct Stage 2 HL:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python \
  experiment/continuous_learning/run_continuous_learning_hl_v2.py \
  --seeds 36 40 42 --resume
```

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

The combined CSV has 63 rows: seven models, three seeds, and three endpoints.
The six Stage 1/continual HL rows were migrated from the previously completed
HL experiment and aligned to the V2 endpoint/status names. Their original
artifact paths belong to the former workspace and are retained only as
provenance; the three new direct Stage 2 HL artifacts use the current V2
layout and are fully verified.
