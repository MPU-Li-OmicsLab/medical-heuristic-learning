"""对比模型名称与固定超参数。

本文件是三个实验共享的模型配置单一来源。类别比例由实验脚本在训练前
构造，本文件中的模型不得启用 class_weight、sample_weight 或加权采样。
"""

from __future__ import annotations

ALL_MODEL_NAMES = (
    "LogisticRegression",
    "DecisionTree",
    "MLP",
    "XGBoost",
    "LightGBM",
    "FT-Transformer",
    "ResNet",
    "EBM",
    "APLR",
    "CORELS",
)

NEW_MODEL_NAMES = (
    "FT-Transformer",
    "ResNet",
    "EBM",
    "APLR",
    "CORELS",
)

EXPERIMENT_MODEL_NAMES = ALL_MODEL_NAMES
DEEPTAB_MODEL_NAMES = ("FT-Transformer", "ResNet")
PRIOR_CASCADE_MODEL_NAMES = NEW_MODEL_NAMES
EXPERIMENT_SEEDS = (36, 40, 42)

STANDARD_MODEL_PARAMS = {
    "LogisticRegression": {
        "max_iter": 2000,
        "solver": "lbfgs",
        "class_weight": None,
    },
    "DecisionTree": {
        "max_depth": None,
        "class_weight": None,
    },
    "MLP": {
        "hidden_layer_sizes": (256, 128),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "max_iter": 200,
    },
    "XGBoost": {
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": 1,
        "scale_pos_weight": 1.0,
    },
    "LightGBM": {
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 64,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_jobs": 1,
        "class_weight": None,
        "is_unbalance": False,
        "scale_pos_weight": 1.0,
        "verbosity": -1,
    },
    "EBM": {"n_jobs": 1},
    # Two folds keep the configuration valid for contrast1's smallest balanced
    # training set (5 positive + 5 negative rows). No observation weights are used.
    "APLR": {"cv_folds": 2, "n_jobs": 1},
    "CORELS": {
        "c": 0.01,
        "n_iter": 10000,
        "map_type": "prefix",
        "policy": "lower_bound",
        "verbosity": [],
        "max_card": 2,
        "min_support": 0.01,
    },
}

CONTINUOUS_MODEL_PARAMS = {
    "LogisticRegression": {
        **STANDARD_MODEL_PARAMS["LogisticRegression"],
        "warm_start": True,
    },
    "DecisionTree": STANDARD_MODEL_PARAMS["DecisionTree"],
    "MLP": {
        **STANDARD_MODEL_PARAMS["MLP"],
        "hidden_layer_sizes": (64, 32),
        "early_stopping": False,
        "warm_start": True,
    },
    "XGBoost": {
        **STANDARD_MODEL_PARAMS["XGBoost"],
        "n_estimators": 200,
        "max_depth": 5,
    },
    "LightGBM": {
        **STANDARD_MODEL_PARAMS["LightGBM"],
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "reg_lambda": 0.0,
    },
}

DEEPTAB_PARAMS = {
    "FT-Transformer": {
        "model": {
            "d_model": 128,
            "n_layers": 4,
            "n_heads": 8,
            "attn_dropout": 0.1,
            "ff_dropout": 0.1,
            "pooling_method": "avg",
        },
        "lr": 3e-4,
    },
    "ResNet": {
        "model": {
            "layer_sizes": [256, 128, 32],
            "num_blocks": 3,
            "dropout": 0.2,
            "norm": False,
        },
        "lr": 1e-3,
    },
}

DEEPTAB_TRAINER_PARAMS = {
    "max_epochs": 100,
    "patience": 15,
    "monitor": "val_loss",
    "mode": "min",
    "weight_decay": 1e-6,
    "shuffle": True,
    "stratify": True,
}


def parse_model_names(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Parse CLI model selection while keeping the canonical model order."""

    if isinstance(value, str):
        requested = ALL_MODEL_NAMES if value.strip().lower() == "all" else tuple(x.strip() for x in value.split(",") if x.strip())
    else:
        requested = tuple(value)
    unknown = sorted(set(requested) - set(ALL_MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}; expected one of {ALL_MODEL_NAMES}")
    selected = set(requested)
    return tuple(name for name in ALL_MODEL_NAMES if name in selected)
