"""构造对比实验使用的普通模型。

build_model() 统一创建 sklearn、XGBoost、LightGBM、DeepTab、InterpretML
和 CORELS 模型。该模块只负责实例化，不负责切分数据或写实验结果。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import (
    ALL_MODEL_NAMES,
    CONTINUOUS_MODEL_PARAMS,
    DEEPTAB_PARAMS,
    DEEPTAB_TRAINER_PARAMS,
    STANDARD_MODEL_PARAMS,
)


def build_model(
    model_name: str,
    seed: int,
    *,
    train_size: int,
    checkpoint_dir: Path | None = None,
    variant: str = "standard",
    stage: int = 1,
) -> tuple[str, Any]:
    """Build one model with automatic class balancing explicitly disabled."""

    if model_name not in ALL_MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    if variant not in {"standard", "continuous"}:
        raise ValueError(f"Unknown model variant: {variant}")

    params_source = CONTINUOUS_MODEL_PARAMS if variant == "continuous" else STANDARD_MODEL_PARAMS
    params = deepcopy(params_source.get(model_name, STANDARD_MODEL_PARAMS.get(model_name, {})))

    if model_name == "LogisticRegression":
        return "sklearn", LogisticRegression(random_state=seed, **params)
    if model_name == "DecisionTree":
        return "sklearn", DecisionTreeClassifier(random_state=seed, **params)
    if model_name == "MLP":
        params.setdefault("early_stopping", variant == "standard" and int(train_size) >= 100)
        params["batch_size"] = min(256 if variant == "standard" else 64, max(2, int(train_size)))
        return "sklearn", MLPClassifier(random_state=seed, **params)
    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        return "sklearn", XGBClassifier(random_state=seed, **params)
    if model_name == "LightGBM":
        from lightgbm import LGBMClassifier

        if variant == "continuous" and int(stage) == 2:
            params["n_estimators"] = 200
        return "sklearn", LGBMClassifier(random_state=seed, **params)
    if model_name == "EBM":
        from interpret.glassbox import ExplainableBoostingClassifier

        return "ebm", ExplainableBoostingClassifier(random_state=seed, **params)
    if model_name == "APLR":
        from interpret.glassbox import APLRClassifier

        return "aplr", APLRClassifier(random_state=seed, **params)
    if model_name == "CORELS":
        from corels import CorelsClassifier

        return "corels", CorelsClassifier(**params)

    from deeptab.configs import FTTransformerConfig, PreprocessingConfig, ResNetConfig, TrainerConfig
    from deeptab.models import FTTransformerClassifier, ResNetClassifier

    checkpoint_dir = checkpoint_dir or Path("model_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    batch_size = max(2, min(128, int(train_size)))
    trainer_params = {
        **DEEPTAB_TRAINER_PARAMS,
        "batch_size": batch_size,
        "lr": DEEPTAB_PARAMS[model_name]["lr"],
        "checkpoint_path": str(checkpoint_dir),
    }
    preprocessing_config = PreprocessingConfig(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
    )
    trainer_config = TrainerConfig(**trainer_params)
    if model_name == "FT-Transformer":
        estimator = FTTransformerClassifier(
            model_config=FTTransformerConfig(**DEEPTAB_PARAMS[model_name]["model"]),
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=seed,
        )
    else:
        estimator = ResNetClassifier(
            model_config=ResNetConfig(**DEEPTAB_PARAMS[model_name]["model"]),
            preprocessing_config=preprocessing_config,
            trainer_config=trainer_config,
            random_state=seed,
        )
    return "deeptab", estimator
