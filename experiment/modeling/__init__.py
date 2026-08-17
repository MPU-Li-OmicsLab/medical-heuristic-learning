"""对比实验普通模型的统一入口。

这里只重导出模型配置、训练、预测和评估函数；数据切分、实验循环、
结果 CSV 与 HL 流程仍由各实验目录负责。
"""

from .config import ALL_MODEL_NAMES, EXPERIMENT_MODEL_NAMES, NEW_MODEL_NAMES
from .train_eval import (
    FittedModel,
    evaluate_model,
    fit_model,
    predict_model,
    predict_positive_probability,
    save_fitted_model,
)

__all__ = [
    "ALL_MODEL_NAMES",
    "EXPERIMENT_MODEL_NAMES",
    "NEW_MODEL_NAMES",
    "FittedModel",
    "fit_model",
    "predict_model",
    "predict_positive_probability",
    "evaluate_model",
    "save_fitted_model",
]
