from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ConstantInputWarning

from hl.metrics import compute_metrics, generate_metric_description
from hl.probes.knowledge import run_knowledge_probe
from hl.probes.univariate import run_univariate_probe
from tests.support import ScriptedLLM, make_knowledge_table, prompt_text


def test_compute_metrics_for_binary_classification() -> None:
    metrics = compute_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0, 1, 1, 0]),
    )

    assert metrics == {
        "ACC": 0.5,
        "F1": 0.5,
        "Sensitivity": 0.5,
        "Specificity": 0.5,
        "TP": 1,
        "FP": 1,
        "TN": 1,
        "FN": 1,
    }


def test_compute_metrics_for_all_correct_and_all_wrong_binary_predictions() -> None:
    y_true = np.asarray([0, 0, 1, 1])

    correct = compute_metrics(y_true, y_true)
    wrong = compute_metrics(y_true, 1 - y_true)

    assert correct == {
        "ACC": 1.0,
        "F1": 1.0,
        "Sensitivity": 1.0,
        "Specificity": 1.0,
        "TP": 2,
        "FP": 0,
        "TN": 2,
        "FN": 0,
    }
    assert wrong == {
        "ACC": 0.0,
        "F1": 0.0,
        "Sensitivity": 0.0,
        "Specificity": 0.0,
        "TP": 0,
        "FP": 2,
        "TN": 0,
        "FN": 2,
    }


@pytest.mark.parametrize(
    ("y_true", "expected_sensitivity", "expected_specificity"),
    [
        ([0, 0, 0], 0.0, 1.0),
        ([1, 1, 1], 1.0, 0.0),
    ],
)
def test_compute_metrics_handles_single_class_without_division_by_zero(
    y_true: list[int],
    expected_sensitivity: float,
    expected_specificity: float,
) -> None:
    metrics = compute_metrics(np.asarray(y_true), np.asarray(y_true))

    assert metrics["Sensitivity"] == expected_sensitivity
    assert metrics["Specificity"] == expected_specificity
    assert all(type(metrics[name]) is float for name in ("ACC", "F1", "Sensitivity", "Specificity"))
    assert all(type(metrics[name]) is int for name in ("TP", "FP", "TN", "FN"))


def test_compute_metrics_for_multiclass_uses_macro_f1() -> None:
    metrics = compute_metrics(
        np.asarray([0, 1, 2, 0, 1, 2]),
        np.asarray([0, 1, 2, 0, 1, 2]),
    )

    assert metrics["ACC"] == 1.0
    assert metrics["F1"] == 1.0
    assert math.isnan(float(metrics["Sensitivity"]))
    assert math.isnan(float(metrics["Specificity"]))
    assert (metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"]) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "priorities",
    [
        (),
        ("F1",),
        ("F1", "ACC"),
        ("F1", "ACC", "Sensitivity"),
        ("F1", "ACC", "Sensitivity", "Specificity"),
    ],
)
def test_metric_description_handles_any_supported_priority_length(priorities: tuple[str, ...]) -> None:
    description = generate_metric_description(priorities)

    assert description
    for metric in priorities:
        assert metric in description


def test_metric_description_ignores_blank_priorities_without_changing_order() -> None:
    description = generate_metric_description(("F1", " ", "ACC", "Sensitivity"))

    assert "F1, ACC, Sensitivity" in description


def test_univariate_probe_handles_continuous_binary_categorical_and_missing_features() -> None:
    size = 24
    frame = pd.DataFrame(
        {
            "continuous": np.linspace(-3.0, 3.0, size),
            "binary": [0] * 12 + [1] * 12,
            "category": ["low"] * 8 + ["mid"] * 8 + ["high"] * 8,
            "many_levels": [f"level_{index}" for index in range(size)],
            "constant": [5.0] * size,
            "all_missing": [None] * size,
            "target": [0] * 12 + [1] * 12,
        }
    )
    frame.loc[[2, 19], "continuous"] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = run_univariate_probe(frame, "target")
    rows = result.set_index("feature")

    assert set(rows.index) == {"continuous", "binary", "category", "many_levels", "constant", "all_missing"}
    assert result["rank"].tolist() == list(range(1, len(result) + 1))
    assert rows.loc["continuous", "feature_type"] == "continuous"
    assert rows.loc["continuous", "n_valid"] == 22
    assert rows.loc["continuous", "missing_rate"] == pytest.approx(2 / 24)
    assert rows.loc["continuous", "mean"] == pytest.approx(frame["continuous"].mean())
    assert rows.loc["continuous", "std"] > 0
    assert rows.loc["continuous", "min"] == -3.0
    assert rows.loc["continuous", "max"] == 3.0
    assert rows.loc["binary", "feature_type"] == "binary"
    assert rows.loc["category", "feature_type"] == "categorical"
    assert rows.loc["constant", "p_value"] == 1.0
    assert rows.loc["all_missing", "n_valid"] == 0

    sort_keys = [
        (float("inf") if pd.isna(p_value) else float(p_value), float(missing_rate))
        for p_value, missing_rate in zip(result["p_value"], result["missing_rate"], strict=True)
    ]
    assert sort_keys == sorted(sort_keys)

    level_counts = json.loads(rows.loc["many_levels", "level_counts"])
    assert len(level_counts) == 21
    assert level_counts["__OTHER__"] == 4


def test_univariate_probe_never_includes_the_label_as_a_feature() -> None:
    frame = pd.DataFrame({"signal": [-1.0, 1.0, -2.0, 2.0], "target": [0, 1, 0, 1]})
    result = run_univariate_probe(frame, "target")
    assert result["feature"].tolist() == ["signal"]


def test_knowledge_probe_uses_text_llm_and_strips_response() -> None:
    table = make_knowledge_table(["risk_score", "age"])
    fake = ScriptedLLM(text_responses=[f"\n{table}\n"])

    result = run_knowledge_probe(
        client=fake,  # type: ignore[arg-type]
        feature_cols=["risk_score", "age"],
        target="target",
        task_description="Synthetic binary task",
    )

    assert result.markdown_table == table
    sent_prompt = prompt_text(fake.text_calls)
    assert "risk_score" in sent_prompt
    assert "Synthetic binary task" in sent_prompt
    fake.assert_exhausted()
