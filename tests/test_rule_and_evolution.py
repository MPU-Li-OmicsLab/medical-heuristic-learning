from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hl.evolution.degradation import (
    collect_degradation_examples,
    detect_degradation,
    format_degradation_warning,
)
from hl.evolution.error_analysis import collect_errors, format_error_report
from hl.evolution.rule_utils import (
    extract_function_name,
    strip_code_fences,
    validate_python_syntax,
    validate_undefined_names,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("  value = 1  ", "value = 1"),
        ("```python\nvalue = 1\n```", "value = 1"),
        ("```py\nvalue = 1\n```", "value = 1"),
        ("```js\nvalue = 1\n```", "value = 1"),
        ("```\nvalue = 1\n```", "value = 1"),
    ],
)
def test_strip_code_fences(source: str, expected: str) -> None:
    assert strip_code_fences(source) == expected


def test_python_rule_validation_accepts_features_globals_and_standard_library() -> None:
    code = '''import math

THRESHOLD = 0.5

def predict_v0(features: dict) -> int:
    value = float(features.get("score", 0.0))
    return int(math.fabs(value) >= THRESHOLD)
'''
    validate_python_syntax(code)
    validate_undefined_names(code)
    assert extract_function_name(code) == "predict_v0"


def test_python_syntax_validation_rejects_invalid_code() -> None:
    with pytest.raises(SyntaxError):
        validate_python_syntax("def predict_v0(:\n    pass")


def test_undefined_name_validation_explains_bare_feature_error() -> None:
    code = '''def predict_v0(features: dict) -> int:
    return int(testosterone > 3.0)
'''
    with pytest.raises(ValueError, match="undefined name 'testosterone'.*features.*dict"):
        validate_undefined_names(code)


def test_extract_function_name_returns_none_for_invalid_or_empty_code() -> None:
    assert extract_function_name("VALUE = 1") is None
    assert extract_function_name("def broken(:") is None


def test_extract_function_name_returns_the_first_top_level_function() -> None:
    code = "def predict_v0(features):\n    return 0\n\ndef helper():\n    return 1\n"
    assert extract_function_name(code) == "predict_v0"


def test_collect_errors_reports_exact_binary_errors_and_default_features() -> None:
    frame = pd.DataFrame(
        {
            "signal": [-2.0, -1.0, 1.0, 2.0],
            "other": [10, 11, 12, 13],
            "target": [0, 0, 1, 1],
        }
    )

    samples = collect_errors(
        frame,
        "target",
        np.asarray([1, 0, 0, 1]),
        max_error_samples=10,
        random_seed=7,
    )

    assert [(sample.idx, sample.kind) for sample in samples] == [(0, "FP"), (2, "FN")]
    assert all(set(sample.features) == {"signal", "other"} for sample in samples)


def test_collect_errors_returns_empty_when_all_predictions_are_correct() -> None:
    frame = pd.DataFrame({"signal": [-1.0, 1.0], "target": [0, 1]})
    assert collect_errors(frame, "target", np.asarray([0, 1]), 10, 3) == []


def test_collect_errors_is_deterministic_and_limits_reported_features() -> None:
    frame = pd.DataFrame(
        {
            "signal": [-2.0, -1.0, 1.0, 2.0, 3.0],
            "unused": [10, 11, 12, 13, 14],
            "target": [0, 0, 1, 1, 1],
        }
    )
    predictions = np.asarray([1, 0, 0, 0, 1])

    first = collect_errors(
        frame,
        "target",
        predictions,
        max_error_samples=2,
        random_seed=7,
        feature_cols=["signal"],
    )
    second = collect_errors(
        frame,
        "target",
        predictions,
        max_error_samples=2,
        random_seed=7,
        feature_cols=["signal"],
    )

    assert first == second
    assert len(first) == 2
    assert all(set(sample.features) == {"signal"} for sample in first)
    assert {sample.kind for sample in first}.issubset({"FP", "FN"})

    report = format_error_report(first, max_details=1)
    assert "Error samples=2" in report
    assert "Showing only the first 1 samples" in report


def test_collect_errors_uses_generic_kind_for_multiclass() -> None:
    frame = pd.DataFrame({"signal": [0, 1, 2], "target": [0, 1, 2]})
    samples = collect_errors(frame, "target", np.asarray([1, 1, 0]), 10, 3)
    assert [sample.kind for sample in samples] == ["ERR", "ERR"]


def test_error_report_handles_no_errors() -> None:
    assert format_error_report([]) == "No error samples."


def test_degradation_detection_warning_and_examples_are_consistent() -> None:
    frame = pd.DataFrame(
        {
            "signal": [-2.0, -1.0, 1.0, 2.0],
            "other": [5, 6, 7, 8],
            "target": [0, 0, 1, 1],
        }
    )
    old_predictions = np.asarray([0, 0, 0, 1])
    new_predictions = np.asarray([1, 0, 1, 0])

    degradation = detect_degradation(frame["target"].to_numpy(), old_predictions, new_predictions)
    assert degradation.degraded_indices == [0, 3]
    assert format_degradation_warning(degradation.degraded_indices, max_items=1) == (
        "Regressions=2, example_indices=[0] (1 more not shown)"
    )

    examples = collect_degradation_examples(
        frame,
        "target",
        degradation.degraded_indices,
        old_predictions,
        new_predictions,
        feature_cols=["signal"],
        max_samples=1,
        random_seed=11,
    )
    assert len(examples) == 1
    assert examples[0]["idx"] in {0, 3}
    assert set(examples[0]["features"]) == {"signal"}


def test_degradation_helpers_handle_no_regressions() -> None:
    assert format_degradation_warning([]) == "No regressions."
    assert collect_degradation_examples(
        pd.DataFrame({"x": [1], "target": [1]}),
        "target",
        [],
        np.asarray([1]),
        np.asarray([1]),
        ["x"],
        10,
        1,
    ) == []
