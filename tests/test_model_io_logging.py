from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

import hl.model as model_module
from hl.model import load_batch_model, load_model
from hl.utils.io import append_text, ensure_dir, write_json, write_text
from hl.utils.progress import log_progress


def test_load_model_returns_exported_predict_function(tmp_path: Path) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text(
        "def predict(features: dict) -> int:\n"
        "    return int(float(features.get('score', 0.0)) >= 0.0)\n",
        encoding="utf-8",
    )

    predict = load_model(model_path)

    assert predict({"score": 2.0}) == 1
    assert predict({"score": -2.0}) == 0


def test_load_batch_model_predicts_feature_rows_in_input_order(tmp_path: Path) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text(
        "def predict(features: dict) -> int:\n"
        "    return int(float(features.get('score', 0.0)) >= 0.0)\n",
        encoding="utf-8",
    )
    data = pd.DataFrame(
        {"score": [2.0, -2.0, 0.0]},
        index=[10, 20, 30],
    )

    predict_batch = load_batch_model(model_path)

    assert predict_batch(data) == [1, 0, 1]
    assert list(data.columns) == ["score"]


def test_load_batch_model_accepts_feature_only_and_empty_dataframes(tmp_path: Path) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text(
        "def predict(features: dict) -> int:\n"
        "    return int(float(features.get('score', 0.0)) >= 0.0)\n",
        encoding="utf-8",
    )
    predict_batch = load_batch_model(model_path)

    assert predict_batch(pd.DataFrame({"score": [-1.0, 1.0]})) == [0, 1]
    assert predict_batch(pd.DataFrame(columns=["score"])) == []


def test_load_batch_model_loads_artifact_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    load_count = 0

    def fake_load_model(_: str | Path):
        nonlocal load_count
        load_count += 1
        return lambda features: int(float(features["score"]) >= 0.0)

    monkeypatch.setattr(model_module, "load_model", fake_load_model)
    predict_batch = load_batch_model("model.py")
    data = pd.DataFrame({"score": [-1.0, 1.0]})

    assert load_count == 1
    assert predict_batch(data) == [0, 1]
    assert predict_batch(data) == [0, 1]
    assert load_count == 1


def test_load_batch_model_validates_dataframe_shape(tmp_path: Path) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text("def predict(features: dict) -> int:\n    return 0\n", encoding="utf-8")
    predict_batch = load_batch_model(model_path)

    with pytest.raises(TypeError, match="pandas DataFrame"):
        predict_batch([{"score": 1.0}])  # type: ignore[arg-type]

    duplicate_columns = pd.DataFrame([[1.0, 2.0]], columns=["score", "score"])
    with pytest.raises(ValueError, match="unique column names"):
        predict_batch(duplicate_columns)


def test_load_model_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="MHL model file not found"):
        load_model(tmp_path / "missing.py")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("VALUE = 1\n", "Callable `predict\\(features\\)` not found"),
        ("raise ValueError('broken artifact')\n", "Failed to execute MHL model file"),
        ("def predict(:\n    pass\n", "Failed to execute MHL model file"),
    ],
)
def test_load_model_wraps_invalid_artifacts(tmp_path: Path, source: str, message: str) -> None:
    model_path = tmp_path / "invalid_model.py"
    model_path.write_text(source, encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        load_model(model_path)


def test_io_helpers_create_parents_and_preserve_utf8(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "artifact.txt"
    ensure_dir(nested.parent)
    ensure_dir(nested.parent)
    write_text(nested, "第一行\n")
    append_text(nested, "second\n")
    append_text(nested, "third\n")

    payload_path = tmp_path / "json" / "result.json"
    payload = {"说明": "合成数据", "values": [1, 2]}
    write_json(payload_path, payload)

    assert nested.read_text(encoding="utf-8") == "第一行\nsecond\nthird\n"
    assert json.loads(payload_path.read_text(encoding="utf-8")) == payload


def test_progress_uses_hl_logger(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="hl"):
        log_progress("TEST", "synthetic progress")

    record = next(record for record in caplog.records if record.name == "hl")
    assert record.levelno == logging.INFO
    assert record.getMessage() == "[TEST] synthetic progress"
