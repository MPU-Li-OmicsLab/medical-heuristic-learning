from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from hl.model import load_model
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
