from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import hl.agent.client as client_module
from hl.agent.client import ChatMessage, LLMClient
from hl.agent.continuous_prompts import (
    get_continuous_iteration_prompt,
    get_continuous_v0_generation_prompt,
)
from hl.agent.prompts import get_iteration_prompt, get_knowledge_probe_prompt, get_rule_generation_prompt


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = SimpleNamespace(content="synthetic response")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAI:
    instances: list["_FakeOpenAI"] = []

    def __init__(self, *, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=_FakeCompletions())
        self.instances.append(self)


@pytest.fixture(autouse=True)
def fake_openai_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeOpenAI.instances.clear()
    monkeypatch.setattr(client_module, "OpenAI", _FakeOpenAI)


def _client(**overrides: Any) -> LLMClient:
    kwargs: dict[str, Any] = {
        "base_url": "https://llm.invalid/v1",
        "api_key_env": "TEST_LLM_KEY",
        "model_name": "fake-model",
        "api_key": "fake-key",
    }
    kwargs.update(overrides)
    return LLMClient(**kwargs)


def test_llm_client_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TEST_LLM_KEY is not set"):
        LLMClient(
            base_url="https://llm.invalid/v1",
            api_key_env="TEST_LLM_KEY",
            model_name="fake-model",
        )
    assert _FakeOpenAI.instances == []


def test_llm_client_reads_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "environment-key")
    _client(api_key=None)
    assert _FakeOpenAI.instances[0].api_key == "environment-key"


@pytest.mark.parametrize("strength", ["low", "medium", "high", "xhigh", "max"])
def test_thinking_strength_is_normalized_and_sent(strength: str) -> None:
    client = _client(thinking_strength=f" {strength.upper()} ")
    kwargs = client._chat_kwargs([ChatMessage("user", "hello")], json_mode=False)

    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert kwargs["reasoning_effort"] == strength


def test_explicit_thinking_disable_is_sent_without_reasoning_effort() -> None:
    client = _client(thinking_mode=False)
    kwargs = client._chat_kwargs([ChatMessage("user", "hello")], json_mode=False)
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in kwargs


def test_default_thinking_controls_do_not_add_vendor_parameters() -> None:
    client = _client()
    kwargs = client._chat_kwargs([ChatMessage("user", "hello")], json_mode=False)

    assert "extra_body" not in kwargs
    assert "reasoning_effort" not in kwargs


def test_existing_extra_body_thinking_setting_wins_without_mutating_input() -> None:
    extra_body = {"thinking": {"type": "disabled"}, "vendor_flag": True}
    client = _client(thinking_mode=True, thinking_strength="high", extra_body=extra_body)
    kwargs = client._chat_kwargs([ChatMessage("user", "hello")], json_mode=False)

    assert kwargs["extra_body"] == extra_body
    assert "reasoning_effort" not in kwargs
    assert extra_body == {"thinking": {"type": "disabled"}, "vendor_flag": True}


def test_invalid_thinking_combinations_are_rejected() -> None:
    with pytest.raises(ValueError, match="thinking_strength must be one of"):
        _client(thinking_strength="extreme")
    with pytest.raises(ValueError, match="cannot be combined"):
        _client(thinking_mode=False, thinking_strength="high")


def test_chat_text_and_json_forward_expected_request_shapes() -> None:
    client = _client(temperature=0.1)
    messages = [ChatMessage("system", "rules"), ChatMessage("user", "question")]

    assert client.chat_text(messages) == "synthetic response"
    assert client.chat_json(messages) == "synthetic response"

    calls = _FakeOpenAI.instances[0].chat.completions.calls
    assert calls[0]["model"] == "fake-model"
    assert calls[0]["temperature"] == 0.1
    assert calls[0]["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
    ]
    assert "response_format" not in calls[0]
    assert calls[1]["response_format"] == {"type": "json_object"}


def test_standard_prompts_include_contract_and_runtime_context() -> None:
    knowledge = get_knowledge_probe_prompt(["age", "risk"], "target", "Synthetic task")
    v0 = get_rule_generation_prompt("summary", "knowledge", "F1 first", "Synthetic task")
    iteration = get_iteration_prompt(
        "def predict_v0(features): ...",
        "errors",
        "trajectory",
        "Regressions=1",
        "F1 first",
        "Synthetic task",
        "v1",
    )

    assert '"age"' in knowledge and "Synthetic task" in knowledge
    assert "Write EVERYTHING in English only" in knowledge
    assert "Suggested threshold" in knowledge and "Evidence confidence" in knowledge
    assert "predict_v0" in v0 and "STRICT JSON" in v0 and "features.get" in v0
    assert "ONLY the Python standard library" in v0
    assert '"version": "v1"' in iteration
    assert "Regressions=1" in iteration


def test_continuous_prompts_include_drift_blueprint_and_next_version() -> None:
    v0 = get_continuous_v0_generation_prompt(
        univariate_summary="new summary",
        knowledge_table="new knowledge",
        metric_desc="F1 first",
        task_description="Drift task",
        dropped_cols=("old_lab",),
        added_cols=("new_lab",),
        renamed_cols=(("old_score", "new_score"),),
        change_note="schema v2",
        blueprint_code="def predict(features): return 0",
    )
    iteration = get_continuous_iteration_prompt(
        current_code="def predict_v0(features): return 0",
        error_report="errors",
        trajectory="None",
        degradation_warning="No regressions.",
        metric_desc="F1 first",
        task_description="Drift task",
        next_version="v2",
    )

    for token in ("old_lab", "new_lab", "old_score", "new_score", "schema v2", "Previous Final Model Blueprint"):
        assert token in v0
    assert '"version": "v2"' in iteration
