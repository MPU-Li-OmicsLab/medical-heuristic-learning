from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable
from typing import Any

from hl.agent.client import ChatMessage


class ScriptedLLM:
    """Small deterministic LLM double used by integration tests.

    Responses are consumed in order. An unexpected call fails immediately,
    which protects the tests from accidentally reaching a real LLM client.
    """

    def __init__(
        self,
        *,
        json_responses: Iterable[str | BaseException] = (),
        text_responses: Iterable[str | BaseException] = (),
    ) -> None:
        self._json_responses = deque(json_responses)
        self._text_responses = deque(text_responses)
        self.json_calls: list[list[ChatMessage]] = []
        self.text_calls: list[list[ChatMessage]] = []

    @staticmethod
    def _consume(queue: deque[str | BaseException], call_name: str) -> str:
        if not queue:
            raise AssertionError(f"Unexpected fake LLM {call_name} call")
        response = queue.popleft()
        if isinstance(response, BaseException):
            raise response
        return response

    def chat_json(self, messages: list[ChatMessage]) -> str:
        self.json_calls.append(list(messages))
        return self._consume(self._json_responses, "chat_json")

    def chat_text(self, messages: list[ChatMessage]) -> str:
        self.text_calls.append(list(messages))
        return self._consume(self._text_responses, "chat_text")

    def assert_exhausted(self) -> None:
        assert not self._json_responses, "Unused fake JSON responses remain"
        assert not self._text_responses, "Unused fake text responses remain"


def make_rule_proposal(
    version: str,
    *,
    feature: str = "risk_score",
    threshold: float = 0.0,
    error_analysis: str | None = None,
) -> str:
    code = f'''def predict_{version}(features: dict) -> int:
    value = float(features.get({feature!r}, 0.0))
    if value >= {threshold!r}:
        # Higher synthetic risk indicates the positive class.
        return 1
    # Lower synthetic risk indicates the negative class.
    return 0'''
    return json.dumps(
        {
            "version": version,
            "error_analysis": error_analysis or f"Synthetic rule for {version}",
            "new_policy_code": code,
        }
    )


def make_knowledge_table(features: Iterable[str]) -> str:
    header = (
        "| Feature | Univariate signal (summary) | Clinical rationale | "
        "Suggested threshold | Evidence confidence (high/medium/low) |"
    )
    separator = "| --- | --- | --- | --- | --- |"
    rows = [
        f"| {feature} | synthetic association | deterministic fixture | 0 | high |"
        for feature in features
    ]
    return "\n".join([header, separator, *rows])


def prompt_text(calls: list[list[ChatMessage]], call_index: int = 0) -> str:
    return "\n".join(message.content for message in calls[call_index])


def install_fake_client(monkeypatch: Any, module: Any, fake: ScriptedLLM) -> dict[str, Any]:
    """Patch an orchestrator's LLMClient symbol and capture constructor kwargs."""

    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> ScriptedLLM:
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(module, "LLMClient", factory)
    return captured
