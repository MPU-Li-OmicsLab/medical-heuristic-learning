from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

from hl.config import THINKING_STRENGTH_LEVELS


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        model_name: str,
        temperature: float = 0.3,
        api_key: str | None = None,
        extra_body: dict | None = None,
        thinking_mode: bool | None = None,
        thinking_strength: str | None = None,
    ) -> None:
        actual_api_key = api_key or os.getenv(api_key_env, "")
        if not actual_api_key:
            raise RuntimeError(f"API key not provided and environment variable {api_key_env} is not set; cannot call the LLM.")

        self._client = OpenAI(base_url=base_url, api_key=actual_api_key)
        self._model = model_name
        self._temperature = temperature
        self._extra_body = dict(extra_body) if extra_body is not None else None
        self._thinking_mode = None if thinking_mode is None else bool(thinking_mode)
        self._thinking_strength = self._normalize_thinking_strength(thinking_strength)
        if self._thinking_mode is False and self._thinking_strength is not None:
            raise ValueError(
                "thinking_strength cannot be combined with thinking_mode=False; "
                "use thinking_mode=True (or leave thinking_mode as None) to set a strength."
            )

    @staticmethod
    def _normalize_thinking_strength(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if normalized not in THINKING_STRENGTH_LEVELS:
            allowed = ", ".join(THINKING_STRENGTH_LEVELS)
            raise ValueError(f"thinking_strength must be one of [{allowed}], got {value!r}")
        return normalized

    def _extra_body_for_request(self) -> dict | None:
        """Build the extra_body payload from the explicit thinking-mode controls.

        Rules:

        * An existing 'thinking' key in extra_body always wins: old callers
          that controlled DeepSeek thinking through extra_body keep working.
        * thinking_mode=None (the default) sends nothing, so the backend's
          official default applies (DeepSeek enables thinking mode by default).
        * thinking_mode=True adds '{"thinking": {"type": "enabled"}}'.
        * thinking_mode=False adds '{"thinking": {"type": "disabled"}}'.
        * A thinking_strength without an explicit mode implies thinking_mode=True,
          because reasoning_effort only exists while thinking is enabled.
        """
        extra_body = dict(self._extra_body) if self._extra_body else {}
        if "thinking" not in extra_body:
            if self._thinking_mode is True:
                extra_body["thinking"] = {"type": "enabled"}
            elif self._thinking_mode is False:
                extra_body["thinking"] = {"type": "disabled"}
            elif self._thinking_strength is not None:
                extra_body["thinking"] = {"type": "enabled"}
        return extra_body or None

    def _chat_kwargs(self, messages: list[ChatMessage], *, json_mode: bool) -> dict:
        kwargs = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        extra_body = self._extra_body_for_request()
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        # reasoning_effort only makes sense while thinking is enabled. Omitting it
        # uses the backend's official default (DeepSeek: effort "high").
        thinking_is_on = self._thinking_mode is True or (
            self._thinking_mode is None and self._thinking_strength is not None
        )
        if thinking_is_on and self._thinking_strength is not None:
            thinking_body = extra_body.get("thinking") if extra_body is not None else None
            if not (isinstance(thinking_body, dict) and thinking_body.get("type") == "disabled"):
                kwargs["reasoning_effort"] = self._thinking_strength
        return kwargs

    def chat_json(self, messages: list[ChatMessage]) -> str:
        resp = self._client.chat.completions.create(**self._chat_kwargs(messages, json_mode=True))
        return resp.choices[0].message.content or ""

    def chat_text(self, messages: list[ChatMessage]) -> str:
        resp = self._client.chat.completions.create(**self._chat_kwargs(messages, json_mode=False))
        return resp.choices[0].message.content or ""
