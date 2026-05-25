from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


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
    ) -> None:
        actual_api_key = api_key or os.getenv(api_key_env, "")
        if not actual_api_key:
            raise RuntimeError(f"API key not provided and environment variable {api_key_env} is not set; cannot call the LLM.")

        self._client = OpenAI(base_url=base_url, api_key=actual_api_key)
        self._model = model_name
        self._temperature = temperature
        self._extra_body = extra_body

    def chat_json(self, messages: list[ChatMessage]) -> str:
        extra = {"extra_body": self._extra_body} if self._extra_body is not None else {}
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=self._temperature,
            response_format={"type":"json_object"},
            **extra,
        )
        return resp.choices[0].message.content or ""

    def chat_text(self, messages: list[ChatMessage]) -> str:
        extra = {"extra_body": self._extra_body} if self._extra_body is not None else {}
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=self._temperature,
            **extra,
        )
        return resp.choices[0].message.content or ""
