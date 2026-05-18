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
    ) -> None:
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未设置，无法调用大模型。")

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model_name
        self._temperature = temperature

    def chat_json(self, messages: list[ChatMessage]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=self._temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    def chat_text(self, messages: list[ChatMessage]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=self._temperature,
        )
        return resp.choices[0].message.content or ""

