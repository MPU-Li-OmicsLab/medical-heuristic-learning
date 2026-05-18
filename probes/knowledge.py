from __future__ import annotations

from dataclasses import dataclass

from agent.client import ChatMessage, LLMClient
from agent.prompts import get_knowledge_probe_prompt


@dataclass(frozen=True)
class KnowledgeProbeResult:
    markdown_table: str


def run_knowledge_probe(
    client: LLMClient, features: list[str], target: str, task_description: str = ""
) -> KnowledgeProbeResult:
    prompt = get_knowledge_probe_prompt(features=features, target=target, task_description=task_description)
    text = client.chat_text([ChatMessage(role="user", content=prompt)])
    return KnowledgeProbeResult(markdown_table=text.strip())
