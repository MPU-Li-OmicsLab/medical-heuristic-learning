from __future__ import annotations

from pathlib import Path

from hl.agent.client import LLMClient
from hl.config import RunConfig
from hl.probes.knowledge import run_knowledge_probe
from hl.utils.io import write_text


def run_knowledge_probe_task(
    *,
    client: LLMClient | None,
    feature_cols: list[str],
    label_col: str,
    run_cfg: RunConfig,
    knowledge_path: Path,
) -> str:
    if client is not None and run_cfg.run_knowledge_probe:
        knowledge = run_knowledge_probe(
            client=client,
            feature_cols=feature_cols,
            target=label_col,
            task_description=run_cfg.task_description,
        )
        knowledge_table = knowledge.markdown_table
        write_text(knowledge_path, knowledge_table)
        return knowledge_table

    if knowledge_path.exists():
        try:
            return knowledge_path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return ""

