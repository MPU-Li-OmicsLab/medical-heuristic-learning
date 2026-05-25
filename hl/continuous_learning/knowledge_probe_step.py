from __future__ import annotations

from pathlib import Path

from hl.agent.client import ChatMessage, LLMClient
from hl.agent.prompts import get_knowledge_probe_prompt
from hl.config import RunConfig
from hl.continuous_learning.config import DriftConfig
from hl.utils.io import write_text
from hl.utils.progress import log_progress


def _read_text_if_exists(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""


def _parse_markdown_table(md: str) -> tuple[list[str], list[list[str]]]:
    lines = [line.strip() for line in (md or "").splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 2:
        return [], []

    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: list[list[str]] = []
    for line in table_lines[2:]:
        row = [cell.strip() for cell in line.strip("|").split("|")]
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        rows.append(row[: len(header)])
    return header, rows


def _render_markdown_table(header: list[str], rows: list[list[str]]) -> str:
    if not header:
        return ""
    separator = ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(separator) + " |"]
    for row in rows:
        out_row = list(row)
        if len(out_row) < len(header):
            out_row = out_row + [""] * (len(header) - len(out_row))
        lines.append("| " + " | ".join(out_row[: len(header)]) + " |")
    return "\n".join(lines).strip()


def _feature_index(header: list[str]) -> int | None:
    for idx, name in enumerate(header):
        if name.lower() in {"feature", "features", "变量", "特征"}:
            return idx
    if header:
        return 0
    return None


def _filter_previous_rows(header: list[str], rows: list[list[str]], drift: DriftConfig) -> list[list[str]]:
    feature_idx = _feature_index(header)
    if feature_idx is None:
        return list(rows)

    kept_rows: list[list[str]] = []
    for row in rows:
        feature_name = str(row[feature_idx]).strip()
        if feature_name in drift.dropped_cols:
            continue
        updated_row = list(row)
        for old_name, new_name in drift.renamed_cols:
            if feature_name == old_name:
                updated_row[feature_idx] = new_name
                break
        kept_rows.append(updated_row)
    return kept_rows


def _query_knowledge_probe(
    *,
    client: LLMClient,
    feature_cols: list[str],
    label_col: str,
    task_description: str,
    prompt_path: Path,
) -> str:
    prompt = get_knowledge_probe_prompt(features=feature_cols, target=label_col, task_description=task_description)
    write_text(prompt_path, prompt + "\n")
    return client.chat_text([ChatMessage(role="user", content=prompt)]).strip()


def run_knowledge_probe_task(
    *,
    client: LLMClient | None,
    feature_cols: list[str],
    label_col: str,
    run_cfg: RunConfig,
    knowledge_path: Path,
    drift: DriftConfig,
) -> str:
    prev_path = drift.prev_hl_out_dir / "probe_knowledge.md" if drift.prev_hl_out_dir is not None else Path("__missing__")
    prev_md = _read_text_if_exists(prev_path).strip()
    if prev_md:
        log_progress("HL-CL-K", f"Loaded previous knowledge probe from {prev_path}.")
    else:
        log_progress("HL-CL-K", "No previous knowledge probe is available under drift context.")
    write_text(knowledge_path.parent / "probe_knowledge_prev.md", prev_md + ("\n" if prev_md else ""))

    prompt_path = knowledge_path.parent / "probe_knowledge_prompt.txt"
    if not run_cfg.run_knowledge_probe or client is None:
        if knowledge_path.exists():
            log_progress("HL-CL-K", f"Reusing existing knowledge probe file: {knowledge_path}.")
            return _read_text_if_exists(knowledge_path).strip()
        if prev_md:
            log_progress("HL-CL-K", "Knowledge probe is skipped; reusing filtered previous knowledge table.")
            header, rows = _parse_markdown_table(prev_md)
            return _render_markdown_table(header, _filter_previous_rows(header, rows, drift))
        write_text(prompt_path, "")
        log_progress("HL-CL-K", "Knowledge probe is unavailable; continuing with empty knowledge context.")
        return ""

    header, rows = _parse_markdown_table(prev_md)
    kept_rows = _filter_previous_rows(header, rows, drift)
    add_features = [col for col in drift.added_cols if col]

    if not prev_md:
        log_progress("HL-CL-K", "No previous knowledge table found; querying a full knowledge probe.")
        full_md = _query_knowledge_probe(
            client=client,
            feature_cols=list(feature_cols),
            label_col=label_col,
            task_description=run_cfg.task_description,
            prompt_path=prompt_path,
        )
        write_text(knowledge_path, full_md + ("\n" if full_md else ""))
        log_progress("HL-CL-K", f"Saved knowledge probe results to {knowledge_path}.")
        return full_md

    if not add_features:
        out_md = _render_markdown_table(header, kept_rows) if header else prev_md
        write_text(prompt_path, "")
        write_text(knowledge_path, out_md + ("\n" if out_md else ""))
        log_progress("HL-CL-K", "No added features detected; wrote filtered previous knowledge table.")
        return out_md

    log_progress("HL-CL-K", f"Querying incremental knowledge probe for {len(add_features)} added features.")
    add_md = _query_knowledge_probe(
        client=client,
        feature_cols=add_features,
        label_col=label_col,
        task_description=run_cfg.task_description,
        prompt_path=prompt_path,
    )
    add_header, add_rows = _parse_markdown_table(add_md)

    if not header and add_header:
        header = add_header
    if not header:
        out_md = add_md.strip()
        write_text(knowledge_path, out_md + ("\n" if out_md else ""))
        return out_md

    normalized_header = [cell.strip().lower() for cell in header]
    normalized_add_header = [cell.strip().lower() for cell in add_header]
    merged_rows = list(kept_rows)
    if normalized_header == normalized_add_header:
        merged_rows.extend(add_rows)
    else:
        for row in add_rows:
            merged_rows.append(row[: len(header)] + [""] * max(0, len(header) - len(row)))

    out_md = _render_markdown_table(header, merged_rows)
    write_text(knowledge_path, out_md + ("\n" if out_md else ""))
    log_progress("HL-CL-K", f"Saved merged knowledge probe results to {knowledge_path}.")
    return out_md
