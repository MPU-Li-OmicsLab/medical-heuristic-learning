from __future__ import annotations

import json
from pathlib import Path

from hl.agent.client import ChatMessage, LLMClient
from hl.config import RunConfig
from hl.continuous_learning.config import DriftConfig
from hl.evolution.rule_utils import extract_function_name, strip_code_fences, validate_python_syntax
from hl.utils.io import append_text, write_text


def _read_text_if_exists(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""


def _read_blueprint_final(prev_out_dir: Path | None, max_chars: int = 20000) -> str:
    if prev_out_dir is None:
        return ""
    code = _read_text_if_exists(prev_out_dir / "final_heuristic_model.py").strip()
    if not code:
        return ""
    if len(code) <= max_chars:
        return code
    head = code[: max_chars // 2]
    tail = code[-max_chars // 2 :]
    return head + "\n\n[...TRUNCATED...]\n\n" + tail


def _build_continuous_v0_prompt(
    *,
    univariate_summary: str,
    knowledge_table: str,
    metric_desc: str,
    task_description: str,
    drift: DriftConfig,
    blueprint_code: str,
) -> str:
    parts = [
        "You are a medical rule-learning agent updating an existing heuristic policy under feature drift.",
        "Write EVERYTHING in English only. Do not output any Chinese.",
        "",
        "Return STRICT JSON with fields:",
        '- version: "v0"',
        "- error_analysis: the design rationale (in English)",
        "- new_policy_code: full Python function definition; function name MUST be predict_v0",
        "",
        "Function signature MUST be: def predict_v0(features: dict) -> int:",
        "Return an integer class label. For binary tasks, return 0/1 matching the dataset label.",
        "Use features.get('<feature>', default) only.",
        "Do not reference dropped features.",
        "You may use added or renamed features if useful.",
        "Do NOT hardcode assumptions like score starting at 0.5 or a fixed decision threshold of 0.5.",
        "Each if/elif/else branch MUST include an English comment briefly explaining the medical rationale or design intent.",
        "The rule must be self-contained and use ONLY the Python standard library (no third-party packages).",
        "",
        "Feature drift summary:",
        f"- Dropped columns: {list(drift.dropped_cols)}",
        f"- Added columns: {list(drift.added_cols)}",
        f"- Renamed columns: {[{old_name: new_name} for old_name, new_name in drift.renamed_cols]}",
        f"- Change note: {drift.change_note}",
        "",
        "Task description:",
        task_description.strip() or "(empty)",
        "",
        "Optimization metrics:",
        metric_desc.strip(),
        "",
        "[Updated Univariate Summary]",
        univariate_summary.strip() or "(empty)",
        "",
        "[Updated Medical Knowledge Table]",
        knowledge_table.strip() or "(empty)",
        "",
        "[Previous Final Model Blueprint]",
        blueprint_code.strip() or "(missing blueprint)",
    ]
    return "\n".join(parts).strip()


def _parse_proposal(text: str) -> tuple[str, str, str]:
    raw = strip_code_fences(text)
    data = json.loads(raw)
    version = str(data.get("version", ""))
    error_analysis = str(data.get("error_analysis", ""))
    new_policy_code = str(data.get("new_policy_code", ""))
    return version, error_analysis, new_policy_code


def generate_v0_task(
    *,
    client: LLMClient | None,
    run_cfg: RunConfig,
    drift: DriftConfig,
    heuristic_path: Path,
    univariate_summary: str,
    knowledge_table: str,
    metric_desc: str,
) -> None:
    if heuristic_path.exists():
        return
    if not run_cfg.run_v0_generation:
        raise RuntimeError("heuristic_system.py not found and run_v0_generation=False; cannot continue.")
    if client is None:
        raise RuntimeError("llm_enabled=False and heuristic_system.py is missing; cannot generate v0.")

    prompt = _build_continuous_v0_prompt(
        univariate_summary=univariate_summary,
        knowledge_table=knowledge_table,
        metric_desc=metric_desc,
        task_description=run_cfg.task_description,
        drift=drift,
        blueprint_code=_read_blueprint_final(drift.prev_hl_out_dir),
    )
    prompt_path = heuristic_path.parent / "v0_prompt.txt"
    write_text(prompt_path, prompt + "\n")

    last_error: Exception | None = None
    last_resp: str = ""
    for attempt in range(1, max(1, run_cfg.max_llm_attempts) + 1):
        resp = client.chat_json([ChatMessage(role="user", content=prompt)])
        last_resp = resp
        try:
            version, error_analysis, new_policy_code = _parse_proposal(resp)
            if version != "v0":
                raise RuntimeError(f"version mismatch (expected v0, got {version})")
            validate_python_syntax(new_policy_code)
            fn_name = extract_function_name(new_policy_code)
            if fn_name != "predict_v0":
                raise RuntimeError(f"function name mismatch (expected predict_v0, got {fn_name})")

            header = "CURRENT_VERSION = 'v0'\n\n"
            write_text(heuristic_path, header + new_policy_code.strip() + "\n")
            append_text(
                heuristic_path,
                f"\n\nERROR_ANALYSIS_predict_v0 = {json.dumps(error_analysis or 'v0', ensure_ascii=False)}\n",
            )
            write_text(
                heuristic_path.parent / "v0_error_analysis.txt",
                (error_analysis or "v0") + "\n",
            )
            write_text(
                heuristic_path.parent / "v0_attempt_summary.txt",
                f"accepted_attempt={attempt}\n",
            )
            return
        except Exception as exc:
            last_error = exc
            write_text(
                heuristic_path.parent / f"v0_attempt_{attempt}_raw.txt",
                (resp or "") + ("\n" if resp else ""),
            )

    preview = (last_resp or "").strip().replace("\n", "\\n")
    preview = preview[:500]
    raise RuntimeError(f"v0 generation failed after retries: {last_error}; resp_preview={preview}")
