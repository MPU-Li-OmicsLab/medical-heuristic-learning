from __future__ import annotations

import json
from pathlib import Path

from hl.agent.client import ChatMessage, LLMClient
from hl.agent.prompts import get_rule_generation_prompt
from hl.config import RunConfig
from hl.evolution.rule_utils import ParsedProposal, extract_function_name, strip_code_fences, validate_python_syntax
from hl.utils.io import append_text, write_text
from hl.utils.progress import log_progress


def _parse_proposal(text: str) -> ParsedProposal:
    raw = strip_code_fences(text)
    data = json.loads(raw)
    return ParsedProposal(
        version=str(data.get("version", "")),
        error_analysis=str(data.get("error_analysis", "")),
        new_policy_code=str(data.get("new_policy_code", "")),
    )


def generate_v0_task(
    *,
    client: LLMClient | None,
    run_cfg: RunConfig,
    heuristic_path: Path,
    univariate_summary: str,
    knowledge_table: str,
    metric_desc: str,
) -> None:
    if heuristic_path.exists():
        log_progress("HL-V0", f"Reusing existing heuristic file: {heuristic_path}.")
        return
    if not run_cfg.run_v0_generation:
        raise RuntimeError("heuristic_system.py not found and run_v0_generation=False; cannot continue.")
    if client is None:
        raise RuntimeError("llm_enabled=False and heuristic_system.py is missing; cannot generate v0.")

    prompt = get_rule_generation_prompt(
        univariate_summary=univariate_summary,
        knowledge_table=knowledge_table,
        metric_desc=metric_desc,
        task_description=run_cfg.task_description,
    )
    last_error: Exception | None = None
    last_resp: str = ""
    p: ParsedProposal | None = None
    for attempt in range(1, max(1, run_cfg.max_llm_attempts) + 1):
        log_progress("HL-V0", f"Requesting v0 heuristic from LLM (attempt {attempt}/{max(1, run_cfg.max_llm_attempts)}).")
        resp = client.chat_json([ChatMessage(role="user", content=prompt)])
        last_resp = resp
        try:
            p = _parse_proposal(resp)
            if p.version != "v0":
                raise RuntimeError(f"v0 generation failed: version mismatch (expected v0, got {p.version})")
            validate_python_syntax(p.new_policy_code)
            fn_name = extract_function_name(p.new_policy_code)
            if fn_name != "predict_v0":
                raise RuntimeError(f"v0 generation failed: function name mismatch (expected predict_v0, got {fn_name})")
            last_error = None
            break
        except Exception as e:
            last_error = e
            p = None
            log_progress("HL-V0", f"Attempt {attempt} failed validation: {e}.")
            continue
    if last_error is not None or p is None:
        preview = (last_resp or "").strip().replace("\n", "\\n")
        preview = preview[:500]
        raise RuntimeError(f"v0 generation failed after retries: {last_error}; resp_preview={preview}")

    header = "CURRENT_VERSION = 'v0'\n\n"
    write_text(heuristic_path, header + p.new_policy_code.strip() + "\n")
    v0_error_analysis = p.error_analysis or "v0"
    append_text(heuristic_path, f"\n\nERROR_ANALYSIS_predict_v0 = {json.dumps(v0_error_analysis, ensure_ascii=False)}\n")
    log_progress("HL-V0", f"Accepted and saved v0 heuristic to {heuristic_path}.")
