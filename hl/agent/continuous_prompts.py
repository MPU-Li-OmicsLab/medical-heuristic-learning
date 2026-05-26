from __future__ import annotations

import json


def get_continuous_knowledge_probe_prompt(features: list[str], target: str, task_description: str = "") -> str:
    return (
        "You are a medical prior-knowledge assistant. Generate clinical prior knowledge for the given tabular medical features.\n"
        "Write EVERYTHING in English only.\n"
        + (f"Task description: {task_description}\n" if task_description else "")
        + f"Outcome/target column: {target}\n"
        "You MUST return a Markdown table that includes the relationships between the features and the outcome, with exactly these columns:\n"
        "| Feature | Univariate signal (summary) | Clinical rationale | Suggested threshold | Evidence confidence (high/medium/low) |\n"
        "Requirements:\n"
        "- You must provide a suggested threshold (if not applicable, write \"no clear threshold\" and explain why)\n"
        "- Clinical rationale should be specific enough to justify rules\n"
        "- Evidence confidence must be one of: high / medium / low\n"
        "Feature list (json):\n"
        + json.dumps(features, ensure_ascii=False)
    )


def get_continuous_v0_generation_prompt(
    *,
    univariate_summary: str,
    knowledge_table: str,
    metric_desc: str,
    task_description: str,
    dropped_cols: tuple[str, ...],
    added_cols: tuple[str, ...],
    renamed_cols: tuple[tuple[str, str], ...],
    change_note: str,
    blueprint_code: str,
) -> str:
    return (
        "You are a medical rule-learning agent. I have already built a pure-Python classification rule function. "
        "However, feature drift has occurred. Based on the following summary of feature drift, together with the "
        "provided univariate summary, medical knowledge table, metric priority, and other inputs, update the "
        "classification rule function.\n"
        "Write EVERYTHING in English only.\n"
        "Inputs include: feature drift summary, updated univariate summary, updated medical knowledge table, "
        "metric priority, and the previous final model blueprint.\n"
        + (f"Task description: {task_description}\n" if task_description else "")
        + f"{metric_desc}\n\n"
        + "[Feature Drift Summary]\n"
        + f"- Dropped columns: {list(dropped_cols)}\n"
        + f"- Added columns: {list(added_cols)}\n"
        + f"- Renamed columns: {[{old_name: new_name} for old_name, new_name in renamed_cols]}\n"
        + f"- Change note: {change_note}\n\n"
        + ("[Updated Univariate Summary]\n" + f"{univariate_summary}\n\n" if univariate_summary else "")
        + ("[Updated Medical Knowledge Table]\n" + f"{knowledge_table}\n\n" if knowledge_table else "")
        + "[Previous Final Model Blueprint]\n"
        + f"{blueprint_code.strip()}\n\n"
        + "Return STRICT JSON with fields:\n"
        '- version: "v0"\n'
        "- error_analysis: the design rationale (in English)\n"
        "- new_policy_code: full Python function definition; function name MUST be predict_v0\n"
        "Function signature MUST be: def predict_v0(features: dict) -> int:\n"
        "Return an integer class label. For binary tasks, return 0/1 matching the dataset label.\n"
        "Do not reference dropped features.\n"
        "You may use added or renamed features if useful.\n"
        "Each if/elif/else branch MUST include an English comment briefly explaining the medical rationale or design intent.\n"
        "The rule must be self-contained and use ONLY the Python standard library (no third-party packages).\n"
    )


def get_continuous_iteration_prompt(
    *,
    current_code: str,
    error_report: str,
    trajectory: str,
    degradation_warning: str,
    metric_desc: str,
    task_description: str,
    next_version: str,
) -> str:
    return (
        "You are a medical rule-learning agent. Update the current Python-based classification rule based on:\n"
        "- Current full code (all historical versions)\n"
        "- This round's training-set error analysis\n"
        "- Iteration trajectory (reasons for previous changes)\n"
        "- Metric priority\n"
        "- (If any) degradation warning: cases that regressed from correct to wrong after the last change\n\n"
        + (f"Task description: {task_description}\n\n" if task_description else "")
        + f"{metric_desc}\n\n"
        "[Current Full Code]\n"
        f"{current_code}\n\n"
        "[Training Error Analysis]\n"
        f"{error_report}\n\n"
        "[Iteration Trajectory]\n"
        f"{trajectory}\n\n"
        "[Degradation Warning]\n"
        f"{degradation_warning}\n\n"
        "Important: if a degradation warning exists, you MUST prioritize fixing regressed cases (previously correct, now wrong) and try not to introduce new regressions.\n"
        "Each if/elif/else branch MUST include an English comment briefly explaining the medical rationale or design intent.\n"
        "Extra constraint: do NOT collapse to predicting almost all 1s or 0s.\n"
        "Minimal-change constraint: only make small adjustments this round (e.g., adjust 1–2 thresholds/weights, or add/remove no more than 2 rules). Do NOT rewrite the whole function.\n"
        "Return STRICT JSON:\n"
        "{\n"
        f'  "version": "{next_version}",\n'
        '  "error_analysis": "...(English)...",\n'
        '  "new_policy_code": "def predict_...\\n ..."\n'
        "}\n"
        "Keep changes minimal and comments clear.\n"
        "The rule must be self-contained and use ONLY the Python standard library (no third-party packages).\n"
    )
