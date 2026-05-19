from __future__ import annotations

import json


def get_knowledge_probe_prompt(features: list[str], target: str, task_description: str = "") -> str:
    return (
        "You are a medical prior-knowledge assistant. Generate clinical prior knowledge for the given tabular medical features.\n"
        "Write EVERYTHING in English only. Do not output any Chinese.\n"
        + (f"Task description: {task_description}\n" if task_description else "")
        + f"Outcome/target column: {target}\n"
        "You MUST return a Markdown table with exactly these columns:\n"
        "| Feature | Univariate signal (summary) | Clinical rationale | Suggested threshold | Evidence confidence (high/medium/low) |\n"
        "Requirements:\n"
        "- You must provide a suggested threshold (if not applicable, write \"no clear threshold\" and explain why)\n"
        "- Clinical rationale should be specific enough to justify rules\n"
        "- Evidence confidence must be one of: high / medium / low\n"
        "Feature list (json):\n"
        + json.dumps(features, ensure_ascii=False)
    )


def get_rule_generation_prompt(
    univariate_summary: str, knowledge_table: str, metric_desc: str, task_description: str = ""
) -> str:
    return (
        "You are a medical rule-learning agent. You will generate a pure-Python classification rule function.\n"
        "Write EVERYTHING in English only. Do not output any Chinese.\n"
        "Inputs include: univariate summary, medical knowledge table, and metric priority.\n"
        + (f"Task description: {task_description}\n" if task_description else "")
        + f"{metric_desc}\n\n"
        "[Univariate Summary]\n"
        f"{univariate_summary}\n\n"
        "[Medical Knowledge Table]\n"
        f"{knowledge_table}\n\n"
        "Return STRICT JSON with fields:\n"
        '- version: "v0"\n'
        "- error_analysis: the design rationale (in English)\n"
        "- new_policy_code: full Python function definition; function name MUST be predict_v0\n"
        "Function signature MUST be: def predict_v0(features: dict) -> int:\n"
        "Return an integer class label. For binary tasks, return 0/1 matching the dataset label.\n"
        # "Do NOT hardcode assumptions like score starting at 0.5 or a fixed decision threshold of 0.5.\n"
        "You may implement a score-based rule, a direct rule-based decision tree, or any deterministic rule set, as long as it is a pure-Python function.\n"
        "Each if/elif/else branch MUST include an English comment briefly explaining the medical rationale or design intent.\n"
        "The rule must be self-contained and use ONLY the Python standard library (no third-party packages).\n"
    )


def get_iteration_prompt(
    current_code: str,
    error_report: str,
    trajectory: str,
    degradation_warning: str,
    metric_desc: str,
    task_description: str,
    next_version: str,
) -> str:
    return (
        "You are a medical rule-learning agent. Update the current classification rule based on:\n"
        "Write EVERYTHING in English only. Do not output any Chinese.\n"
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
