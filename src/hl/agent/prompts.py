from __future__ import annotations

import json


_EVIDENCE_GUIDANCE = (
    "- Use the univariate summary as training-data evidence and the medical knowledge table as prior knowledge, not as measured dataset results.\n"
    "- Consider association direction, effect size, sample support, missingness, and feature distributions together; a small p-value alone does not justify a rule or establish causality.\n"
    "- Treat suggested clinical thresholds as candidates. Check units, scale, encoding, and observed ranges before using them; do not apply raw clinical cutoffs to normalized values.\n"
    "- When data and clinical priors disagree, explain the conflict and prefer a conservative, data-supported choice. Do not invent units, category meanings, statistics, or evidence.\n"
    "- Missing or incomplete references mean uncertainty, not absence of an association. Use the available evidence and state relevant limitations.\n\n"
)

_RULE_CONSTRAINTS = (
    "[Code Requirements]\n"
    "- All feature values MUST be read from the 'features' dict (features['col'] or features.get('col', default)); never write a feature name as a bare variable. Local variables assigned from these reads are allowed.\n"
    "- Use exact provided feature names and observed encodings. Do not use the target label, sample indices, or memorized examples as predictors.\n"
    "- Return an integer class label on every path. For binary tasks, return 0/1 matching the dataset label.\n"
    "- Use a score-based rule, a direct decision tree, or another deterministic rule set. Justify score initialization, weights, and decision thresholds; do not assume a score is a calibrated probability or default to 0.5 without support.\n"
    "- Each if/elif/else branch MUST include an English comment briefly explaining the medical rationale or design intent.\n"
    "- The rule must be self-contained and use ONLY the Python standard library (no third-party packages). Put required imports and helpers inside the function; do not depend on historical functions or external state.\n\n"
)


def get_knowledge_probe_prompt(features: list[str], target: str, task_description: str = "") -> str:
    return (
        "You are a medical prior-knowledge assistant. Summarize clinically plausible relationships between the supplied tabular features and the outcome.\n"
        "Write EVERYTHING in English only, preserving exact feature names and identifiers.\n"
        + (f"Task description: {task_description}\n" if task_description else "")
        + f"Outcome/target column: {target}\n\n"
        "Return only a Markdown table with exactly these columns and one row per supplied feature:\n"
        "| Feature | Univariate signal (summary) | Clinical rationale | Suggested threshold | Evidence confidence (high/medium/low) |\n"
        "Requirements:\n"
        "- Preserve feature names exactly. Relate each feature to the specified outcome, rather than to general health alone.\n"
        "- In 'Univariate signal (summary)', describe the expected association from prior knowledge.\n"
        "- Explain the clinical rationale, expected direction, and relevant context or limitations sufficiently to inform interpretable rules.\n"
        "- Provide a suggested threshold only when defensible, with units and applicability conditions. If units, encoding, feature meaning, or a useful cutoff are unclear, write \"no clear threshold\" and explain why. Do not invent a numeric cutoff.\n"
        "- Evidence confidence must be exactly high / medium / low: high for well-established, directly applicable knowledge; medium for context-dependent or indirect knowledge; low for uncertain or ambiguous relationships.\n"
        "- Do not fabricate statistics, study citations, or guideline claims. Acknowledge uncertainty when the feature or outcome is ambiguous.\n"
        "Feature list (json):\n"
        + json.dumps(features, ensure_ascii=False)
    )


def get_rule_generation_prompt(
    univariate_summary: str, knowledge_table: str, metric_desc: str, task_description: str = ""
) -> str:
    return (
        "You are a medical rule-learning agent. Generate a compact, interpretable pure-Python classification rule from the supplied evidence.\n"
        "Write EVERYTHING in English only, preserving exact feature names and identifiers.\n"
        + (f"Task description: {task_description}\n" if task_description else "")
        + f"[Metric Priority]\n{metric_desc}\n\n"
        + f"[Univariate Summary]\n{univariate_summary or 'Not available.'}\n\n"
        + f"[Medical Knowledge Table]\n{knowledge_table or 'Not available.'}\n\n"
        + "[Evidence Use]\n"
        + _EVIDENCE_GUIDANCE
        + "[Rule Design]\n"
        "- Follow the stated metric priority when balancing false positives and false negatives; do not optimize a lower-priority metric at the expense of the primary objective without justification.\n"
        "- Start with a small set of supported rules. Avoid redundant contributions from strongly related features and add interactions only when justified by the supplied evidence.\n"
        "- Explain the selected features, rule directions, candidate thresholds, missing-value handling, and expected metric tradeoffs. Distinguish supported facts from design assumptions; do not claim unmeasured performance.\n\n"
        + _RULE_CONSTRAINTS
        + "[Output Contract]\n"
        "Return STRICT JSON only, without Markdown fences or surrounding text, containing exactly these fields:\n"
        '- version: "v0"\n'
        "- error_analysis: a concise English design rationale covering the evidence, assumptions, and tradeoffs above\n"
        "- new_policy_code: the complete executable Python function definition as a JSON string, not a patch or pseudocode\n"
        "Function signature MUST be: def predict_v0(features: dict) -> int:\n"
        "Escape newlines and quotes correctly in JSON strings.\n"
    )


def get_iteration_prompt(
    current_code: str,
    error_report: str,
    trajectory: str,
    degradation_warning: str,
    metric_desc: str,
    task_description: str,
    next_version: str,
    univariate_summary: str = "",
    knowledge_table: str = "",
) -> str:
    return (
        "You are a medical rule-learning agent. Improve the current Python classification rule with a small, evidence-supported revision.\n"
        "Write EVERYTHING in English only, preserving exact feature names and identifiers.\n"
        + (f"Task description: {task_description}\n" if task_description else "")
        + f"[Metric Priority]\n{metric_desc}\n\n"
        + f"[Univariate Summary]\n{univariate_summary or 'Not available.'}\n\n"
        + f"[Medical Knowledge Table]\n{knowledge_table or 'Not available.'}\n\n"
        + f"[Current Full Code]\n{current_code}\n\n"
        + f"[Training Error Analysis]\n{error_report}\n\n"
        + f"[Iteration Trajectory]\n{trajectory}\n\n"
        + f"[Degradation Warning]\n{degradation_warning}\n\n"
        + "[Revision Strategy]\n"
        "- Develop the next version by iterating on the active rule identified by CURRENT_VERSION in the current code. Use the previous versions, their reported outcomes, the current training error analysis, and the iteration trajectory as the primary basis for deciding what to retain and what to change; preserve successful logic and learn from earlier changes.\n"
        "- Identify a recurring false-positive or false-negative pattern and connect it to the responsible branch, threshold, or interaction. Error examples may be a subset and do not describe the full class distribution.\n"
        "- Use the medical knowledge table and univariate summary only as auxiliary references when adding, removing, or modifying rules. They can help assess feature choices, directions, and thresholds, but must not replace the previous results and current error patterns as the basis for iteration or trigger a redesign on their own. Explain relevant conflicts or missing support instead of forcing a clinical prior onto the data.\n"
        "- If actual regressed cases are reported (previously correct, now wrong), prioritize repairing their shared failure pattern while preserving successful rules and the stated metric priority. A report of no regressions requires no regression repair.\n"
        "- Consider which previously correct cases a change could harm. Avoid memorizing individual samples or repeating previously unsuccessful changes without new evidence.\n"
        "- Minimal-change constraint: adjust at most 1-2 thresholds/weights OR add/remove no more than 2 rules this round. Preserve the remaining logic; do NOT rewrite the whole function.\n"
        "- Do not collapse to predicting almost all 1s or 0s merely to fix the listed errors, and do not force a balanced prediction distribution without evidence.\n"
        "- If no defensible improvement is supported, preserve the current logic under the requested new function name and explain why.\n"
        "- In error_analysis, briefly state how the revision follows from previous results and the targeted error pattern, any auxiliary statistical/clinical support used, exact changes (old -> new where applicable), expected metric tradeoffs, and regression risks. Do not claim measured improvement before evaluation.\n\n"
        + "[Evidence Use]\n"
        "- This task is to derive the next version from previous results. The primary evidence for iteration is the historical versions, their reported outcomes, the current training error analysis, and the iteration trajectory. Use the medical knowledge table and univariate summary only as auxiliary references when adding, removing, or modifying rules; they must not override this iteration history or independently drive a redesign.\n"
        + _EVIDENCE_GUIDANCE
        + _RULE_CONSTRAINTS
        + "[Output Contract]\n"
        "Return STRICT JSON only, without Markdown fences or surrounding text, containing exactly these fields:\n"
        "{\n"
        f'  "version": "{next_version}",\n'
        '  "error_analysis": "Concise English rationale as specified above",\n'
        '  "new_policy_code": "Complete executable Python function definition as a JSON string"\n'
        "}\n"
        f"Function signature MUST be: def predict_{next_version}(features: dict) -> int:\n"
        "Return only the new function in new_policy_code, not historical code or a patch. Preserve unchanged logic within this complete function.\n"
        "Escape newlines and quotes correctly in JSON strings.\n"
    )
