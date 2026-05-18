from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedProposal:
    version: str
    error_analysis: str
    new_policy_code: str
    new_tests: list[dict]
    modified_tests: list[dict]


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def validate_python_syntax(code: str) -> None:
    ast.parse(code)


def extract_function_name(code: str) -> str | None:
    try:
        module = ast.parse(code)
    except Exception:
        return None
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None

