from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedProposal:
    version: str
    error_analysis: str
    new_policy_code: str


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def validate_python_syntax(code: str) -> None:
    ast.parse(code)


def _collect_bound_names(node: ast.AST, bound: set[str]) -> None:
    """Collect names a node binds in its own scope (assignments, imports, args).

    Function/class bodies are not descended into, because their locals belong
    to a nested scope.
    """
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        bound.add(node.id)
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            bound.add(alias.asname or alias.name.split(".")[0])
        return
    for child in ast.iter_child_nodes(node):
        _collect_bound_names(child, bound)


def _check_function_undefined_names(fn: ast.FunctionDef, module_scope: set[str]) -> None:
    """Reject a function that reads a name it never binds.

    MHL executes generated rule functions on rows the generator never saw, so a
    NameError may hide in a branch that is not exercised during validation.
    This static check catches the common failure mode where the LLM writes a
    bare feature name (e.g. 'testosterone') instead of reading it from the
    'features' dict.
    """
    bound: set[str] = set(module_scope)
    for arg in (
        list(fn.args.posonlyargs)
        + list(fn.args.args)
        + list(fn.args.kwonlyargs)
        + ([fn.args.vararg] if fn.args.vararg is not None else [])
        + ([fn.args.kwarg] if fn.args.kwarg is not None else [])
    ):
        bound.add(arg.arg)

    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.Lambda):
            for arg in (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
                + ([node.args.vararg] if node.args.vararg is not None else [])
                + ([node.args.kwarg] if node.args.kwarg is not None else [])
            ):
                bound.add(arg.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
            bound.add(node.name)
            for arg in (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
                + ([node.args.vararg] if node.args.vararg is not None else [])
                + ([node.args.kwarg] if node.args.kwarg is not None else [])
            ):
                bound.add(arg.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])

    allowed = bound | set(dir(builtins))
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in allowed:
            raise ValueError(
                f"undefined name '{node.id}' at line {node.lineno} in {fn.name}; "
                "feature values must be read from the 'features' dict "
                "(e.g. features['col'] or features.get('col')) instead of bare variable names"
            )


def validate_undefined_names(code: str) -> None:
    """Validate that every function in *code* only reads names it defines."""
    try:
        module = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"invalid Python syntax: {exc}") from exc

    module_scope: set[str] = set()
    for node in module.body:
        _collect_bound_names(node, module_scope)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_scope.add(node.name)

    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_function_undefined_names(node, module_scope)


def extract_function_name(code: str) -> str | None:
    try:
        module = ast.parse(code)
    except Exception:
        return None
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None

