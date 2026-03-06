
from __future__ import annotations

import ast
from typing import Any, Dict, Optional


def find_target_function(code: str, func_name: str) -> Optional[ast.FunctionDef]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == func_name:
                return node
            if target is None:
                target = node
    return target


def extract_description(code: str, func_name: str) -> str:
    target = find_target_function(code, func_name)
    if not target:
        return ""

    docstring = ast.get_docstring(target) or ""
    if not docstring:
        return ""

    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped:
            if stripped.lower() in ("args:", "parameters:", "returns:", "raises:", "example:", "examples:"):
                continue
            return stripped

    return ""


def extract_parameters(code: str, func_name: str) -> Dict[str, Any]:
    target = find_target_function(code, func_name)
    if not target:
        return {}

    param_descs = _parse_docstring_params(ast.get_docstring(target) or "")

    args = target.args.args
    defaults = target.args.defaults
    offset = len(args) - len(defaults)

    parameters: Dict[str, Any] = {}
    for i, arg_node in enumerate(args):
        name = arg_node.arg
        if name == "self":
            continue

        param_info: Dict[str, Any] = {}

        if arg_node.annotation:
            param_info["type"] = _annotation_to_str(arg_node.annotation)
        else:
            param_info["type"] = "Any"

        default_idx = i - offset
        if 0 <= default_idx < len(defaults):
            try:
                param_info["default"] = ast.literal_eval(defaults[default_idx])
            except (ValueError, TypeError):
                pass

        param_info["description"] = param_descs.get(name, "")

        parameters[name] = param_info

    return parameters


def _annotation_to_str(annotation: ast.expr) -> str:
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Constant):
        return str(annotation.value)
    if isinstance(annotation, ast.Attribute):
        return f"{_annotation_to_str(annotation.value)}.{annotation.attr}"
    if isinstance(annotation, ast.Subscript):
        base = _annotation_to_str(annotation.value)
        inner = _annotation_to_str(annotation.slice)
        return f"{base}[{inner}]"
    if isinstance(annotation, ast.Tuple):
        parts = [_annotation_to_str(e) for e in annotation.elts]
        return ", ".join(parts)
    return "Any"


def _parse_docstring_params(docstring: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not docstring:
        return result

    in_args = False
    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("args:") or stripped.lower().startswith("parameters:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped.startswith("-") and ":" not in stripped:
                if stripped.endswith(":"):
                    in_args = False
                    continue
            if ":" in stripped:
                key_part, _, desc = stripped.partition(":")
                key_part = key_part.strip().lstrip("-").strip()
                if "(" in key_part:
                    key_part = key_part[: key_part.index("(")].strip()
                if key_part and key_part.isidentifier():
                    result[key_part] = desc.strip()

    return result
