
from __future__ import annotations

import ast

from core.config.logging import get_logger
from .parsing import parse_code

logger = get_logger(__name__)


def find_entry_function(
    code: str,
    skill_name: str = "",
    *,
    tree: ast.Module | None = None,
) -> str | None:
    if tree is None:
        tree = parse_code(code)
    if tree is None:
        return None

    functions: list[str] = [
        node.name
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]
    if not functions:
        return None

    if skill_name:
        skill_snake = skill_name.replace("-", "_").lower()
        skill_kebab = skill_name.replace("_", "-").lower()
        for fn in functions:
            if fn.lower() in (skill_snake, skill_kebab, skill_name.lower()):
                return fn

    for fn in functions:
        if fn in ("main", "run", "execute"):
            return fn

    return functions[0]


def extract_parameters(
    code: str,
    skill_name: str = "",
    *,
    tree: ast.Module | None = None,
) -> dict[str, dict]:
    if not code or not code.strip():
        return {}

    if tree is None:
        tree = parse_code(code)
    if tree is None:
        return {}

    func_name = find_entry_function(code, skill_name, tree=tree)
    if not func_name:
        return {}

    func_node = next(
        (
            node for node in ast.iter_child_nodes(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
        ),
        None,
    )
    if func_node is None:
        return {}

    args = func_node.args
    all_args = args.args
    defaults = args.defaults
    num_no_default = len(all_args) - len(defaults)

    params: dict[str, dict] = {}
    for i, arg in enumerate(all_args):
        if arg.arg in ("self", "cls"):
            continue

        param_info: dict = {}
        if arg.annotation:
            try:
                param_info["type"] = ast.unparse(arg.annotation)
            except (ValueError, TypeError):
                param_info["type"] = "Any"
        else:
            param_info["type"] = "Any"

        default_idx = i - num_no_default
        if 0 <= default_idx < len(defaults):
            param_info["required"] = False
            default_node = defaults[default_idx]
            try:
                param_info["default"] = ast.literal_eval(default_node)
            except (ValueError, SyntaxError, TypeError):
                try:
                    param_info["default"] = ast.unparse(default_node)
                except (ValueError, TypeError):
                    param_info["default"] = "..."
        else:
            param_info["required"] = True

        params[arg.arg] = param_info

    if args.vararg:
        params[f"*{args.vararg.arg}"] = {"type": "Any", "required": False, "variadic": True}
    if args.kwarg:
        params[f"**{args.kwarg.arg}"] = {"type": "Any", "required": False, "variadic": True}

    if params:
        logger.debug("Extracted parameters for '%s': %s", skill_name or func_name, list(params.keys()))

    return params
