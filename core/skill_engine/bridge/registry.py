"""Bridge tool registry extracted from skill_executor.py."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.config import FILESYSTEM_OP_TYPES, TERMINAL_OP_TYPES, UV_PIP_OP_TYPES, WEB_OP_TYPES


@dataclass(frozen=True)
class ToolSchema:
    required: tuple[str, ...] = ()
    typed: dict[str, tuple[type, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    target_skill: str | None
    forward_working_dir: bool = False
    schema: ToolSchema = field(default_factory=ToolSchema)


def _schema(
    *,
    required: tuple[str, ...] = (),
    typed: dict[str, tuple[type, ...]] | None = None,
) -> ToolSchema:
    return ToolSchema(required=required, typed=typed or {})


def build_tool_registry() -> dict[str, ToolSpec]:
    registry: dict[str, ToolSpec] = {"call_skill": ToolSpec(target_skill=None)}

    def _register_many(
        op_types: set[str],
        target_skill: str,
        *,
        forward_working_dir: bool = False,
    ) -> None:
        for op_type in op_types:
            registry[op_type] = ToolSpec(
                target_skill=target_skill,
                forward_working_dir=forward_working_dir,
            )

    _register_many(FILESYSTEM_OP_TYPES, "filesystem", forward_working_dir=True)
    _register_many(TERMINAL_OP_TYPES, "terminal", forward_working_dir=True)
    _register_many(WEB_OP_TYPES, "web-search")
    _register_many(UV_PIP_OP_TYPES, "uv-pip-install", forward_working_dir=True)

    schema_overrides: dict[str, ToolSchema] = {
        "run_command": _schema(
            required=("command",),
            typed={
                "command": (str,),
                "timeout": (int,),
            },
        ),
        "web_search": _schema(
            required=("query",),
            typed={
                "query": (str,),
                "num_results": (int,),
            },
        ),
        "fetch": _schema(
            required=("url",),
            typed={
                "url": (str,),
                "max_length": (int,),
            },
        ),
        "check": _schema(
            required=("package",),
            typed={"package": (str,)},
        ),
        "install": _schema(
            required=("package",),
            typed={
                "package": (str,),
                "extras": (str,),
            },
        ),
        "read_file": _schema(required=("path",), typed={"path": (str,)}),
        "write_file": _schema(required=("path",), typed={"path": (str,)}),
        "edit_file": _schema(required=("path",), typed={"path": (str,)}),
        "replace_text": _schema(required=("path",), typed={"path": (str,)}),
        "append_file": _schema(required=("path",), typed={"path": (str,)}),
        "list_directory": _schema(required=("path",), typed={"path": (str,)}),
        "directory_tree": _schema(required=("path",), typed={"path": (str,)}),
        "create_directory": _schema(required=("path",), typed={"path": (str,)}),
        "mkdir": _schema(required=("path",), typed={"path": (str,)}),
        "delete_file": _schema(required=("path",), typed={"path": (str,)}),
        "file_info": _schema(required=("path",), typed={"path": (str,)}),
        "file_exists": _schema(required=("path",), typed={"path": (str,)}),
        "setup_initial_env_with_uv": _schema(required=("env_path",), typed={"env_path": (str,)}),
        "setup_initial_env_with_venv": _schema(required=("env_path",), typed={"env_path": (str,)}),
        "clone_current_environment": _schema(required=("env_path",), typed={"env_path": (str,)}),
    }

    for tool_name, schema in schema_overrides.items():
        spec = registry.get(tool_name)
        if spec is None:
            continue
        registry[tool_name] = ToolSpec(
            target_skill=spec.target_skill,
            forward_working_dir=spec.forward_working_dir,
            schema=schema,
        )
    return registry


__all__ = [
    "ToolSchema",
    "ToolSpec",
    "build_tool_registry",
]

