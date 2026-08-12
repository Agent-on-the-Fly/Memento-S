# SPDX-License-Identifier: Apache-2.0
"""Unified tool-argument processing pipeline.

Stages:
1) map_args: schema-based normalization + structured warnings
2) enrich_args: inject context-derived args (work_dir)
3) rewrite_paths: resolve path-like arguments using ToolContext
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any

from tools import get_registry
from core.skill.execution.tool_context import (
    RuntimeToolContext,
    ToolContext,
    PATH_LIKE_KEYS,
    PathResolutionRejected,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def _tool_schema(tool_name: str) -> dict | None:
    """Get tool schema by name from ToolRegistry."""
    schemas = get_registry().get_schemas_by_names([tool_name])
    return schemas[0] if schemas else None


def _get_tool_schema_props(schema: dict | None) -> dict[str, Any]:
    """Extract properties from a tool schema (handles OpenAI-wrapped format).

    Registry returns OpenAI function-calling format:
        {"type": "function", "function": {"parameters": {"properties": {...}}}}
    We need to access schema["function"]["parameters"]["properties"].
    """
    if not schema:
        return {}
    # Unwrap OpenAI-wrapped schema
    if "function" in schema:
        params = schema.get("function", {}).get("parameters", {})
        return params.get("properties", {})
    return schema.get("properties", {})


def _get_tool_schema_required(schema: dict | None) -> list[str]:
    """Extract required fields from a tool schema (handles OpenAI-wrapped format)."""
    if not schema:
        return []
    if "function" in schema:
        params = schema.get("function", {}).get("parameters", {})
        return params.get("required", [])
    return schema.get("required", [])


def _extract_final_cd_target_from_token(token: str) -> str | None:
    """Extract the cd target path from a bash token that starts with 'cd '."""
    stripped = token.strip()
    if not stripped.startswith("cd "):
        return None
    path_part = stripped[3:].strip()  # Remove "cd " prefix
    if not path_part:
        return None
    # Remove surrounding quotes
    if (path_part.startswith("'") and path_part.endswith("'")) or (
        path_part.startswith('"') and path_part.endswith('"')
    ):
        path_part = path_part[1:-1]
    if path_part.startswith("-"):
        return None
    return path_part


def _rewrite_bash_paths(args: dict, context: RuntimeToolContext) -> dict:
    """Rewrite absolute paths in bash command text to be within root_dir.

    Shell operators (|, >, &&, ;), glob characters (*, ?),
    and all non-path tokens are preserved verbatim.
    """
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return args

    try:
        tokens = shlex.split(command)
    except ValueError:
        return args

    replacements: list[tuple[str, str]] = []
    rewritten_cd_targets: set[str] = set()

    for tok in tokens:
        if not tok.startswith("/"):
            continue

        cd_target = _extract_final_cd_target_from_token(tok)
        if cd_target is not None:
            # cd target: rewrite to root_dir
            try:
                resolved = str(context.resolve_path(cd_target))
                if resolved != cd_target:
                    replacements.append((cd_target, resolved))
                    rewritten_cd_targets.add(cd_target)
            except (ValueError, PermissionError):
                pass
        else:
            # Non-cd absolute path token: delegate to resolve_path
            # resolve_path handles: rewrite if common workspace parent,
            # allow if system path, raise if completely unrelated.
            try:
                resolved = str(context.resolve_path(tok))
                if resolved != tok:
                    replacements.append((tok, resolved))
            except (ValueError, PermissionError):
                # Cannot resolve — pass through, let bash/sandbox enforce boundary.
                pass

    if not replacements:
        return args

    new_command = command
    for original, resolved in replacements:
        new_command = new_command.replace(original, resolved)

    new_args = dict(args)
    new_args["command"] = new_command
    return new_args


def _maybe_rewrite_bash_input(args: dict) -> dict:
    """Extract stdin from bash command if needed."""
    if not isinstance(args, dict):
        return args
    if args.get("stdin") is not None:
        return args

    command = args.get("command")
    if not isinstance(command, str) or "--input" not in command:
        return args

    def _extract_quoted_payload(cmd: str) -> tuple[str, str] | None:
        for quote in ("'", '"'):
            token = f"--input {quote}"
            idx = cmd.find(token)
            if idx == -1:
                continue
            start = idx + len(token)
            end = cmd.find(quote, start)
            if end == -1:
                continue
            payload = cmd[start:end]
            new_cmd = cmd[:idx] + "--input -" + cmd[end + 1 :]
            return new_cmd, payload
        return None

    extracted = _extract_quoted_payload(command)
    if not extracted:
        return args

    new_cmd, payload = extracted
    new_args = dict(args)
    new_args["command"] = new_cmd
    new_args["stdin"] = payload
    return new_args


def _maybe_resolve_tool_paths(
    args: dict,
    skill_root: Path,
    tool_name: str,
) -> dict:
    """Resolve relative paths against skill_root if they exist."""
    if not isinstance(args, dict):
        return args

    try:
        max_name_len = os.pathconf(str(skill_root), "PC_NAME_MAX")
    except (AttributeError, ValueError, OSError):
        max_name_len = 255

    def _is_relative_path(value: str) -> bool:
        if not value or not isinstance(value, str):
            return False
        if "\n" in value or "\r" in value or "\t" in value:
            return False
        if len(value) > max_name_len:
            return False
        if Path(value).is_absolute() or value.startswith("~"):
            return False
        return True

    def _resolve_candidate(value: str) -> str | None:
        try:
            candidate = skill_root / value
            return str(candidate) if candidate.exists() else None
        except OSError:
            return None

    new_args = dict(args)

    if tool_name == "bash":
        command = new_args.get("command")
        if isinstance(command, str) and command.strip():
            segments = re.split(r"(&&|;|\|\|)", command)
            changed = False

            for i, seg in enumerate(segments):
                stripped = seg.strip()
                if not stripped or stripped in {"&&", ";", "||"}:
                    continue

                try:
                    parts = shlex.split(stripped)
                except ValueError:
                    continue

                if not parts:
                    continue

                first = parts[0]
                if _is_relative_path(first):
                    resolved = _resolve_candidate(first)
                    if resolved:
                        segments[i] = seg.replace(first, shlex.quote(resolved), 1)
                        changed = True
                        continue

                if len(parts) >= 2 and parts[0] in {"python", "python3"}:
                    script_arg = parts[1]
                    if _is_relative_path(script_arg):
                        resolved = _resolve_candidate(script_arg)
                        if resolved:
                            new_cmd = (
                                "bash"
                                if str(resolved).endswith(".sh")
                                else parts[0]
                            )
                            segments[i] = seg.replace(parts[0], new_cmd, 1).replace(
                                script_arg, shlex.quote(resolved), 1
                            )
                            changed = True

            if changed:
                new_args["command"] = "".join(segments)
        return new_args

    skip_keys = {
        "base_dir",
        "work_dir",
        "content",
        "stdin",
        "text",
        "data",
        "body",
    }
    for key, value in args.items():
        if key in skip_keys:
            continue
        if isinstance(value, str) and _is_relative_path(value):
            resolved = _resolve_candidate(value)
            if resolved:
                new_args[key] = resolved

    return new_args


class ToolArgsProcessor:
    """Process tool arguments in three explicit stages."""

    def process(
        self,
        *,
        tool_name: str,
        raw_args: dict,
        props: dict,
        context: RuntimeToolContext,
    ) -> tuple[dict, list[dict[str, Any]]]:
        """Run map -> enrich -> rewrite pipeline in one call."""
        mapped_args, warnings = self.map_args(tool_name, raw_args)
        logger.debug(
            f"[ToolArgsProcessor] stage1_map: tool={tool_name}, "
            f"keys={list(mapped_args.keys())}"
        )

        enriched_args = self.enrich_args(
            tool_name=tool_name,
            args=mapped_args,
            props=props,
            context=context,
        )
        logger.debug(
            f"[ToolArgsProcessor] stage2_enrich: tool={tool_name}, "
            f"keys={list(enriched_args.keys())}, "
            f"skill_name={enriched_args.get('skill_name')}"
        )

        try:
            rewritten_args = self.rewrite_paths(
                tool_name=tool_name,
                args=enriched_args,
                context=context,
            )
        except PathResolutionRejected:
            # Hallucinated path segment — re-raise so the tool call fails
            # with a clear error instead of silently using the wrong path.
            raise

        logger.debug(
            f"[ToolArgsProcessor] stage3_rewrite: tool={tool_name}, "
            f"keys={list(rewritten_args.keys())}"
        )

        return rewritten_args, warnings

    # -------------------- stage 1: mapping/normalization --------------------
    def map_args(
        self,
        tool_name: str,
        raw_args: dict,
    ) -> tuple[dict, list[dict[str, Any]]]:
        warnings: list[dict[str, Any]] = []
        schema = _tool_schema(tool_name)
        if not schema:
            return raw_args, warnings

        props = _get_tool_schema_props(schema)
        required = _get_tool_schema_required(schema)

        auto_fillable_params = {
            "skill_name", "session_id", "source_dir", "work_dir",
            "workspace_dir", "dir_path", "path",
        }

        for req_param in required:
            if req_param not in raw_args:
                if req_param in auto_fillable_params:
                    continue
                warning = {
                    "type": "missing_required_param",
                    "tool": tool_name,
                    "param": req_param,
                    "message": f"Missing required parameter '{req_param}' for tool '{tool_name}'",
                }
                warnings.append(warning)
                logger.warning(warning["message"])

        normalized: dict[str, Any] = {}

        for param_name, param_info in props.items():
            if param_name in raw_args:
                value = raw_args[param_name]
                param_type = param_info.get("type")

                if param_type == "integer" and value is not None:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        warning = {
                            "type": "invalid_integer",
                            "tool": tool_name,
                            "param": param_name,
                            "value": value,
                            "message": f"Cannot convert '{value}' to integer for '{param_name}'",
                        }
                        warnings.append(warning)
                        logger.warning(warning["message"])
                        default = param_info.get("default")
                        if default is not None:
                            value = default
                        else:
                            continue

                elif param_type == "boolean":
                    parsed_bool, bool_warning = self._parse_boolean(
                        value,
                        tool_name,
                        param_name,
                    )
                    if bool_warning:
                        warnings.append(bool_warning)
                        logger.warning(bool_warning["message"])
                    if parsed_bool is None:
                        default = param_info.get("default")
                        if default is not None:
                            value = default
                        else:
                            continue
                    else:
                        value = parsed_bool

                normalized[param_name] = value
            elif param_name in required:
                continue
            else:
                default = param_info.get("default")
                if default is not None:
                    normalized[param_name] = default

        return normalized, warnings

    @staticmethod
    def _parse_boolean(
        value: Any,
        tool_name: str,
        param_name: str,
    ) -> tuple[bool | None, dict[str, Any] | None]:
        if isinstance(value, bool):
            return value, None

        if isinstance(value, int):
            if value in (0, 1):
                return bool(value), None
            return None, {
                "type": "invalid_boolean",
                "tool": tool_name,
                "param": param_name,
                "value": value,
                "message": f"Invalid boolean value '{value}' for '{param_name}'",
            }

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True, None
            if normalized in {"false", "0", "no", "n", "off"}:
                return False, None
            return None, {
                "type": "invalid_boolean",
                "tool": tool_name,
                "param": param_name,
                "value": value,
                "message": f"Invalid boolean value '{value}' for '{param_name}'",
            }

        return None, {
            "type": "invalid_boolean",
            "tool": tool_name,
            "param": param_name,
            "value": value,
            "message": f"Invalid boolean value '{value}' for '{param_name}'",
        }

    # -------------------- stage 2: enrichment --------------------
    def enrich_args(
        self,
        *,
        tool_name: str,
        args: dict,
        props: dict,
        context: RuntimeToolContext,
    ) -> dict:
        """Enrich args with context-derived values if needed."""
        new_args = dict(args) if isinstance(args, dict) else args

        if isinstance(new_args, dict):
            self._auto_fill_required_args(new_args, tool_name, context)

            # Only inject work_dir if the tool actually accepts it.
            schema = _tool_schema(tool_name)
            if schema:
                accepted = set(_get_tool_schema_props(schema).keys())
                if "work_dir" not in new_args and "work_dir" in accepted and (
                    "command" in new_args or "code" in new_args
                ):
                    new_args["work_dir"] = str(context.root_dir)

                if context.skill_root and "source_dir" in accepted:
                    if "source_dir" not in new_args:
                        new_args["source_dir"] = str(context.skill_root)

        return new_args

    def _auto_fill_required_args(
        self,
        args: dict,
        tool_name: str,
        context: RuntimeToolContext,
    ) -> None:
        """Automatically fill missing context-derived params only if the tool accepts them."""
        # Pull tool schema so we only inject params the tool actually declares.
        schema = _tool_schema(tool_name)
        if not schema:
            return

        accepted_params = set(_get_tool_schema_props(schema).keys())

        param_sources = {
            "skill_name": lambda ctx: ctx.skill_name,
            "session_id": lambda ctx: ctx.session_id,
            "source_dir": lambda ctx: str(ctx.skill_root) if ctx.skill_root else None,
            "work_dir": lambda ctx: str(ctx.root_dir),
            "workspace_dir": lambda ctx: str(ctx.workspace_dir),
            "dir_path": lambda ctx: str(ctx.root_dir),
            "path": lambda ctx: str(ctx.root_dir),
        }

        for param_name, source_fn in param_sources.items():
            if param_name in args or param_name not in accepted_params:
                continue
            value = source_fn(context)
            if value:
                args[param_name] = value

    # -------------------- stage 3: rewrite --------------------
    def rewrite_paths(
        self,
        *,
        tool_name: str,
        args: dict,
        context: RuntimeToolContext,
    ) -> dict:
        """Resolve all path-like arguments using RuntimeToolContext.resolve_path()."""
        new_args = dict(args) if isinstance(args, dict) else args

        # Stage 3.2: Resolve path-like arguments
        for key in PATH_LIKE_KEYS:
            if key not in new_args or not isinstance(new_args[key], str):
                continue

            resolved_path = new_args[key]
            try:
                _resolved = context.resolve_path(resolved_path)
                new_args[key] = str(_resolved)
                if key == "image":
                    logger.info(
                        "Image path resolved: tool='{}' raw='{}' resolved='{}'",
                        tool_name,
                        resolved_path,
                        new_args[key],
                    )
            except PathResolutionRejected:
                # Hallucinated path segment — re-raise so the tool call fails
                # with a clear error instead of silently using the wrong path.
                raise
            except (ValueError, PermissionError) as e:
                logger.warning(
                    "Path resolution rejected for tool '{}' arg '{}': raw='{}' error='{}'",
                    tool_name,
                    key,
                    resolved_path,
                    e,
                )

        # Stage 3.3: Rewrite bash command paths
        # Non-cd absolute paths are now handled by resolve_path in Stage 3.2,
        # so this only handles cd target rewriting (already delegated to resolve_path).
        if tool_name == "bash" and "command" in new_args:
            new_args = _rewrite_bash_paths(new_args, context)

        # Stage 3.4: Resolve skill-relative paths
        if context.skill_root:
            new_args = _maybe_resolve_tool_paths(
                new_args,
                context.skill_root,
                tool_name,
            )

        return new_args
