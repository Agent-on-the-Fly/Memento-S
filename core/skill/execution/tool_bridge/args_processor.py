"""Unified tool-argument processing pipeline.

Stages:
1) map_args: schema-based normalization + structured warnings
2) enrich_args: inject context-derived args (allow_roots/work_dir)
3) rewrite_paths: resolve path-like arguments using ToolContext
4) sanitize_for_execution: strip orchestration-only metadata before runner
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any

from core.shared.tools_facade import get_tool_schema
from core.skill.execution.tool_bridge.context import ToolContext, PATH_LIKE_KEYS
from utils.logger import get_logger

logger = get_logger(__name__)


class ToolArgsProcessor:
    """Process tool arguments in three explicit stages."""

    def process(
        self,
        *,
        tool_name: str,
        raw_args: dict,
        props: dict,
        context: ToolContext,
    ) -> tuple[dict, list[dict[str, Any]]]:
        """Run map -> enrich -> rewrite pipeline in one call."""
        mapped_args, warnings = self.map_args(tool_name, raw_args)
        enriched_args = self.enrich_args(
            args=mapped_args,
            props=props,
            context=context,
        )
        rewritten_args = self.rewrite_paths(
            tool_name=tool_name,
            args=enriched_args,
            context=context,
        )
        sanitized_args = self.sanitize_for_execution(rewritten_args)
        return sanitized_args, warnings

    # -------------------- stage 1: mapping/normalization --------------------
    def map_args(
        self,
        tool_name: str,
        raw_args: dict,
    ) -> tuple[dict, list[dict[str, Any]]]:
        warnings: list[dict[str, Any]] = []
        schema = get_tool_schema(tool_name)
        if not schema:
            return raw_args, warnings

        props = schema.get("properties", {})
        required = schema.get("required", [])

        for req_param in required:
            if req_param not in raw_args:
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
        args: dict,
        props: dict,
        context: ToolContext,
    ) -> dict:
        """Enrich args with context-derived values if needed."""
        new_args = dict(args) if isinstance(args, dict) else args

        # Unify execution root to per-run root_dir (@ROOT) for runtime tools.
        if isinstance(new_args, dict):
            if "work_dir" not in new_args and (
                "command" in new_args or "code" in new_args
            ):
                new_args["work_dir"] = str(context.root_dir)

        return new_args

    # -------------------- stage 3: rewrite --------------------
    def rewrite_paths(
        self,
        *,
        tool_name: str,
        args: dict,
        context: ToolContext,
    ) -> dict:
        """Resolve all path-like arguments using ToolContext.resolve_path().

        This is the single entry point for path resolution, delegating to
        ToolContext which handles aliases, validation, and coercion.
        """
        new_args = dict(args) if isinstance(args, dict) else args

        # Resolve path-like arguments
        for key in PATH_LIKE_KEYS:
            if key in new_args and isinstance(new_args[key], str):
                try:
                    new_args[key] = str(context.resolve_path(new_args[key]))
                except (ValueError, PermissionError) as e:
                    # Keep original arg so validator can return structured, user-facing error.
                    logger.warning(
                        "Path resolution rejected for tool '{}' arg '{}': {}",
                        tool_name,
                        key,
                        e,
                    )

        # Special handling for bash command
        if tool_name == "bash" and "command" in new_args:
            new_args = self._rewrite_bash_paths(new_args, context)

        # Resolve skill-relative paths if skill_root is available
        if context.skill_root:
            new_args = self._maybe_resolve_tool_paths(
                new_args,
                context.skill_root,
                tool_name,
            )

        return new_args

    @staticmethod
    def _rewrite_bash_paths(args: dict, context: ToolContext) -> dict:
        """Rewrite absolute paths in bash command to be within root_dir.

        Only performs targeted string replacements on path tokens that
        actually change after resolution.  Shell operators (``|``,
        ``>``, ``&&``, ``;``), glob characters (``*``, ``?``), and all
        other non-path tokens are preserved verbatim.
        """
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return args

        try:
            tokens = shlex.split(command)
        except ValueError:
            return args

        # Collect (original_path, resolved_path) pairs for tokens that
        # are absolute paths and actually change after resolution.
        replacements: list[tuple[str, str]] = []
        for tok in tokens:
            if not tok.startswith("/"):
                continue
            try:
                resolved = str(context.resolve_path(tok))
            except (ValueError, PermissionError):
                continue
            if resolved != tok:
                replacements.append((tok, resolved))

        if not replacements:
            return args

        # Apply replacements directly on the original command string
        # so that shell operators, globs, and quoting are preserved.
        new_command = command
        for original, resolved in replacements:
            # Replace both bare and quoted forms of the original path
            new_command = new_command.replace(original, resolved)

        new_args = dict(args)
        new_args["command"] = new_command
        return new_args

    @staticmethod
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

    @staticmethod
    def sanitize_for_execution(args: dict) -> dict:
        """Remove orchestration metadata before tool invocation."""
        if not isinstance(args, dict):
            return args
        return {
            key: value
            for key, value in args.items()
            if key not in ("skill_name", "session_id", "task_id")
        }

    @staticmethod
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
                            # Replace only the command token in the
                            # original segment, preserving all other
                            # tokens (pipes, globs, etc.) verbatim.
                            segments[i] = seg.replace(first, shlex.quote(resolved), 1)
                            changed = True
                            continue

                    if len(parts) >= 2 and parts[0] in {"python", "python3"}:
                        script_arg = parts[1]
                        if _is_relative_path(script_arg):
                            resolved = _resolve_candidate(script_arg)
                            if resolved:
                                # Replace command + script path in the
                                # original segment string.
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
