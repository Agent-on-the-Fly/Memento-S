"""Terminal/uv-pip executor split from skill_executor.py."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional, Set

from core.config import BUILTIN_BRIDGE_SKILLS, WORKSPACE_DIR
from core.utils.path_utils import (
    _find_venv,
    _rewrite_command_paths_for_skill,
    _resolve_dir,
    _shell_command,
    _venv_bin_dir,
)
from core.skill_engine.skill_resolver import _resolve_skill_dir

from ..executor_utils import canonicalize_op_type, parse_bool, parse_int

# ---------------------------------------------------------------------------
# Terminal toolkit (lazy import, moved from core/config.py)
# ---------------------------------------------------------------------------
_TERMINAL_IMPORT_ERROR: Exception | None = None
try:
    from camel.toolkits.terminal_toolkit import utils as terminal_utils
except Exception as exc:  # pragma: no cover - runtime environment dependent
    terminal_utils = None  # type: ignore[assignment]
    _TERMINAL_IMPORT_ERROR = exc


def _coerce_existing_dir(raw_dir: Any) -> Path | None:
    if not isinstance(raw_dir, str) or not raw_dir.strip():
        return None
    path = Path(raw_dir.strip())
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if path.exists() and path.is_dir():
        return path
    return None


def _read_skill_context(raw_ctx: Any) -> tuple[str | None, Path | None]:
    if not isinstance(raw_ctx, dict):
        return None, None
    raw_name = raw_ctx.get("name")
    skill_name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
    return skill_name, _coerce_existing_dir(raw_ctx.get("dir"))


def _extract_skill_context(plan: dict[str, Any]) -> tuple[str | None, Path | None, bool]:
    skill_name, skill_dir = _read_skill_context(plan.get("_skill_context"))
    if skill_dir is None and skill_name:
        skill_dir = _resolve_skill_dir(skill_name)
    prefer_skill_paths = bool(skill_name and skill_name not in BUILTIN_BRIDGE_SKILLS)
    return skill_name, skill_dir, prefer_skill_paths


def _default_workspace_base_dir() -> Path:
    try:
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return WORKSPACE_DIR.resolve()


def _resolve_working_dir_or_workspace(
    raw_working_dir: Any,
    *,
    base_dir: Path | None = None,
) -> tuple[Path, bool]:
    workspace_dir = _default_workspace_base_dir()
    if raw_working_dir is None:
        return workspace_dir, False
    if isinstance(raw_working_dir, str) and not raw_working_dir.strip():
        return workspace_dir, False

    anchor = base_dir.resolve() if isinstance(base_dir, Path) else Path.cwd().resolve()
    resolved = _resolve_dir(anchor, raw_working_dir)
    if resolved.exists() and resolved.is_dir():
        return resolved, False
    return workspace_dir, True


def convert_pip_to_uv(command: str, working_dir: Path) -> str:
    current = working_dir.resolve()
    for _ in range(5):
        if (current / ".venv").exists():
            command = re.sub(
                r"(^|&&\s*|;\s*|\|\s*)(?:[^\s\"']*python(?:\d+(?:\.\d+)*)?)\s+-m\s+uv\s+pip\b",
                r"\1uv pip",
                command,
            )
            command = re.sub(
                r"(^|&&\s*|;\s*|\|\s*)(?:[^\s\"']*python(?:\d+(?:\.\d+)*)?)\s+-m\s+pip\b",
                r"\1uv pip",
                command,
            )
            command = re.sub(r"(^|&&\s*|;\s*|\|\s*)(?:[^\s\"']*/)?pip(?:3)?\s+", r"\1uv pip ", command)
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return command


def _callback(report: list[str], prefix: str):
    def _cb(message: str | None):
        if message:
            report.append(f"{prefix}{message}")

    return _cb


def _resolve_terminal_working_dir(
    base_dir: Path,
    op: dict[str, Any],
    *,
    op_name: str,
    report: list[str],
) -> Path:
    raw_op_working_dir = op.get("working_dir")
    working_dir, op_wd_fallback = _resolve_working_dir_or_workspace(
        raw_op_working_dir,
        base_dir=base_dir,
    )
    if op_wd_fallback:
        report.append(
            f"{op_name} WARN: invalid working_dir={raw_op_working_dir!r}; fallback to {working_dir}"
        )
    return working_dir


def _append_terminal_call_result(
    report: list[str],
    *,
    op_name: str,
    call: Callable[[], Any],
) -> None:
    try:
        result = call()
        report.append(f"{op_name} result: {result}")
    except Exception as exc:
        report.append(f"{op_name} ERR: {exc}")


def _prepare_terminal_env_operation(
    op: dict[str, Any],
    *,
    op_name: str,
    base_dir: Path,
    report: list[str],
) -> tuple[str, Path, Callable[[str | None], None]] | None:
    env_path = op.get("env_path")
    if not env_path:
        report.append(f"{op_name} SKIP: missing env_path")
        return None
    working_dir = _resolve_terminal_working_dir(
        base_dir,
        op,
        op_name=op_name,
        report=report,
    )
    resolved_env_path = str(_resolve_dir(base_dir, str(env_path)))
    cb = _callback(report, f"{op_name}: ")
    return resolved_env_path, working_dir, cb


def execute_terminal_ops(plan: dict[str, Any]) -> str:
    if terminal_utils is None:
        return f"ERR: camel is not available: {_TERMINAL_IMPORT_ERROR}"
    ops = plan.get("ops", [])
    if not isinstance(ops, list) or not ops:
        return "Invalid tool_calls"

    raw_plan_working_dir = plan.get("working_dir")
    base_dir, plan_wd_fallback = _resolve_working_dir_or_workspace(raw_plan_working_dir)
    skill_name, skill_dir, prefer_skill_paths = _extract_skill_context(plan)
    report: list[str] = []
    if plan_wd_fallback:
        report.append(
            f"terminal WARN: invalid working_dir={raw_plan_working_dir!r}; fallback to {base_dir}"
        )

    for op in ops:
        if not isinstance(op, dict):
            report.append("SKIP op (not a dict)")
            continue
        op_type = canonicalize_op_type(op.get("type"))

        if op_type == "run_command":
            command = str(op.get("command") or "").strip()
            if not command:
                report.append("run_command SKIP: missing command")
                continue

            working_dir = _resolve_terminal_working_dir(
                base_dir,
                op,
                op_name="run_command",
                report=report,
            )
            command = convert_pip_to_uv(command, working_dir)
            command = _rewrite_command_paths_for_skill(
                command,
                working_dir=working_dir,
                skill_dir=skill_dir,
                prefer_skill_paths=prefer_skill_paths,
            )

            allowed_commands: Optional[Set[str]] = None
            safe_mode = parse_bool(op.get("safe_mode"), True)
            use_docker_backend = parse_bool(op.get("use_docker_backend"), False)
            timeout = parse_int(op.get("timeout"), 60, minimum=1)

            try:
                is_safe, reason = terminal_utils.check_command_safety(command, allowed_commands)
            except Exception as exc:
                report.append(f"run_command ERR: check_command_safety failed: {exc}")
                continue

            try:
                ok, msg_or_cmd = terminal_utils.sanitize_command(
                    command=command,
                    use_docker_backend=use_docker_backend,
                    safe_mode=safe_mode,
                    working_dir=str(working_dir),
                    allowed_commands=allowed_commands,
                )
            except Exception as exc:
                report.append(f"run_command ERR: sanitize_command failed: {exc}")
                continue

            if not is_safe:
                report.append(f"run_command REFUSED: {reason}")
                continue
            if not ok:
                report.append(f"run_command REFUSED: {msg_or_cmd}")
                continue
            if use_docker_backend:
                report.append("run_command REFUSED: docker backend not supported")
                continue

            venv_dir = _find_venv(working_dir)
            final_cmd = str(msg_or_cmd)
            env = os.environ.copy()
            if venv_dir:
                venv_bin = str(_venv_bin_dir(venv_dir))
                env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
                env["VIRTUAL_ENV"] = str(venv_dir)
            if skill_name:
                env["MEMENTO_SKILL_NAME"] = skill_name
            if skill_dir:
                env["MEMENTO_SKILL_DIR"] = str(skill_dir)

            try:
                proc = subprocess.run(
                    _shell_command(final_cmd),
                    cwd=str(working_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                report.append(f"run_command TIMEOUT after {timeout}s: {msg_or_cmd}")
                continue
            except FileNotFoundError as exc:
                report.append(f"run_command ERR: shell not found: {exc}")
                continue
            except Exception as exc:
                report.append(f"run_command ERR: {exc}")
                continue

            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            if proc.returncode != 0:
                report.append(f"run_command ERR ({proc.returncode}): {stderr or stdout}")
            else:
                report.append(stdout or stderr or "OK")
            continue

        if op_type == "is_uv_environment":
            try:
                result = terminal_utils.is_uv_environment()
                report.append(f"is_uv_environment: {result}")
            except Exception as exc:
                report.append(f"is_uv_environment ERR: {exc}")
            continue

        if op_type == "ensure_uv_available":
            cb = _callback(report, "ensure_uv_available: ")
            try:
                success, uv_path = terminal_utils.ensure_uv_available(cb)
                report.append(f"ensure_uv_available result: {success} {uv_path or ''}".strip())
            except Exception as exc:
                report.append(f"ensure_uv_available ERR: {exc}")
            continue

        if op_type == "setup_initial_env_with_uv":
            prepared = _prepare_terminal_env_operation(
                op,
                op_name="setup_initial_env_with_uv",
                base_dir=base_dir,
                report=report,
            )
            if prepared is None:
                continue
            resolved_env_path, working_dir, cb = prepared
            uv_path = op.get("uv_path")
            if not uv_path:
                try:
                    success, uv_path = terminal_utils.ensure_uv_available(cb)
                except Exception as exc:
                    report.append(f"setup_initial_env_with_uv ERR: {exc}")
                    continue
                if not success or not uv_path:
                    report.append("setup_initial_env_with_uv ERR: uv not available")
                    continue
            _append_terminal_call_result(
                report,
                op_name="setup_initial_env_with_uv",
                call=lambda: terminal_utils.setup_initial_env_with_uv(
                    resolved_env_path,
                    str(uv_path),
                    str(working_dir),
                    cb,
                ),
            )
            continue

        if op_type == "setup_initial_env_with_venv":
            prepared = _prepare_terminal_env_operation(
                op,
                op_name="setup_initial_env_with_venv",
                base_dir=base_dir,
                report=report,
            )
            if prepared is None:
                continue
            resolved_env_path, working_dir, cb = prepared
            _append_terminal_call_result(
                report,
                op_name="setup_initial_env_with_venv",
                call=lambda: terminal_utils.setup_initial_env_with_venv(
                    resolved_env_path,
                    str(working_dir),
                    cb,
                ),
            )
            continue

        if op_type == "clone_current_environment":
            prepared = _prepare_terminal_env_operation(
                op,
                op_name="clone_current_environment",
                base_dir=base_dir,
                report=report,
            )
            if prepared is None:
                continue
            resolved_env_path, working_dir, cb = prepared
            _append_terminal_call_result(
                report,
                op_name="clone_current_environment",
                call=lambda: terminal_utils.clone_current_environment(
                    resolved_env_path,
                    str(working_dir),
                    cb,
                ),
            )
            continue

        if op_type == "check_nodejs_availability":
            cb = _callback(report, "check_nodejs_availability: ")
            try:
                result = terminal_utils.check_nodejs_availability(cb)
                report.append(f"check_nodejs_availability: {result}")
            except Exception as exc:
                report.append(f"check_nodejs_availability ERR: {exc}")
            continue

        report.append(f"unknown op: {op_type}")

    return "\n".join(report) if report else "OK"


def run_uv_pip(
    args: list[str],
    working_dir: Path,
    venv_dir: Path,
) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = f"{_venv_bin_dir(venv_dir)}{os.pathsep}{env.get('PATH', '')}"
    cmd = ["uv", "pip"] + args
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", "uv command not found. Please install uv first."
    except Exception as exc:
        return -1, "", str(exc)


def execute_uv_pip_ops(plan: dict[str, Any]) -> str:
    ops = plan.get("ops", [])
    if not isinstance(ops, list) or not ops:
        return "Invalid tool_calls"
    report: list[str] = []
    raw_working_dir = plan.get("working_dir")
    working_dir, wd_fallback = _resolve_working_dir_or_workspace(raw_working_dir)
    if wd_fallback:
        report.append(
            f"uv-pip-install WARN: invalid working_dir={raw_working_dir!r}; fallback to {working_dir}"
        )
    venv_dir = _find_venv(working_dir)
    if not venv_dir:
        report.append(f"ERR: No .venv or venv found in {working_dir} or parent directories")
        return "\n".join(report)
    report.append(f"Using venv: {venv_dir}")

    for op in ops:
        if not isinstance(op, dict):
            report.append("SKIP op (not a dict)")
            continue
        op_type = canonicalize_op_type(op.get("type"))

        if op_type == "check":
            package = str(op.get("package", "")).strip()
            if not package:
                report.append("check SKIP: missing package name")
                continue
            returncode, stdout, _stderr = run_uv_pip(["show", package], working_dir, venv_dir)
            if returncode == 0:
                version = "unknown"
                for line in stdout.split("\n"):
                    if line.startswith("Version:"):
                        version = line.split(":", 1)[1].strip()
                        break
                report.append(f"check OK: {package} is installed (version {version})")
            else:
                report.append(f"check: {package} is NOT installed")
            continue

        if op_type == "install":
            package = str(op.get("package", "")).strip()
            if not package:
                report.append("install SKIP: missing package name")
                continue
            extras = str(op.get("extras", "")).strip()
            pkg_spec = f"{package}{extras}" if extras else package
            returncode, stdout, stderr = run_uv_pip(["install", pkg_spec], working_dir, venv_dir)
            if returncode == 0:
                output = stdout or stderr or "OK"
                if len(output) > 1000:
                    output = output[:1000] + "\n...[truncated]"
                report.append(f"install OK: {pkg_spec}\n{output}")
            else:
                report.append(f"install ERR: {pkg_spec}\n{stderr or stdout}")
            continue

        if op_type == "list":
            returncode, stdout, stderr = run_uv_pip(["list"], working_dir, venv_dir)
            if returncode == 0:
                output = stdout or "No packages installed"
                if len(output) > 3000:
                    output = output[:3000] + "\n...[truncated]"
                report.append(f"list OK:\n{output}")
            else:
                report.append(f"list ERR: {stderr or stdout}")
            continue

        report.append(f"unknown op: {op_type}")

    return "\n".join(report) if report else "OK"


__all__ = [
    "convert_pip_to_uv",
    "execute_terminal_ops",
    "run_uv_pip",
    "execute_uv_pip_ops",
]

