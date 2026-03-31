"""Pre-execute gate checks for skill execution."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any

from core.shared.dependency_aliases import (
    normalize_dependency_name,
    normalize_dependency_spec,
    strip_version_extras,
)
from middleware.utils.environment import get_config_env_vars

from core.skill.execution.policy.types import PolicyDecision, PolicyStage
from core.skill.schema import Skill
from utils.logger import get_logger

logger = get_logger(__name__)

def _deny(reason: str, detail: dict[str, Any] | None = None) -> PolicyDecision:
    return PolicyDecision(
        allowed=False,
        stage=PolicyStage.PRE_EXECUTE,
        reason=reason,
        detail=detail,
    )


def _allow() -> PolicyDecision:
    return PolicyDecision(
        allowed=True,
        stage=PolicyStage.PRE_EXECUTE,
        reason="",
    )


def _get_available_keys() -> set[str]:
    keys: set[str] = set()

    try:
        for key, value in get_config_env_vars().items():
            if value and "KEY" in key.upper():
                keys.add(key.upper())
    except Exception:
        pass

    for key, value in os.environ.items():
        if "KEY" in key.upper() and value:
            keys.add(key.upper())

    return keys


def _check_skill_keys(skill: Skill) -> tuple[bool, list[str]]:
    if not skill.required_keys:
        return True, []

    available = _get_available_keys()
    required = [str(k).upper() for k in skill.required_keys if str(k).strip()]
    missing = [k for k in required if k not in available]
    return len(missing) == 0, missing


def check_api_keys(skill: Skill) -> PolicyDecision:
    keys_ok, missing_keys = _check_skill_keys(skill)
    if keys_ok:
        return _allow()

    return _deny(
        reason=f"Missing API keys: {', '.join(missing_keys)}",
        detail={
            "error_type": "environment_error",
            "category": "environment",
            "retryable": False,
            "hint": "Configure required keys in environment/config before retry.",
            "missing_keys": missing_keys,
        },
    )


def check_skill_structure(skill: Skill) -> PolicyDecision:
    if not skill.is_playbook:
        return _allow()

    if not skill.source_dir:
        return _deny(
            "Playbook skill missing source_dir; cannot resolve scripts directory.",
            detail={"error_type": "resource_missing", "category": "skill_structure"},
        )

    skill_root = Path(skill.source_dir)
    if not skill_root.exists() or not skill_root.is_dir():
        return _deny(
            f"Skill source_dir not found or invalid: {skill.source_dir}",
            detail={"error_type": "resource_missing", "category": "skill_structure"},
        )

    scripts_dir = skill_root / "scripts"
    entry_script = (skill.entry_script or "").strip()

    if entry_script:
        entry_candidates = [
            scripts_dir / entry_script,
            scripts_dir / f"{entry_script}.py",
            skill_root / entry_script,
            skill_root / f"{entry_script}.py",
        ]
        if not any(p.exists() and p.is_file() for p in entry_candidates):
            return _deny(
                f"Configured entry_script '{entry_script}' does not exist in skill directory.",
                detail={
                    "error_type": "resource_missing",
                    "category": "skill_structure",
                },
            )

    if not entry_script:
        has_script = False
        if scripts_dir.exists() and scripts_dir.is_dir():
            for p in scripts_dir.glob("*"):
                if p.is_file() and p.suffix in {".py", ".sh", ".js", ".ts"}:
                    has_script = True
                    break
        if not has_script:
            return _deny(
                "Playbook skill has no entry_script and no executable scripts under scripts/.",
                detail={
                    "error_type": "resource_missing",
                    "category": "skill_structure",
                },
            )

    return _allow()


def check_allowed_tools(skill: Skill) -> PolicyDecision:
    if not skill.allowed_tools:
        return _allow()

    try:
        from core.shared.tools_facade import BUILTIN_TOOL_SCHEMAS

        available = {
            t.get("function", {}).get("name", "")
            for t in BUILTIN_TOOL_SCHEMAS
            if isinstance(t, dict)
        }
    except Exception as e:
        logger.warning("Failed to inspect BUILTIN_TOOL_SCHEMAS: {}", e)
        return _allow()

    invalid = sorted({tool for tool in skill.allowed_tools if tool not in available})
    if not invalid:
        return _allow()

    return _deny(
        reason=f"Skill allowed_tools contains unknown tool(s): {', '.join(invalid)}",
        detail={
            "error_type": "input_invalid",
            "category": "allowed_tools",
            "retryable": False,
            "hint": "Fix skill metadata allowed_tools to match builtin tool names.",
            "invalid_tools": invalid,
        },
    )


def check_input_schema(skill: Skill, params: dict[str, Any] | None) -> PolicyDecision:
    if params is None:
        return _allow()
    if not isinstance(params, dict):
        return _deny(
            "Parameters must be a JSON object.",
            detail={"error_type": "input_invalid", "category": "input"},
        )

    schema = skill.parameters if isinstance(skill.parameters, dict) else None
    required_fields = []
    if schema:
        required = schema.get("required")
        if isinstance(required, list):
            required_fields = [str(x) for x in required if str(x).strip()]

    if not required_fields:
        return _allow()

    missing = [name for name in required_fields if name not in params]
    if missing:
        return _deny(
            f"Missing required parameter(s): {', '.join(missing)}",
            detail={
                "error_type": "input_invalid",
                "category": "input",
                "missing": missing,
            },
        )

    invalid_empty = []
    for name in required_fields:
        value = params.get(name)
        if value is None:
            invalid_empty.append(name)
        elif isinstance(value, str) and not value.strip():
            invalid_empty.append(name)

    if invalid_empty:
        return _deny(
            f"Required parameter(s) cannot be empty: {', '.join(invalid_empty)}",
            detail={
                "error_type": "input_invalid",
                "category": "input",
                "empty": invalid_empty,
            },
        )

    return _allow()


# ---- dependency helpers (migrated from utils/dependencies.py) ----


def parse_dependency(dep: str) -> tuple[str, str, str]:
    """Parse dependency spec into (kind, name, install_spec)."""
    raw = (dep or "").strip()
    if not raw:
        return ("none", "", "")

    lowered = raw.lower()
    if lowered.startswith("cli:"):
        tool = raw.split(":", 1)[1].strip()
        return ("cli", tool, tool)
    if lowered.startswith("pip:"):
        pkg = raw.split(":", 1)[1].strip()
        normalized_pkg = normalize_dependency_spec(pkg)
        if not normalized_pkg:
            return ("none", "", "")
        base = strip_version_extras(normalized_pkg)
        return ("python", base, normalized_pkg)
    if lowered.startswith("py:"):
        mod = raw.split(":", 1)[1].strip()
        normalized_mod = normalize_dependency_name(mod)
        if not normalized_mod:
            return ("none", "", "")
        return ("python", normalized_mod, normalized_mod)

    normalized_raw = normalize_dependency_spec(raw)
    if not normalized_raw:
        return ("none", "", "")

    base = strip_version_extras(normalized_raw)
    if not base:
        return ("none", "", "")

    if base.lower() in {"ffmpeg"}:
        return ("cli", base, base)

    return ("python", base, normalized_raw)


def is_installed(name: str) -> bool:
    """Check if a Python package/module is available."""
    try:
        if importlib.util.find_spec(name) is not None:
            return True
    except (ModuleNotFoundError, ValueError):
        pass

    try:
        importlib.metadata.distribution(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def check_missing_dependencies(dependencies: list[str]) -> list[str]:
    """Check which dependencies are missing.

    Returns original dependency specs to keep install behavior unchanged.
    """
    missing: list[str] = []
    for dep in dependencies:
        kind, name, _ = parse_dependency(dep)
        if kind == "none" or not name:
            continue
        if kind == "cli":
            if shutil.which(name) is None:
                missing.append(dep)
            continue
        if not is_installed(name):
            missing.append(dep)
    return missing


def run_pre_execute_gate(
    skill: Skill,
    params: dict[str, Any] | None = None,
) -> PolicyDecision:
    checks = (
        check_api_keys(skill),
        check_skill_structure(skill),
        check_allowed_tools(skill),
        check_input_schema(skill, params),
    )
    for decision in checks:
        if not decision.allowed:
            return decision
    return _allow()
