"""Shared path / utility functions extracted from agent.py.

Every public symbol that lived in agent.py and is purely about path
resolution, text helpers, subprocess helpers, or lightweight JSON
parsing is gathered here so other modules can import them without
pulling in the full agent.
"""

from __future__ import annotations

import json
import ntpath
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from core.config import (
    BUILTIN_BRIDGE_SKILLS,
    PROJECT_ROOT,
    SKILL_DYNAMIC_FETCH_TIMEOUT_SEC,
    SKILL_LOCAL_COMMAND_PATH_RE,
    SKILL_LOCAL_DIR_PREFIXES,
)

# ---------------------------------------------------------------------------
# Text truncation helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_chars: int = 4000) -> str:
    """Truncate *text* to *max_chars*, appending a marker when cut."""
    s = str(text or "")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 15] + "...[truncated]"


def _truncate_middle(text: str, max_chars: int = 4000) -> str:
    """Keep head + tail of *text* so the model can still see endings."""
    s = str(text or "")
    if len(s) <= max_chars:
        return s
    marker = f"\n...[truncated {len(s)} chars total]...\n"
    keep = max_chars - len(marker)
    if keep <= 0:
        return marker.strip()
    head = max(0, keep // 2)
    tail = max(0, keep - head)
    return s[:head] + marker + s[-tail:]


def _truncate_text(text: str, max_chars: int = 50000) -> str:
    """Simple head-only truncation (used by web ops)."""
    s = str(text or "")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 15] + "...[truncated]"


# ---------------------------------------------------------------------------
# Stringify / XML helpers
# ---------------------------------------------------------------------------


def _stringify_result(result: Any) -> str:
    """Convert an arbitrary result value to a human-readable string."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception:
        return str(result)


def _xml_escape(text: str) -> str:
    """Minimal XML entity escaping for skill catalog XML."""
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# Directory / path resolution
# ---------------------------------------------------------------------------


def _resolve_dir(base_dir: Path, raw: str | None) -> Path:
    """Resolve a working-directory override relative to *base_dir*.

    Handles Windows-absolute paths on non-Windows hosts (WSL mapping).
    """
    if raw is None:
        return base_dir
    raw_text = str(raw).strip()
    if not raw_text:
        return base_dir
    p = Path(raw_text).expanduser()
    if not p.is_absolute():
        # On non-Windows hosts, keep Windows-absolute paths as-is
        # (or map to /mnt/<drive> in WSL) instead of incorrectly
        # treating them as relative.
        if os.name != "nt" and ntpath.isabs(raw_text):
            translated = _windows_path_to_wsl(raw_text)
            if translated is not None:
                return translated
            return Path(raw_text)
        p = (base_dir / p).resolve()
    else:
        p = p.resolve()
    return p


def _resolve_runtime_path(
    base_dir: Path,
    path_str: str | None,
    *,
    skill_dir: Path | None = None,
    prefer_skill_paths: bool = False,
) -> Path:
    """Resolve a user-supplied path, preferring the skill directory for
    well-known local prefixes (scripts/, references/, …) when
    *prefer_skill_paths* is set.
    """
    if not path_str:
        return base_dir

    raw = str(path_str).strip()
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()

    cwd_target = (base_dir / p).resolve()
    rel = _skill_local_rel_path(raw)
    if not rel or skill_dir is None or not prefer_skill_paths:
        return cwd_target

    skill_target = (skill_dir / rel).resolve()
    skill_exists = skill_target.exists()
    if skill_exists:
        return skill_target
    return cwd_target


# Convenience alias used throughout the codebase.
_resolve_path = _resolve_runtime_path


def _skill_local_rel_path(path_str: str | None) -> str | None:
    """Return the cleaned relative path if it starts with one of the
    known skill-local directory prefixes, else ``None``."""
    if not isinstance(path_str, str):
        return None
    cleaned = path_str.strip().replace("\\", "/")
    if not cleaned or cleaned.startswith("/"):
        return None
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    root = cleaned.split("/", 1)[0]
    if root not in SKILL_LOCAL_DIR_PREFIXES:
        return None
    return cleaned


def _rewrite_command_paths_for_skill(
    command: str,
    *,
    working_dir: Path,
    skill_dir: Path | None,
    prefer_skill_paths: bool,
) -> str:
    """Rewrite ``scripts/…`` style tokens inside a shell command so they
    point at the skill directory when the corresponding file exists there.
    """
    if not command or skill_dir is None:
        return command

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        rewritten = _resolve_runtime_path(
            working_dir,
            token,
            skill_dir=skill_dir,
            prefer_skill_paths=prefer_skill_paths,
        )
        if rewritten == (working_dir / Path(token)).resolve():
            return token
        return str(rewritten)

    return SKILL_LOCAL_COMMAND_PATH_RE.sub(_replace, command)


def _windows_path_to_wsl(raw_path: str) -> Path | None:
    """Translate a Windows-absolute path (``C:\\foo``) to a WSL mount
    path (``/mnt/c/foo``).  Returns ``None`` if the mount point does
    not exist on this host.
    """
    match = re.match(r"^([A-Za-z]):[\\/ ]*(.*)$", str(raw_path or "").strip())
    if not match:
        return None
    drive = match.group(1).lower()
    tail = match.group(2).replace("\\", "/").lstrip("/")
    root = Path(f"/mnt/{drive}")
    if not root.exists():
        return None
    translated = (root / tail) if tail else root
    return translated.resolve()


# ---------------------------------------------------------------------------
# Shell / subprocess helpers
# ---------------------------------------------------------------------------


def _shell_command(command: str) -> list[str]:
    """Return the argv list to execute *command* via the system shell."""
    if os.name == "nt":
        comspec = (os.environ.get("COMSPEC") or "").strip() or "cmd.exe"
        return [comspec, "/d", "/s", "/c", command]
    bash_path = shutil.which("bash")
    if bash_path:
        return [bash_path, "--norc", "--noprofile", "-c", command]
    sh_path = shutil.which("sh") or "/bin/sh"
    return [sh_path, "-c", command]


# ---------------------------------------------------------------------------
# Virtualenv discovery
# ---------------------------------------------------------------------------


def _venv_bin_dir(venv_dir: Path) -> Path:
    """Return the ``bin`` (or ``Scripts`` on Windows) directory inside
    *venv_dir*.
    """
    preferred = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    if preferred.exists():
        return preferred
    fallback = venv_dir / ("bin" if os.name == "nt" else "Scripts")
    if fallback.exists():
        return fallback
    return preferred


def _is_valid_venv_dir(venv_dir: Path) -> bool:
    """Return ``True`` if *venv_dir* looks like a usable virtual-env."""
    if not venv_dir.exists() or not venv_dir.is_dir():
        return False
    bin_dir = _venv_bin_dir(venv_dir)
    python_candidates = ("python.exe", "python") if os.name == "nt" else ("python",)
    if any((bin_dir / name).exists() for name in python_candidates):
        return True
    # Last-resort marker for partially provisioned environments.
    return (venv_dir / "pyvenv.cfg").exists()


def _find_venv(working_dir: Path) -> Path | None:
    """Walk up to five parent directories looking for a ``.venv`` or
    ``venv`` directory that appears to be a valid virtual environment.
    """
    current = working_dir.resolve()
    for _ in range(5):
        venv_dir = current / ".venv"
        if _is_valid_venv_dir(venv_dir):
            return venv_dir
        venv_dir = current / "venv"
        if _is_valid_venv_dir(venv_dir):
            return venv_dir
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


# ---------------------------------------------------------------------------
# Safe sub-path check
# ---------------------------------------------------------------------------


def _safe_subpath(base: Path, rel_path: str) -> Path:
    """Resolve *rel_path* under *base* and raise if it escapes."""
    target = (base / rel_path).resolve()
    if target == base or base in target.parents:
        return target
    raise ValueError(f"path escapes base dir: {rel_path}")


# ---------------------------------------------------------------------------
# Git / subprocess capture helpers
# ---------------------------------------------------------------------------


def _no_git_prompt_env() -> dict[str, str]:
    """Return a copy of ``os.environ`` with variables that suppress
    interactive Git/SSH credential prompts so subprocesses never hang
    waiting for input.
    """
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = "/bin/echo"
    env["GIT_SSH_COMMAND"] = "ssh -oBatchMode=yes"
    return env


_NO_GIT_PROMPT_ENV: dict[str, str] = _no_git_prompt_env()


def _run_command_capture(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = SKILL_DYNAMIC_FETCH_TIMEOUT_SEC,
) -> tuple[bool, str]:
    """Run *cmd* and return ``(success, output_or_error)``.

    Uses ``_NO_GIT_PROMPT_ENV`` to prevent interactive prompts.
    """
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=_NO_GIT_PROMPT_ENV,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if p.returncode == 0:
        return True, (p.stdout or "").strip()
    err = (p.stderr or p.stdout or "").strip()
    return False, err or f"command failed: {' '.join(cmd)}"


# ---------------------------------------------------------------------------
# Lightweight JSON object parser
# ---------------------------------------------------------------------------


def _parse_json_object(raw: Any) -> dict[str, Any]:
    """Best-effort parse of *raw* into a ``dict``.

    Accepts an already-parsed dict, a JSON string, or returns ``{}``
    on failure.
    """
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}
