
from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Coroutine

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg"}
_IMAGE_MAX_BYTES = 20 * 1024 * 1024  # 20 MB

_base_dir: Path = Path.cwd()
_skill_library: Any = None


def configure(workspace: Path, skill_library: Any = None) -> None:
    global _base_dir, _skill_library
    _base_dir = Path(workspace).expanduser().resolve()
    if skill_library is not None:
        _skill_library = skill_library


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        return _base_dir / p
    return p



async def bash_tool(command: str, description: str) -> str:
    if not command.strip():
        return "bash_tool ERR: empty command"

    wd = _base_dir
    try:
        wd.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    env = os.environ.copy()

    def _run() -> str:
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=str(wd),
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return f"bash_tool TIMEOUT after 120s: {command}"
        except FileNotFoundError as exc:
            return f"bash_tool ERR: shell not found: {exc}"
        except Exception as exc:
            return f"bash_tool ERR: {exc}"

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return f"bash_tool ERR (exit {proc.returncode}):\n{stderr or stdout}"
        return stdout or stderr or "OK"

    return await asyncio.to_thread(_run)


async def str_replace_tool(
    description: str,
    path: str,
    old_str: str,
    new_str: str = "",
) -> str:
    def _run() -> str:
        p = _resolve_path(path)
        if not p.exists():
            return f"str_replace ERR: file not found: {p}"
        if not p.is_file():
            return f"str_replace ERR: not a file: {p}"

        content = p.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_str)

        if count == 0:
            return f"str_replace ERR: old_str not found in {p}"
        if count > 1:
            return f"str_replace ERR: old_str appears {count} times in {p} (must be unique)"

        new_content = content.replace(old_str, new_str, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"str_replace OK: {p}"

    return await asyncio.to_thread(_run)


async def file_create_tool(
    description: str,
    path: str,
    file_text: str,
) -> str:
    def _run() -> str:
        p = _resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(file_text, encoding="utf-8")
        return p

    p = await asyncio.to_thread(_run)
    result = f"file_create OK: {p}"

    try:
        skills_dir = _base_dir / "skills"
        if _skill_library and p.resolve().is_relative_to(skills_dir.resolve()):
            skill_dir = p.parent if p.name == "SKILL.md" else p.parent.parent
            if (skill_dir / "SKILL.md").exists():
                added = _skill_library.refresh_from_disk()
                if added:
                    result += f" (auto-refreshed: {added} new skill(s))"
    except Exception:
        pass

    return result


async def view_tool(
    description: str,
    path: str,
    view_range: list[int] | None = None,
) -> str:
    def _run() -> str:
        p = _resolve_path(path)

        if not p.exists():
            return f"view ERR: not found: {p}"

        if p.is_dir():
            return _view_directory(p, max_depth=2)

        if p.suffix.lower() in _IMAGE_EXTS:
            size = p.stat().st_size
            if size > _IMAGE_MAX_BYTES:
                return f"view ERR: image too large ({size} bytes, max {_IMAGE_MAX_BYTES})"
            mime = mimetypes.guess_type(str(p))[0] or "image/png"
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{b64}"

        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"view ERR: cannot read {p}: {exc}"

        lines = content.splitlines()

        if view_range is not None and len(view_range) == 2:
            start, end = view_range
            start = max(1, start)
            if end == -1:
                end = len(lines)
            end = min(end, len(lines))
            lines = lines[start - 1 : end]
            offset = start
        else:
            offset = 1

        numbered = [f"{offset + i:>6}\t{line}" for i, line in enumerate(lines)]
        return "\n".join(numbered)

    return await asyncio.to_thread(_run)


def _view_directory(
    path: Path,
    max_depth: int = 2,
    current_depth: int = 0,
    prefix: str = "",
) -> str:
    lines: list[str] = []
    if current_depth == 0:
        lines.append(str(path) + "/")

    try:
        entries = sorted(
            path.iterdir(),
            key=lambda x: (not x.is_dir(), x.name.lower()),
        )
    except PermissionError:
        return f"{prefix}[permission denied]"

    entries = [
        e for e in entries if not e.name.startswith(".") and e.name != "node_modules"
    ]

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{prefix}{connector}{entry.name}{suffix}")
        if entry.is_dir() and current_depth < max_depth:
            extension = "    " if is_last else "│   "
            sub = _view_directory(entry, max_depth, current_depth + 1, prefix + extension)
            if sub:
                lines.append(sub)

    return "\n".join(lines)


async def read_skill_tool(skill_name: str) -> str:
    from core.config import g_settings
    from core.skills.provider.delta_skills.skills.store.persistence import to_kebab_case

    skills_dir = g_settings.workspace_path / "skills"
    for dirname in [skill_name, to_kebab_case(skill_name), skill_name.replace("-", "_")]:
        skill_md = skills_dir / dirname / "SKILL.md"
        if skill_md.exists():
            skill_dir = skill_md.parent
            content = await asyncio.to_thread(skill_md.read_text, "utf-8")
            hint = (
                f"[Skill Location] {skill_dir}\n"
                f"To run scripts in this skill, use: "
                f"cd {skill_dir} && python3 scripts/<script>.py <args>\n"
                f"Do NOT use `from skills.* import ...` — always run scripts via bash_tool with the path above.\n\n"
            )
            return hint + content
    return f"ERR: skill '{skill_name}' not found. Available skills are in the [Matched Skills] section."



BUILTIN_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash_tool",
            "description": "Run a bash command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to run",
                    },
                    "description": {
                        "type": "string",
                        "description": "Why I'm running this command",
                    },
                },
                "required": ["command", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": (
                "Replace a unique string in a file with another string. "
                "The string to replace must appear exactly once in the file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Why I'm making this edit",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "String to replace (must be unique in file)",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "String to replace with (empty to delete)",
                        "default": "",
                    },
                },
                "required": ["description", "path", "old_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_create",
            "description": "Create a new file with content. Parent directories are created automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Why I'm creating this file. ALWAYS PROVIDE THIS PARAMETER FIRST.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the file to create. ALWAYS PROVIDE THIS PARAMETER SECOND.",
                    },
                    "file_text": {
                        "type": "string",
                        "description": "Content to write to the file. ALWAYS PROVIDE THIS PARAMETER LAST.",
                    },
                },
                "required": ["description", "path", "file_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view",
            "description": (
                "View text files (with line numbers), directories (tree listing), or images (base64). "
                "Supports optional line range for text files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Why I need to view this",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to file or directory",
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Optional line range [start_line, end_line]. "
                            "Lines are 1-indexed. Use [start, -1] for start to end of file."
                        ),
                    },
                },
                "required": ["description", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": "Read a skill's SKILL.md documentation. Use this to understand how a skill works before executing it via bash_tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to read (e.g. 'web-search', 'skill-creator')",
                    },
                },
                "required": ["skill_name"],
            },
        },
    },
]



BUILTIN_TOOL_REGISTRY: dict[str, Callable[..., Coroutine[Any, Any, str]]] = {
    "bash_tool": bash_tool,
    "str_replace": str_replace_tool,
    "file_create": file_create_tool,
    "view": view_tool,
    "read_skill": read_skill_tool,
}


def is_builtin_tool(name: str) -> bool:
    return name in BUILTIN_TOOL_REGISTRY


async def execute_builtin_tool(name: str, arguments: dict[str, Any]) -> str:
    fn = BUILTIN_TOOL_REGISTRY.get(name)
    if fn is None:
        return f"ERR: unknown builtin tool '{name}'"
    return await fn(**arguments)
