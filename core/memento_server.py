from __future__ import annotations

import base64
import mimetypes
import os
import subprocess
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP

from core.config import WORKSPACE_DIR

mcp = FastMCP("memento")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg"}
_IMAGE_MAX_BYTES = 20 * 1024 * 1024  # 20 MB

_base_dir: Path = WORKSPACE_DIR

def configure(*, base_dir: Path | None = None) -> None:
    global _base_dir
    if base_dir is not None:
        _base_dir = base_dir


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        return _base_dir / p
    # Trust absolute paths as-is; create parent dirs if needed
    return p


# ===================================================================
# 1. bash_tool
# ===================================================================

@mcp.tool
def bash_tool(
    command: Annotated[str, "Bash command to run in container"],
    description: Annotated[str, "Why I'm running this command"],
) -> str:
    """Run a bash command in the container."""
    if not command.strip():
        return "bash_tool ERR: empty command"

    wd = _base_dir
    try:
        wd.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    env = os.environ.copy()

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


# ===================================================================
# 2. str_replace
# ===================================================================

@mcp.tool
def str_replace(
    description: Annotated[str, "Why I'm making this edit"],
    path: Annotated[str, "Path to the file to edit"],
    old_str: Annotated[str, "String to replace (must be unique in file)"],
    new_str: Annotated[str, "String to replace with (empty to delete)"] = "",
) -> str:
    """Replace a unique string in a file with another string. The string to replace must appear exactly once in the file."""
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


# ===================================================================
# 3. file_create
# ===================================================================

@mcp.tool
def file_create(
    description: Annotated[str, "Why I'm creating this file. ALWAYS PROVIDE THIS PARAMETER FIRST."],
    path: Annotated[str, "Path to the file to create. ALWAYS PROVIDE THIS PARAMETER SECOND."],
    file_text: Annotated[str, "Content to write to the file. ALWAYS PROVIDE THIS PARAMETER LAST."],
) -> str:
    """Create a new file with content in the container."""
    p = _resolve_path(path)

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(file_text, encoding="utf-8")
    return f"file_create OK: {p}"


# ===================================================================
# 4. view
# ===================================================================

@mcp.tool
def view(
    description: Annotated[str, "Why I need to view this"],
    path: Annotated[str, "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`."],
    view_range: Annotated[
        list[int] | None,
        "Optional line range for text files. Format: [start_line, end_line] where lines are indexed starting at 1. Use [start_line, -1] to view from start_line to end of file.",
    ] = None,
) -> str:
    """Supports viewing text, images, and directory listings.

    Supported path types:
    - Directories: Lists files and directories up to 2 levels deep, ignoring hidden items and node_modules
    - Image files (.jpg, .jpeg, .png, .gif, .webp): Displays the image visually
    - Text files: Displays numbered lines. You can optionally specify a view_range to see specific lines.

    Note: Files with non-UTF-8 encoding will display hex escapes (e.g. \\x84) for invalid bytes"""
    p = _resolve_path(path)

    if not p.exists():
        return f"view ERR: not found: {p}"

    # Directory listing
    if p.is_dir():
        return _view_directory(p, max_depth=2)

    # Image files — return base64 for multimodal models
    if p.suffix.lower() in _IMAGE_EXTS:
        size = p.stat().st_size
        if size > _IMAGE_MAX_BYTES:
            return f"view ERR: image too large ({size} bytes, max {_IMAGE_MAX_BYTES})"
        mime = mimetypes.guess_type(str(p))[0] or "image/png"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    # Text files
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

    # Filter hidden items and node_modules
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


# ===================================================================
# 5. read_skill
# ===================================================================

@mcp.tool
def read_skill(
    skill_name: Annotated[str, "Name of the skill to read"],
) -> str:
    """Read a skill's SKILL.md content."""
    from core.skill_engine.skill_resolver import ensure_skill_available, openskills_read
    # Auto-fetch from catalog if not locally available
    ensure_skill_available(skill_name)
    try:
        return openskills_read(skill_name)
    except Exception as exc:
        return f"read_skill ERR: {exc}"


# ===================================================================
# 6. refresh_skills — notify catalog of newly created skills
# ===================================================================

@mcp.tool
def refresh_skills() -> str:
    """Refresh the skill catalog to pick up newly created or modified skills.

    Call this after creating a new skill (e.g. via skill-creator) so it
    becomes immediately available for routing and read_skill.
    """
    try:
        from core.skill_engine.skill_catalog import get_skill_catalog
        catalog = get_skill_catalog()
        catalog.invalidate()
        # Force rescan
        skills = catalog.route("", top_k=1)
        total = len(catalog._ensure_catalog())
        return f"refresh_skills OK: catalog refreshed, {total} skills indexed"
    except Exception as exc:
        return f"refresh_skills ERR: {exc}"



# ---------------------------------------------------------------------------
# Non-tool helper for CLI /skills command
# ---------------------------------------------------------------------------

def _list_local_skills_text() -> str:
    """List all locally available skills with their descriptions (non-tool helper)."""
    from core.skill_engine.skill_resolver import _iter_skill_roots
    seen: set[str] = set()
    lines: list[str] = []
    for root in _iter_skill_roots():
        if not root.exists() or not root.is_dir():
            continue
        try:
            for skill_dir in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue
                name = skill_dir.name
                if name in seen:
                    continue
                seen.add(name)
                desc = _extract_skill_description(skill_md)
                lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        except Exception:
            continue
    return "\n".join(lines) if lines else "(no local skills found)"


def _extract_skill_description(skill_md: Path) -> str:
    """Extract description from SKILL.md YAML frontmatter."""
    try:
        text = skill_md.read_text(encoding="utf-8")
    except Exception:
        return ""
    # Parse YAML frontmatter between --- markers
    if not text.startswith("---"):
        return ""
    end = text.find("---", 3)
    if end == -1:
        return ""
    for line in text[3:end].splitlines():
        line = line.strip()
        if line.lower().startswith("description:"):
            desc = line[len("description:"):].strip().strip("\"'")
            return desc[:200]
    return ""
