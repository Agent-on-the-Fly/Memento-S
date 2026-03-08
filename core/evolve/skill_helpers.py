"""Shared skill-folder helpers — used by optimizer, feedback, and unit_test_gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .text_utils import normalize_skill_name


def read_skill_md_text(skill_dir: Path, max_chars: int = 6000) -> str:
    """Read SKILL.md from a skill directory, truncating if needed."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return ""
    try:
        text = skill_md.read_text(encoding="utf-8")
    except Exception:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]..."
    return text


def resolve_local_skill_name(
    raw: str,
    *,
    base_skills_root: Path,
    extra_skills_root: Path,
) -> str | None:
    """Resolve a raw skill name to its canonical directory name."""
    name = str(raw or "").strip()
    if not name:
        return None
    if (extra_skills_root / name / "SKILL.md").exists():
        return name
    if (base_skills_root / name / "SKILL.md").exists():
        return name
    norm = normalize_skill_name(name)
    for root in (extra_skills_root, base_skills_root):
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir() and normalize_skill_name(child.name) == norm:
                if (child / "SKILL.md").exists():
                    return child.name
    return None
