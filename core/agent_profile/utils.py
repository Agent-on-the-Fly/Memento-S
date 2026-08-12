# SPDX-License-Identifier: Apache-2.0
"""SOUL.md parsing and formatting utilities."""

from core.agent_profile.defaults import _DEFAULT_SOUL_TEMPLATE


def _parse_soul(raw: str) -> dict:
    """Parse SOUL.md text into a dict. Falls back to default template on empty/parse failure."""
    if not raw.strip():
        raw = _DEFAULT_SOUL_TEMPLATE

    data = {
        "name": "Memento-S",
        "role": "AI assistant with skill-based task execution",
        "core_truths": [],
        "boundaries": [],
        "vibe": "",
        "tone_examples": [],
    }
    lines = raw.splitlines()
    current_section: str | None = None
    section_lines: list[str] = []

    def flush():
        if current_section is None or not section_lines:
            return
        if current_section == "Core Truths":
            data["core_truths"] = [ln.lstrip("-•* ").strip() for ln in section_lines if ln.strip()]
        elif current_section == "Boundaries":
            data["boundaries"] = [ln.lstrip("-•* ").strip() for ln in section_lines if ln.strip()]
        elif current_section == "Vibe":
            data["vibe"] = "\n".join(section_lines).strip()
        elif current_section == "Role":
            data["role"] = "\n".join(section_lines).strip()
        elif current_section == "Tone Examples":
            pairs: list[tuple[str, str]] = []
            for ln in section_lines:
                if "|" in ln and not ln.startswith("|"):
                    parts = [p.strip() for p in ln.split("|")]
                    if len(parts) >= 3:
                        pairs.append((parts[1], parts[2]))
            data["tone_examples"] = pairs

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            flush()
            current_section = stripped[3:].strip()
            section_lines = []
        elif stripped.startswith("# ") and current_section is None:
            title = stripped[2:].strip()
            if title.startswith("SOUL.md"):
                parts = title.split("—", 1)
                if len(parts) >= 2:
                    data["name"] = parts[1].strip()
        elif stripped.startswith("### "):
            flush()
            current_section = stripped[4:].strip()
            section_lines = []
        elif stripped and current_section:
            section_lines.append(stripped)
    flush()
    return data


def _format_soul(data: dict) -> str:
    """Format a SOUL dict back into .md text."""
    lines = [f"# SOUL.md — {data.get('name', 'Memento-S')} Identity", ""]
    lines.append("## Core Truths")
    for t in data.get("core_truths", []):
        lines.append(f"- {t}")
    lines.append("")
    lines.append("## Boundaries")
    for b in data.get("boundaries", []):
        lines.append(f"- {b}")
    lines.append("")
    lines.append("## Vibe")
    lines.append(data.get("vibe", ""))
    if data.get("role"):
        lines.extend(["", "## Role", data["role"]])
    tone_examples = data.get("tone_examples", [])
    if tone_examples:
        lines.extend(["", "## Tone Examples", "| Flat | Alive |", "| --- | --- |"])
        for flat, alive in tone_examples:
            lines.append(f"| {flat} | {alive} |")
    return "\n".join(lines)
