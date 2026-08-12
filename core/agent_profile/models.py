# SPDX-License-Identifier: Apache-2.0
"""AgentProfile dataclass — pure data container for agent identity (SOUL.md pattern)."""

from dataclasses import dataclass, field


@dataclass
class AgentProfile:
    """Agent identity and soul model (SOUL.md pattern). Pure data container."""

    name: str = "Memento-S"
    role: str = "AI assistant with skill-based task execution"

    core_truths: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)
    vibe: str = ""
    tone_examples: list[tuple[str, str]] = field(default_factory=list)

    capabilities: list[str] = field(default_factory=list)
    model_info: str = ""
    available_tools: list[str] = field(default_factory=list)
    user_context: str = ""

    def to_prompt_section(self) -> str:
        """Generate a SOUL.md-style system prompt section."""
        lines = [
            "## Agent Soul",
            f"Name: {self.name}",
            f"Role: {self.role}",
        ]

        if self.core_truths:
            lines.append("\n### Core Truths")
            for truth in self.core_truths:
                lines.append(f"- {truth}")

        if self.vibe:
            lines.append(f"\n### Vibe\n{self.vibe}")

        if self.tone_examples:
            lines.append("\n### Tone Examples")
            lines.append("| Flat | Alive |")
            lines.append("| --- | --- |")
            for flat, alive in self.tone_examples:
                lines.append(f"| {flat} | {alive} |")

        if self.boundaries:
            lines.append("\n### Boundaries")
            for b in self.boundaries:
                lines.append(f"- {b}")

        if self.capabilities:
            lines.append("\n### Available Local Skills")
            for cap in self.capabilities:
                lines.append(f"- {cap}")

        if self.user_context:
            lines.append(self.user_context)

        return "\n".join(lines)
