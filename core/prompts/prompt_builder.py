"""PromptBuilder — priority-ordered section assembly for system prompts.

Sections are registered with a numeric priority (lower = earlier in output).
``build()`` concatenates them in priority order with a configurable separator.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(order=True)
class _Section:
    priority: int
    content: str = field(compare=False)
    label: str = field(default="", compare=False)


class PromptBuilder:
    """Accumulate named prompt sections and render them in priority order."""

    def __init__(self, separator: str = "\n\n---\n\n") -> None:
        self._sections: list[_Section] = []
        self._separator = separator

    def add(self, content: str, *, priority: int = 50, label: str = "") -> "PromptBuilder":
        """Register a prompt section.

        Args:
            content: The text to include.
            priority: Lower values appear first. Default 50.
            label: Optional label for debugging / inspection.

        Returns:
            ``self`` for fluent chaining.
        """
        if content and content.strip():
            self._sections.append(_Section(priority=priority, content=content, label=label))
        return self

    def build(self) -> str:
        """Render all sections in priority order, joined by the separator."""
        self._sections.sort()
        return self._separator.join(s.content for s in self._sections)

    @property
    def section_count(self) -> int:
        return len(self._sections)

    def labels(self) -> list[str]:
        """Return labels of registered sections in priority order (debug helper)."""
        self._sections.sort()
        return [s.label for s in self._sections]
