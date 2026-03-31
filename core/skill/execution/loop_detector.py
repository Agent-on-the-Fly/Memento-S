"""Generic loop detection for ReAct execution.

Detects various loop patterns based on tool usage behavior,
not hardcoded tool names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCallRecord:
    """Record of a single tool call."""

    tool_name: str
    category: str  # "observation", "effect", "mixed"
    turn: int
    new_entities: int = 0  # New URLs, files, etc. discovered
    created_artifacts: int = 0  # Files created/modified


class LoopDetector:
    """Detects execution loops based on behavior patterns."""

    def __init__(
        self,
        max_observation_chain: int = 6,
        min_effect_ratio: float = 0.15,
        window_size: int = 10,
    ):
        """
        Args:
            max_observation_chain: Max consecutive observation tools before warning
            min_effect_ratio: Minimum ratio of effect tools to total tools
            window_size: Sliding window size for analysis
        """
        self.max_observation_chain = max_observation_chain
        self.min_effect_ratio = min_effect_ratio
        self.window_size = window_size
        self.history: list[ToolCallRecord] = []

    def record(
        self,
        tool_name: str,
        category: str,
        turn: int,
        new_entities: int = 0,
        created_artifacts: int = 0,
    ) -> None:
        """Record a tool call."""
        self.history.append(
            ToolCallRecord(
                tool_name=tool_name,
                category=category,
                turn=turn,
                new_entities=new_entities,
                created_artifacts=created_artifacts,
            )
        )

    def detect(self) -> dict[str, Any] | None:
        """Detect if execution is in a loop.

        Returns:
            Loop info dict if detected, None otherwise
        """
        if len(self.history) < 5:
            return None

        # Pattern 1: Long chain of observation without effect
        result = self._check_observation_chain()
        if result:
            return result

        # Pattern 2: Low effect ratio in sliding window
        result = self._check_effect_ratio()
        if result:
            return result

        # Pattern 3: Diminishing returns (info collection loop)
        result = self._check_diminishing_returns()
        if result:
            return result

        # Pattern 4: Repeating tool sequences
        result = self._check_repeating_sequence()
        if result:
            return result

        return None

    def _check_observation_chain(self) -> dict[str, Any] | None:
        """Check for long consecutive observation tool chains."""
        # Count trailing observation tools
        obs_chain = 0
        for record in reversed(self.history):
            if record.category == "observation":
                obs_chain += 1
            else:
                break

        if obs_chain >= self.max_observation_chain:
            return {
                "type": "observation_chain",
                "severity": "high",
                "message": (
                    f"You've used {obs_chain} consecutive observation tools "
                    f"(search/read/grep) without creating or modifying anything. "
                    "This is a RESEARCH LOOP. Stop collecting information and "
                    "start creating the deliverable."
                ),
                "chain_length": obs_chain,
            }
        return None

    def _check_effect_ratio(self) -> dict[str, Any] | None:
        """Check if effect tools ratio is too low."""
        window = self.history[-self.window_size :]
        total = len(window)
        effect_count = sum(1 for r in window if r.category == "effect")
        ratio = effect_count / total if total > 0 else 0

        if total >= self.window_size and ratio < self.min_effect_ratio:
            return {
                "type": "low_effect_ratio",
                "severity": "medium",
                "message": (
                    f"In the last {total} actions, only {effect_count} "
                    f"({ratio:.0%}) created or modified files. "
                    "You're collecting information faster than using it. "
                    "Switch to creation mode."
                ),
                "ratio": ratio,
                "window_size": total,
            }
        return None

    def _check_diminishing_returns(self) -> dict[str, Any] | None:
        """Check if new information discovery is decreasing."""
        # Look at last 6 observation tools
        obs_records = [r for r in self.history if r.category == "observation"][-6:]

        if len(obs_records) < 4:
            return None

        # Check if new entities are decreasing
        entities = [r.new_entities for r in obs_records]
        if all(e <= 1 for e in entities[-3:]) and sum(entities) > 0:
            return {
                "type": "diminishing_returns",
                "severity": "medium",
                "message": (
                    "Your last few searches/fetches found very little new information. "
                    "You're in a DIMINISHING RETURNS loop. "
                    "Use what you have already found to create the deliverable."
                ),
                "recent_entities": entities[-3:],
            }
        return None

    def _check_repeating_sequence(self) -> dict[str, Any] | None:
        """Check for repeating tool call patterns (e.g., A-B-A-B)."""
        if len(self.history) < 6:
            return None

        # Check for 2-tool or 3-tool repeating patterns
        for pattern_len in [2, 3]:
            if len(self.history) < pattern_len * 2:
                continue

            recent = [r.tool_name for r in self.history[-pattern_len * 2 :]]
            pattern = recent[:pattern_len]

            # Check if pattern repeats exactly
            if recent == pattern * 2:
                return {
                    "type": "repeating_sequence",
                    "severity": "high",
                    "message": (
                        f"You're repeating the same {pattern_len}-step sequence: "
                        f"{' → '.join(pattern)}. This is a LOOP. "
                        "Break the pattern by using a different approach."
                    ),
                    "sequence": pattern,
                    "repetitions": 2,
                }
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        if not self.history:
            return {}

        total = len(self.history)
        categories = {}
        for r in self.history:
            categories[r.category] = categories.get(r.category, 0) + 1

        return {
            "total_calls": total,
            "categories": categories,
            "effect_ratio": categories.get("effect", 0) / total,
            "observation_ratio": categories.get("observation", 0) / total,
        }
