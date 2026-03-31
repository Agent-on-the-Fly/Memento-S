"""Error pattern detection and automatic recovery system.

This module provides intelligent error analysis and recovery hint injection
to help LLM break out of repetitive error loops.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ErrorPattern:
    """Definition of a detectable error pattern."""

    name: str
    signatures: list[str]
    frequency_threshold: int
    recovery_hint: str
    severity: str = "medium"  # low, medium, high


class StatefulErrorPatternDetector:
    """Detects repetitive error patterns and generates recovery hints."""

    PATTERNS: dict[str, ErrorPattern] = {
        "stateless_variable_loss": ErrorPattern(
            name="stateless_variable_loss",
            signatures=["NameError", "is not defined"],
            frequency_threshold=2,
            recovery_hint=(
                "PATTERN_DETECTED: You are trying to use variables from previous python_repl calls. "
                "The sandbox is STATELESS - variables do not persist between calls. "
                "SOLUTION: Either (1) Include all code in a single python_repl call, "
                "or (2) Use context_manager_tool to save variables, "
                "or (3) Write code to file and execute with bash."
            ),
            severity="high",
        ),
        "syntax_chinese_quotes": ErrorPattern(
            name="syntax_chinese_quotes",
            signatures=["SyntaxError", "invalid character", " unexpected"],
            frequency_threshold=1,
            recovery_hint=(
                "PATTERN_DETECTED: Syntax error may be caused by Chinese quotes or special characters. "
                "SOLUTION: Replace Chinese quotes '「」' '“”' '‘’' with ASCII quotes \"'\" and '\"'. "
                "Check for full-width characters in your code."
            ),
            severity="medium",
        ),
        "module_not_found": ErrorPattern(
            name="module_not_found",
            signatures=["ModuleNotFoundError", "No module named"],
            frequency_threshold=2,
            recovery_hint=(
                "PATTERN_DETECTED: Required Python module not installed. "
                "SOLUTION: Install the module first using python_repl with deps parameter, "
                "or check if the module name is correct (e.g., 'PIL' vs 'Pillow')."
            ),
            severity="medium",
        ),
        "file_not_found": ErrorPattern(
            name="file_not_found",
            signatures=["FileNotFoundError", "No such file"],
            frequency_threshold=2,
            recovery_hint=(
                "PATTERN_DETECTED: File or directory not found. "
                "SOLUTION: Use list_dir to verify the path exists. "
                "Check for typos in filename or use absolute path."
            ),
            severity="medium",
        ),
        "permission_denied": ErrorPattern(
            name="permission_denied",
            signatures=["PermissionError", "Permission denied"],
            frequency_threshold=1,
            recovery_hint=(
                "PATTERN_DETECTED: Permission denied when accessing file. "
                "SOLUTION: Check file permissions with bash 'ls -la'. "
                "You may need to use a different directory or check workspace boundaries."
            ),
            severity="medium",
        ),
        "infinite_retry_loop": ErrorPattern(
            name="infinite_retry_loop",
            signatures=[],  # Detected by behavior analysis, not signatures
            frequency_threshold=3,
            recovery_hint=(
                "PATTERN_DETECTED: You appear to be stuck in a retry loop with the same approach. "
                "SOLUTION: Step back and use update_scratchpad to document what you've tried. "
                "Consider a completely different approach or tool. "
                "DO NOT repeat the same tool call with the same arguments."
            ),
            severity="high",
        ),
        "bash_pipe_escape": ErrorPattern(
            name="bash_pipe_escape",
            signatures=["unexpected token", "syntax error", "command not found"],
            frequency_threshold=2,
            recovery_hint=(
                "PATTERN_DETECTED: Bash command syntax error, possibly due to special character escaping. "
                "SOLUTION: For pipes and redirects, ensure proper quoting. "
                "Use bash 'cd dir && ls' pattern instead of separate cd and ls calls. "
                "Avoid using special shell characters that may be escaped incorrectly."
            ),
            severity="medium",
        ),
        "import_failure": ErrorPattern(
            name="import_failure",
            signatures=["ImportError", "cannot import name"],
            frequency_threshold=2,
            recovery_hint=(
                "PATTERN_DETECTED: Failed to import a name from a module. "
                "SOLUTION: Check the module structure with read_file. "
                "Verify the function/class name exists and is exported. "
                "Check for circular import issues."
            ),
            severity="medium",
        ),
    }

    @classmethod
    def analyze(
        cls,
        error_history: list[dict[str, Any]],
        action_history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Analyze error history and return recovery hints.

        Args:
            error_history: List of error records with 'error', 'tool', 'turn' keys
            action_history: Optional list of action records for behavior analysis

        Returns:
            List of recovery hint dicts with 'pattern', 'hint', 'severity' keys
        """
        hints = []
        recent_errors = error_history[-5:] if len(error_history) > 5 else error_history

        # Check signature-based patterns
        for pattern_name, pattern in cls.PATTERNS.items():
            if pattern_name == "infinite_retry_loop":
                # Special handling for retry loop detection
                continue

            match_count = sum(
                1
                for e in recent_errors
                if all(sig in str(e.get("error", "")) for sig in pattern.signatures)
            )

            if match_count >= pattern.frequency_threshold:
                hints.append(
                    {
                        "pattern": pattern_name,
                        "hint": pattern.recovery_hint,
                        "severity": pattern.severity,
                        "match_count": match_count,
                    }
                )

        # Check for retry loop by analyzing action history
        if action_history and len(action_history) >= 3:
            if cls._detect_retry_loop(action_history):
                pattern = cls.PATTERNS["infinite_retry_loop"]
                hints.append(
                    {
                        "pattern": "infinite_retry_loop",
                        "hint": pattern.recovery_hint,
                        "severity": pattern.severity,
                        "match_count": len(action_history),
                    }
                )

        return hints

    @classmethod
    def _detect_retry_loop(cls, action_history: list[dict[str, Any]]) -> bool:
        """Detect if LLM is stuck retrying the same approach."""
        if len(action_history) < 3:
            return False

        # Get recent actions
        recent = action_history[-5:]

        # Check for repeated tool + argument pattern
        action_sigs = []
        for action in recent:
            tool = action.get("tool", "")
            args = str(action.get("arguments", {}))
            # Normalize args for comparison
            args_normalized = re.sub(r"\s+", "", args.lower())
            action_sigs.append(f"{tool}:{args_normalized}")

        # Check if same action repeated
        if len(set(action_sigs)) == 1 and len(action_sigs) >= 3:
            return True

        # Check for alternating between two failed approaches
        if len(action_sigs) >= 4:
            unique_sigs = list(dict.fromkeys(action_sigs))
            if len(unique_sigs) == 2:
                # Check if alternating pattern: A, B, A, B
                pattern = action_sigs[:2]
                if action_sigs == pattern * (len(action_sigs) // 2):
                    return True

        return False

    @classmethod
    def get_error_fingerprint(cls, error: str) -> str | None:
        """Generate a normalized fingerprint for an error message.

        This allows us to track if the same error is occurring repeatedly,
        even if line numbers or paths differ.
        """
        if not error:
            return None

        # Normalize error message
        normalized = error.lower()

        # Remove variable content
        normalized = re.sub(r"line\s+\d+", "line <n>", normalized)
        normalized = re.sub(r"/[^\s:\"']+", "<path>", normalized)
        normalized = re.sub(r"'[^']+'", "'<var>'", normalized)
        normalized = re.sub(r'"[^"]+"', '"<str>"', normalized)
        normalized = re.sub(r"\b\d+\b", "<n>", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    @classmethod
    def should_inject_recovery_hint(
        cls,
        error_history: list[dict[str, Any]],
        current_turn: int,
        min_turns_between_hints: int = 2,
    ) -> bool:
        """Determine if we should inject a recovery hint now.

        Prevents hint spam by spacing out hints.
        """
        if not error_history:
            return False

        # Check last hint injection turn
        last_hint_turn = 0
        for e in reversed(error_history):
            if e.get("was_recovery_hint_injected"):
                last_hint_turn = e.get("turn", 0)
                break

        return (current_turn - last_hint_turn) >= min_turns_between_hints
