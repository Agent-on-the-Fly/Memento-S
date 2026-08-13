# SPDX-License-Identifier: Apache-2.0
"""Lifecycle classifier for file change detection.

Classifies files as TEMPORARY, ARTIFACT, or UNKNOWN based on
multiple heuristics including tool source, path patterns, extensions, etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.skill.execution.detection.types import FileLifecycle
from shared.fs.types import ChangeType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LifecycleClassification:
    """Result of a lifecycle classification."""
    lifecycle: FileLifecycle
    confidence: float  # 0.0 - 1.0
    reasons: list[str]
    overrides: list[str]  # List of rule names that matched


# Default classification rules
DEFAULT_TEMPORARY_EXTENSIONS: frozenset[str] = frozenset({
    ".tmp", ".bak", ".log", ".cache",
    ".pyc", ".pyo", ".pyd",
    ".pyi", ".pyw",
    ".swp", ".swo",  # vim swap
    ".DS_Store", "Thumbs.db",
    ".orig", ".rej",  # diff artifacts
    ".stderr", ".stdout",
})

DEFAULT_TEMPORARY_PATH_PATTERNS: frozenset[str] = frozenset({
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".venv",
    "venv",
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".tmp",
    "node_modules",
    ".npm",
    ".parcel-cache",
    ".next",
    ".nuxt",
    ".svelte-kit",
})

# 文件扩展名到产物类型的映射
DEFAULT_ARTIFACT_EXTENSIONS: frozenset[str] = frozenset({
    # Documents
    ".pdf", ".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt",
    ".md", ".rst",
    # Media
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".mp4", ".mp3", ".wav", ".flac", ".webm",
    # Code
    ".py", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp",
    ".js", ".ts", ".jsx", ".tsx", ".vue", ".svelte",
    ".html", ".css", ".scss", ".sass", ".less",
    ".rb", ".php", ".swift", ".kt", ".scala",
    # Data/Config
    ".json", ".yaml", ".yml", ".toml", ".xml", ".ini", ".cfg",
    ".env", ".properties",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    # Executables
    ".exe", ".app", ".dmg", ".deb", ".rpm",
    # Other
    ".txt", ".csv", ".tsv", ".sql",
})

# 工具来源分类
TEMPORARY_TOOLS: frozenset[str] = frozenset({
    "python_repl",
    "js_repl",
})

ARTIFACT_TOOLS: frozenset[str] = frozenset({
    "file_create",
    "edit_file_by_lines",
    "bash",  # bash 可以产生最终产物
})


class FileLifecycleClassifier:
    """Classifies files based on lifecycle characteristics.

    Uses multiple heuristics:
    1. Tool source (what tool created the file)
    2. Path patterns (temporary directories)
    3. File extension (known artifact types)
    4. File size and content hints
    """

    def __init__(
        self,
        temporary_extensions: frozenset[str] | None = None,
        temporary_path_patterns: frozenset[str] | None = None,
        artifact_extensions: frozenset[str] | None = None,
        temporary_tools: frozenset[str] | None = None,
        artifact_tools: frozenset[str] | None = None,
    ):
        self.temporary_extensions = temporary_extensions or DEFAULT_TEMPORARY_EXTENSIONS
        self.temporary_path_patterns = temporary_path_patterns or DEFAULT_TEMPORARY_PATH_PATTERNS
        self.artifact_extensions = artifact_extensions or DEFAULT_ARTIFACT_EXTENSIONS
        self.temporary_tools = temporary_tools or TEMPORARY_TOOLS
        self.artifact_tools = artifact_tools or ARTIFACT_TOOLS

    def classify(
        self,
        path: Path,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        change_type: ChangeType | None = None,
    ) -> LifecycleClassification:
        """Classify the lifecycle of a file.

        Args:
            path: Absolute path to the file.
            tool_name: Name of the tool that was executed.
            tool_args: Arguments passed to the tool.
            change_type: Type of change detected.

        Returns:
            LifecycleClassification with lifecycle, confidence, and reasons.
        """
        reasons: list[str] = []
        overrides: list[str] = []
        confidence = 0.5

        # [ANALYSIS-LOG] Always log at INFO so it appears in logs regardless of DEBUG level
        logger.info(
            "[ANALYSIS-LOG] classify ENTRY: tool={}, path={}, suffix={}",
            tool_name, path, path.suffix.lower()
        )

        # Python REPL runner files are implementation details, never user
        # artifacts, even though they use an otherwise artifact-like .py suffix.
        if path.name.startswith("__runner__"):
            return LifecycleClassification(
                lifecycle=FileLifecycle.TEMPORARY,
                confidence=1.0,
                reasons=["Internal REPL runner"],
                overrides=["internal_runner"],
            )

        # Rule 0: Artifact extension override (highest priority for artifacts)
        # Check BEFORE tool_source check to allow .md/.txt/.json to bypass TEMPORARY_TOOLS
        ext = path.suffix.lower()
        if ext in self.artifact_extensions:
            reasons.append(f"Artifact extension: {ext}")
            overrides.append("artifact_extension_override")
            confidence = 0.92
            logger.info(
                "[ANALYSIS-LOG] classify: tool={} matched artifact_extension_override "
                "for {}, confidence={}, ext={}",
                tool_name, path, confidence, ext
            )
            return LifecycleClassification(
                lifecycle=FileLifecycle.ARTIFACT,
                confidence=confidence,
                reasons=reasons,
                overrides=overrides,
            )

        # Rule 1: Tool source override (TEMPORARY_TOOLS)
        # Only applies if NOT already matched by artifact extension
        if tool_name in self.temporary_tools:
            reasons.append(f"Tool '{tool_name}' produces temporary output")
            overrides.append("tool_source_temporary")
            confidence = 0.95

            # Check if explicitly configured to produce artifact
            if self._is_explicit_artifact_path(path, tool_args):
                reasons.append("Explicit artifact path override")
                overrides.append("explicit_artifact_path")
                confidence = 0.9
                return LifecycleClassification(
                    lifecycle=FileLifecycle.ARTIFACT,
                    confidence=confidence,
                    reasons=reasons,
                    overrides=overrides,
                )

            return LifecycleClassification(
                lifecycle=FileLifecycle.TEMPORARY,
                confidence=confidence,
                reasons=reasons,
                overrides=overrides,
            )

        if tool_name in self.artifact_tools:
            reasons.append(f"Tool '{tool_name}' produces artifacts")
            overrides.append("tool_source_artifact")
            confidence = 0.9

            # Check if it's actually a temporary output
            if self._is_temporary_path(path):
                reasons.append("Temporary path pattern detected")
                overrides.append("temporary_path_override")
                confidence = 0.8
                return LifecycleClassification(
                    lifecycle=FileLifecycle.TEMPORARY,
                    confidence=confidence,
                    reasons=reasons,
                    overrides=overrides,
                )

            return LifecycleClassification(
                lifecycle=FileLifecycle.ARTIFACT,
                confidence=confidence,
                reasons=reasons,
                overrides=overrides,
            )

        # Rule 2: Temporary path patterns
        if self._is_temporary_path(path):
            reasons.append("Located in temporary directory pattern")
            overrides.append("temporary_path_pattern")
            confidence = 0.85
            return LifecycleClassification(
                lifecycle=FileLifecycle.TEMPORARY,
                confidence=confidence,
                reasons=reasons,
                overrides=overrides,
            )

        # Rule 3: Temporary file extension
        ext = path.suffix.lower()
        if ext in self.temporary_extensions:
            reasons.append(f"Temporary extension: {ext}")
            overrides.append("temporary_extension")
            confidence = 0.8
            return LifecycleClassification(
                lifecycle=FileLifecycle.TEMPORARY,
                confidence=confidence,
                reasons=reasons,
                overrides=overrides,
            )

        # Rule 4: Artifact file extension
        if ext in self.artifact_extensions:
            reasons.append(f"Artifact extension: {ext}")
            overrides.append("artifact_extension")
            confidence = 0.75
            return LifecycleClassification(
                lifecycle=FileLifecycle.ARTIFACT,
                confidence=confidence,
                reasons=reasons,
                overrides=overrides,
            )

        # Rule 5: File size heuristics
        try:
            if path.exists():
                size = path.stat().st_size
                if size == 0:
                    reasons.append("Empty file (likely temporary)")
                    overrides.append("empty_file")
                    confidence = 0.6
                    return LifecycleClassification(
                        lifecycle=FileLifecycle.TEMPORARY,
                        confidence=confidence,
                        reasons=reasons,
                        overrides=overrides,
                    )
                # Very small files might be configuration, not artifacts
                if size < 100:
                    reasons.append(f"Very small file ({size} bytes)")
                    overrides.append("very_small_file")
                    confidence = 0.5
        except OSError:
            pass

        # Rule 6: Output path hints from tool args
        if tool_args:
            output_path = tool_args.get("output_path") or tool_args.get("output")
            if output_path and str(output_path) == str(path):
                reasons.append("Tool explicitly specified this as output")
                overrides.append("explicit_output_path")
                confidence = 0.85
                return LifecycleClassification(
                    lifecycle=FileLifecycle.ARTIFACT,
                    confidence=confidence,
                    reasons=reasons,
                    overrides=overrides,
                )

        # Default: UNKNOWN
        reasons.append("No classification rules matched")
        confidence = 0.3
        return LifecycleClassification(
            lifecycle=FileLifecycle.UNKNOWN,
            confidence=confidence,
            reasons=reasons,
            overrides=overrides,
        )

    def classify_batch(
        self,
        changes: list["FileChange"],  # noqa: F821
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
    ) -> list[LifecycleClassification]:
        """Classify a batch of file changes.

        Args:
            changes: List of FileChange objects.
            tool_name: Name of the tool that was executed.
            tool_args: Arguments passed to the tool.

        Returns:
            List of LifecycleClassification objects (same order as changes).
        """
        results = []
        for change in changes:
            classification = self.classify(
                path=change.path,
                tool_name=tool_name,
                tool_args=tool_args,
                change_type=change.change_type,
            )
            change.lifecycle = classification.lifecycle
            results.append(classification)
        return results

    def _is_temporary_path(self, path: Path) -> bool:
        """Check if path matches temporary directory patterns."""
        path_str = str(path)

        # Check for common temporary directory patterns
        for pattern in self.temporary_path_patterns:
            if pattern in path_str:
                return True

        # Check for /tmp/ prefix (absolute paths)
        if path_str.startswith("/tmp/") or "/tmp/" in path_str:
            return True

        # Check for temp directory names
        for part in path.parts:
            if part.lower() in {"temp", "tmp", "cache", ".cache"}:
                return True

        return False

    def _is_explicit_artifact_path(
        self,
        path: Path,
        tool_args: dict[str, Any] | None,
    ) -> bool:
        """Check if path was explicitly specified as an artifact output."""
        if not tool_args:
            return False

        # Check common output path arguments
        output_keys = {"output_path", "output", "path", "dest", "destination", "file"}
        for key in output_keys:
            if key in tool_args:
                value = str(tool_args[key])
                if value == str(path) or str(path).endswith(value):
                    return True

        return False

    def is_artifact_extension(self, ext: str) -> bool:
        """Check if an extension is considered an artifact type."""
        return ext.lower() in self.artifact_extensions

    def is_temporary_extension(self, ext: str) -> bool:
        """Check if an extension is considered temporary."""
        return ext.lower() in self.temporary_extensions
