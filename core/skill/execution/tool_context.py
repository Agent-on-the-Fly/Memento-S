# SPDX-License-Identifier: Apache-2.0
"""Runtime execution context for tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from shared.schema import SkillConfig
from shared.tools.path_boundary import PathBoundary, get_boundary, detect_platform
from core.skill.schema import Skill
from utils.strings import to_kebab_case
from utils.logger import get_logger

if TYPE_CHECKING:
    from shared.tools.path_boundary import Platform

logger = get_logger(__name__)


class PathResolutionRejected(Exception):
    """Raised when a path is rejected due to a hallucinated segment.

    This is a specific subclass of ValueError so it can be caught and
    re-raised by the caller (tool_args_processor) without being silently
    swallowed.
    """

    pass


# Path-related parameter keys (centralized here to avoid duplication)
PATH_LIKE_KEYS: frozenset[str] = frozenset(
    {
        # File/content paths — subject to rewrite and boundary enforcement
        "path",
        "target_path",
        "file_path",
        "src_path",
        "dst_path",
        "source",
        "destination",
        "directory",
        "dir",
        "output_path",
        "input_path",
        "from_path",
        "to_path",
        "filename",
        "filepath",
        # Vision / media
        "image",
        "images",
        "media",
        # Document
        "pdf",
        "document",
        "doc",
    }
)


@dataclass(frozen=True)
class RuntimeToolContext:
    """Runtime execution context for tool calls within a skill execution session.

    Provides unified path resolution, parameter auto-filling (e.g. skill_name),
    and boundary checking with cross-platform support.
    """

    workspace_dir: Path
    root_dir: Path
    skill_root: Path | None = None
    skill_name: str | None = None  # populated from skill.name directly
    session_id: str = ""           # skill execution session identifier

    # Platform detection (cross-platform support)
    platform: "Platform" = field(default_factory=detect_platform)

    # PathBoundary instance (lazy, per-platform)
    _boundary: PathBoundary = field(default_factory=get_boundary, repr=False)

    # Simplified alias management (in-memory only)
    _aliases: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # Ensure _boundary uses the same platform as self.platform
        if self._boundary.platform != self.platform:
            object.__setattr__(self, "_boundary", get_boundary(self.platform))

    @staticmethod
    def _deduplicate_path_components(
        root_parts: list[str], rel_parts: list[str]
    ) -> list[str]:
        """Deduplicate overlapping path components between root and relative path.

        Compares path components from the end of root_parts with the beginning
        of rel_parts to find and remove overlapping segments.
        """
        if not rel_parts:
            return root_parts

        # Special case: if the first rel component starts with a common prefix
        # of the last root component and extends it (hallucinated segment), skip
        # the first rel component.  E.g. root last='2517d7da', rel first='2517d7d7da'.
        # The overlap loop below handles exact-component matches; this handles
        # partial/misaligned overlaps.
        if root_parts and rel_parts:
            last_root = root_parts[-1]
            first_rel = rel_parts[0]
            # Compute longest common prefix of last_root and first_rel
            lcp_len = 0
            for i in range(min(len(last_root), len(first_rel))):
                if last_root[i] == first_rel[i]:
                    lcp_len = i + 1
                else:
                    break
            # If they share a non-trivial prefix (longer than 2 chars to avoid
            # false positives) and first_rel is longer (extends rather than equals),
            # skip the first_rel component to avoid duplicate segments.
            if lcp_len > 2 and len(first_rel) > len(last_root) and first_rel[:lcp_len] == last_root[:lcp_len]:
                return root_parts + rel_parts[1:]

        max_overlap = min(len(root_parts), len(rel_parts))
        for overlap in range(max_overlap, 0, -1):
            if root_parts[-overlap:] == rel_parts[:overlap]:
                return root_parts + rel_parts[overlap:]

        return root_parts + rel_parts

    def resolve_path(self, raw: str | Path) -> Path:
        """Unified path resolution: parse -> validate -> reject-on-violation.

        This is the single entry point for all path operations.
        Handles:
        - Platform-aware normalization (slash direction, drive letter)
        - Alias resolution (@ROOT, etc.)
        - ~ and environment variable expansion (cross-platform)
        - Relative -> absolute conversion (with smart deduplication)
        - System directory blocking (is_system_path)

        Returns:
            Resolved absolute path.

        Raises:
            ValueError: when path is invalid or empty
            PermissionError: when resolved path is a system directory
        """
        raw_str = str(raw).strip()
        if not raw_str:
            raise ValueError("Path is empty")

        # Step 1: Normalize for platform
        normalized = self._boundary.normalize_path(raw_str)

        # Step 2: Resolve aliases
        if raw_str == "@ROOT":
            return self.root_dir.resolve()
        elif raw_str.startswith("@ROOT/"):
            suffix = raw_str[len("@ROOT/"):]
            return (self.root_dir / suffix).resolve()
        elif raw_str.startswith("@"):
            raw_str = self._aliases.get(raw_str, raw_str)
            normalized = self._boundary.normalize_path(raw_str)

        # Step 3: Expand ~ and environment variables
        p = self._boundary.expand_path(Path(raw_str))

        root_resolved = self.root_dir.resolve()

        if p.is_absolute():
            resolved = p.resolve()

            # Check if resolved is inside root_dir
            try:
                rel = resolved.relative_to(root_resolved)
                # Case A: resolved is inside root_dir.
                # Check for hallucinated segment (duplicate workspace dir name).
                root_parts = list(root_resolved.parts)
                rel_parts = list(rel.parts)
                if root_parts and rel_parts:
                    lcp_len = 0
                    for i in range(min(len(root_parts[-1]), len(rel_parts[0]))):
                        if root_parts[-1][i] == rel_parts[0][i]:
                            lcp_len = i + 1
                        else:
                            break
                    if lcp_len > 2 and rel_parts[0] != root_parts[-1]:
                        raise PathResolutionRejected(
                            f"Path '{raw_str}' appears to contain a duplicate "
                            f"workspace directory segment ('{rel_parts[0]}' "
                            f"extends '{root_parts[-1]}'). "
                            f"Use a relative path from workspace root instead."
                        )
                # OK: inside root_dir with no hallucinated segments.
                logger.debug(
                    "resolve_path resolved: raw='{}' root_dir='{}' result='{}' platform='{}'",
                    raw_str,
                    self.root_dir,
                    resolved,
                    self.platform,
                )
                return resolved

            except ValueError:
                # Resolved path is outside root_dir.
                # Strip the common prefix with root_dir and rebase onto root_dir.
                # E.g. root_dir="/workspace/2026-04-19/abc", path="/workspace/output/file.pdf"
                #      → root_dir / "output/file.pdf"
                root_parts = list(root_resolved.parts)
                resolved_parts = list(resolved.parts)
                prefix_len = 0
                for i, (r, w) in enumerate(zip(resolved_parts, root_parts)):
                    if r == w:
                        prefix_len = i + 1
                    else:
                        break
                # Check for hallucinated segment: "workspace/workspace/..." pattern
                if prefix_len < len(resolved_parts) and prefix_len < len(root_parts):
                    first_extra = resolved_parts[prefix_len]
                    if root_parts and first_extra == root_parts[-1]:
                        raise PathResolutionRejected(
                            f"Path '{raw_str}' appears to contain a duplicate "
                            f"directory segment ('{first_extra}'). "
                            f"Use a relative path from workspace root instead."
                        )
                suffix = resolved_parts[prefix_len:]
                if suffix:
                    resolved = root_resolved.joinpath(*suffix)
                else:
                    resolved = root_resolved
                logger.debug(
                    "resolve_path rebased: raw='{}' root_dir='{}' prefix_len={} result='{}'",
                    raw_str,
                    self.root_dir,
                    prefix_len,
                    resolved,
                )
        else:
            # Relative path: merge with root_dir, no deduplication needed.
            # Deduplication was removed because it incorrectly deduplicated:
            # 1. Absolute paths whose leading "/" was stripped by normalize_path
            #    (e.g. "/workspace/output/file.txt" became "workspace/output/file.txt")
            # 2. Paths that naturally share path component names with root_dir
            # The correct behavior: relative path → root_dir / rel_path
            resolved = (self.root_dir / p).resolve()

        logger.info(
            "[ANALYSIS-LOG] resolve_path: raw='{}' root_dir='{}' is_abs={} result='{}' platform='{}'",
            raw_str,
            self.root_dir,
            p.is_absolute(),
            resolved,
            self.platform,
        )

        # Step 5: Only check system directories for RELATIVE paths.
        # Absolute paths (like /Applications, /usr/bin) bypass this check.
        # System directory enforcement for absolute paths is handled by PathPolicyHook.
        if not p.is_absolute() and self._boundary.is_system_path(resolved):
            logger.warning(
                "Path '{}' resolved to '{}' is a system directory and is blocked.",
                raw_str,
                resolved,
            )
            raise PermissionError(
                f"Path '{raw_str}' (resolved: {resolved}) is a system directory and is not allowed"
            )
        return resolved

    def register_alias(self, alias: str, path: str) -> None:
        """Register a path alias (for artifact tracking)."""
        if alias.startswith("@"):
            object.__setattr__(self, "_aliases", {**self._aliases, alias: path})

    @classmethod
    def from_skill(
        cls,
        *,
        config: "SkillConfig",
        skill: Skill,
        workspace_dir: Path,
        session_id: str | None = None,
    ) -> "RuntimeToolContext":
        """Create RuntimeToolContext from skill configuration."""
        platform = detect_platform()
        boundary = get_boundary(platform)

        skill_root = None
        if skill.source_dir:
            skill_root = Path(skill.source_dir)
        else:
            try:
                skills_dir = config.skills_dir
                candidates = [
                    skills_dir / skill.name,
                    skills_dir / to_kebab_case(skill.name),
                ]
                for candidate in candidates:
                    if candidate.exists():
                        skill_root = candidate
                        logger.debug(
                            "Resolved skill_root from fallback for skill '{}': {}",
                            skill.name,
                            candidate,
                        )
                        break
                if skill_root is None:
                    logger.debug(
                        "Skill '{}' has no source_dir and no fallback path found in {}",
                        skill.name,
                        skills_dir,
                    )
            except Exception as e:
                logger.warning(
                    "Failed resolving fallback skill_root for skill '{}': {}",
                    skill.name,
                    e,
                )
                skill_root = None

        root_dir = workspace_dir.resolve()

        try:
            root_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Failed to ensure root dir {}: {}", root_dir, e)

        return cls(
            workspace_dir=workspace_dir,
            root_dir=root_dir,
            skill_root=skill_root,
            skill_name=skill.name,
            session_id=session_id or "",
            platform=platform,
            _boundary=boundary,
        )


# Backward-compat alias
ToolContext = RuntimeToolContext
