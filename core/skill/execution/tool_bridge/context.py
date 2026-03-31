"""Tool bridge execution context."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.skill.config import SkillConfig
from middleware.utils.path_security import build_allow_roots, is_path_within
from core.skill.schema import Skill
from core.utils.text import to_kebab_case
from utils.logger import get_logger

logger = get_logger(__name__)

# Path-related parameter keys (centralized here to avoid duplication)
PATH_LIKE_KEYS: frozenset[str] = frozenset(
    {
        "path",
        "target_path",
        "file_path",
        "src_path",
        "dst_path",
        "source",
        "destination",
        "directory",
        "dir",
        "work_dir",
        "workspace",
        "output_path",
        "input_path",
        "from_path",
        "to_path",
        "root",
        "base_dir",
        "output_dir",
        "input_dir",
        "filename",
        "filepath",
    }
)


@dataclass(frozen=True)
class ToolContext:
    """Execution context for tool bridge.

    Provides unified path resolution and boundary checking.
    """

    workspace_dir: Path
    root_dir: Path  # Unified root for all operations (replaces primary_root + session_output_dir)
    skill_root: Path | None = None
    allow_roots: tuple[Path, ...] = ()

    # Simplified alias management (in-memory only)
    _aliases: dict[str, str] = field(default_factory=dict, repr=False)

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

        # Find the maximum possible overlap (up to min length)
        max_overlap = min(len(root_parts), len(rel_parts))

        # Check for overlapping segments from largest to smallest
        for overlap in range(max_overlap, 0, -1):
            # Compare last 'overlap' elements of root_parts with first 'overlap' of rel_parts
            if root_parts[-overlap:] == rel_parts[:overlap]:
                # Found overlap, merge by taking root_parts + remaining rel_parts
                return root_parts + rel_parts[overlap:]

        # No overlap found, simple concatenation
        return root_parts + rel_parts

    def resolve_path(self, raw: str | Path) -> Path:
        """Unified path resolution: parse → validate → reject-on-violation.

        This is the single entry point for all path operations.
        Handles:
        - Alias resolution (@OUT_ROOT, etc.)
        - Relative → absolute conversion (with smart deduplication)
        - Boundary checking and explicit rejection

        Returns:
            Resolved absolute path if within allow_roots.

        Raises:
            ValueError: when path is invalid or empty
            PermissionError: when resolved path is outside allowed roots
        """
        raw_str = str(raw).strip()
        if not raw_str:
            raise ValueError("Path is empty")

        # Step 1: Resolve aliases
        if raw_str == "@ROOT":
            raw_str = str(self.root_dir)
        elif raw_str.startswith("@ROOT/"):
            suffix = raw_str[len("@ROOT/") :]
            raw_str = str(self.root_dir / suffix)
        elif raw_str.startswith("@"):
            raw_str = self._aliases.get(raw_str, raw_str)

        # Step 2: Parse path
        p = Path(raw_str)

        # Step 3: Convert to absolute (relative paths are under root_dir)
        if not p.is_absolute():
            # Use smart deduplication to avoid nested paths like:
            # /root/a/b/c + c/d/e.txt -> /root/a/b/c/d/e.txt (not /root/a/b/c/c/d/e.txt)
            root_parts = list(self.root_dir.parts)
            rel_parts = list(p.parts)
            merged_parts = self._deduplicate_path_components(root_parts, rel_parts)
            p = Path(*merged_parts)

        try:
            resolved = p.resolve()
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid path '{raw_str}': {e}") from e

        # Step 4: Check boundaries and reject if outside
        if self._is_within_bounds(resolved):
            return resolved

        logger.warning(
            "Path '{}' resolved to '{}' is outside allowed roots.",
            raw_str,
            resolved,
        )
        raise PermissionError(
            f"Path '{raw_str}' (resolved: {resolved}) is outside allowed roots"
        )

    def _is_within_bounds(self, path: Path) -> bool:
        """Check if path is within allowed roots."""
        if not self.allow_roots:
            return True

        for root in self.allow_roots:
            try:
                if is_path_within(path, root):
                    return True
            except (OSError, ValueError):
                continue
        return False

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
    ) -> "ToolContext":
        """Create ToolContext from skill configuration."""
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

        _ = session_id  # reserved for future task/session scoping
        root_dir = workspace_dir.resolve()

        # Ensure root directory exists
        try:
            root_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Failed to ensure root dir {}: {}", root_dir, e)

        # Build allow_roots as access boundaries
        extra_roots = [config.skills_dir] if config.skills_dir else []
        allow_roots = tuple(
            build_allow_roots(
                workspace_dir=root_dir,
                skill_root=skill_root,
                extra_roots=extra_roots,
            )
        )

        return cls(
            workspace_dir=workspace_dir,
            root_dir=root_dir,
            skill_root=skill_root,
            allow_roots=allow_roots,
        )
