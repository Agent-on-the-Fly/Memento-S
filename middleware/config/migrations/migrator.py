"""Configuration migration system.

Handles version comparison, migration detection, and config upgrades.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class MigrationResult:
    """Result of a configuration migration."""

    migrated: bool
    old_version: str
    new_version: str
    backup_path: Path | None
    changes: list[str]


def parse_semver(version: str) -> tuple[int, ...] | None:
    """Parse semantic version string (e.g., '1.2.3') into tuple.

    Args:
        version: Version string like "1.2.3"

    Returns:
        Tuple of integers (1, 2, 3) or None if invalid
    """
    if not version:
        return None

    parts = version.split(".")
    if not parts:
        return None

    nums: list[int] = []
    for part in parts:
        if not part.isdigit():
            return None
        nums.append(int(part))

    return tuple(nums)


def is_newer_version(new_version: str, old_version: str) -> bool:
    """Check if new_version is newer than old_version.

    Args:
        new_version: The new version to check
        old_version: The current/old version

    Returns:
        True if new_version > old_version
    """
    new = parse_semver(new_version)
    old = parse_semver(old_version)

    # New version invalid - not newer
    if new is None:
        return False

    # Old version invalid - treat as upgrade
    if old is None:
        return True

    return new > old


def merge_configs(template: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    """Merge old config values into new template.

    Strategy:
    - Keep template defaults for missing keys.
    - Overwrite with user values when present.
    - Preserve user-only keys (important for dynamic maps such as llm.profiles).

    Args:
        template: New template with default values
        old: Old user configuration

    Returns:
        Merged configuration
    """
    if not isinstance(template, dict) or not isinstance(old, dict):
        return old if old is not None else template

    merged: dict[str, Any] = {}

    # 1) Start from template keys (template defaults + user overrides)
    for key, template_value in template.items():
        if key in old:
            old_value = old[key]
            if isinstance(template_value, dict) and isinstance(old_value, dict):
                merged[key] = merge_configs(template_value, old_value)
            else:
                merged[key] = old_value
        else:
            merged[key] = template_value

    # 2) Preserve old keys not present in template
    for key, old_value in old.items():
        if key not in merged:
            merged[key] = old_value

    return merged


def detect_changes(
    old: dict[str, Any], new: dict[str, Any], path: str = ""
) -> list[str]:
    """Detect configuration changes between versions.

    Args:
        old: Old configuration
        new: New configuration
        path: Current path for nested keys

    Returns:
        List of change descriptions
    """
    changes: list[str] = []

    if not isinstance(old, dict) or not isinstance(new, dict):
        if old != new:
            changes.append(f"{path}: {old} -> {new}")
        return changes

    # Check for new keys
    for key in new:
        current_path = f"{path}.{key}" if path else key
        if key not in old:
            changes.append(f"+ {current_path}")
        else:
            changes.extend(detect_changes(old[key], new[key], current_path))

    # Check for removed keys
    for key in old:
        if key not in new:
            current_path = f"{path}.{key}" if path else key
            changes.append(f"- {current_path}")

    return changes


class ConfigMigrator:
    """Handles configuration migrations and upgrades."""

    def __init__(
        self,
        config_file: Path,
        template_file: Path | None = None,
        template_loader: Callable[[], dict[str, Any]] | None = None,
    ):
        """Initialize migrator.

        Args:
            config_file: Path to user config file
            template_file: Path to template file (optional)
            template_loader: Function to load template (optional, for loading from packages)
        """
        self.config_file = config_file
        self.template_file = template_file
        self._template_loader = template_loader

    def _read_json(self, path: Path) -> dict[str, Any]:
        """Read JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        """Write JSON file with formatting."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _create_backup(self, version: str) -> Path:
        """Create backup of current config.

        Args:
            version: Version tag for backup filename

        Returns:
            Path to backup file
        """
        version_tag = version if version else "unknown"
        backup_path = self.config_file.with_name(f"config.v{version_tag}.json")

        # Handle existing backups with incrementing suffix
        if backup_path.exists():
            idx = 1
            while True:
                candidate = self.config_file.with_name(
                    f"config.v{version_tag}.{idx}.json"
                )
                if not candidate.exists():
                    backup_path = candidate
                    break
                idx += 1

        shutil.copy2(self.config_file, backup_path)
        return backup_path

    def load_template(self) -> dict[str, Any]:
        """Load template configuration.

        Priority:
        1. Use template_loader if provided
        2. Use template_file if set and exists
        3. Try to find template relative to config

        Returns:
            Template configuration dict

        Raises:
            FileNotFoundError: If template cannot be found
        """
        # 1. Use custom loader (e.g., ConfigManager.load_template)
        if self._template_loader is not None:
            return self._template_loader()

        # 2. Use template file
        if self.template_file and self.template_file.exists():
            return self._read_json(self.template_file)

        # 3. Try to find template relative to config
        template_path = self.config_file.parent / "user_config_tlp.json"
        if template_path.exists():
            return self._read_json(template_path)

        raise FileNotFoundError(
            f"Template file not found. Searched: {self.template_file}, {template_path}"
        )

    def needs_migration(self) -> tuple[bool, str, str]:
        """Check if migration is needed.

        Returns:
            Tuple of (needs_migration, old_version, new_version)
        """
        template = self.load_template()
        user = self._read_json(self.config_file)

        template_version = str(template.get("version", ""))
        user_version = str(user.get("version", ""))

        needs = is_newer_version(template_version, user_version)
        return needs, user_version, template_version

    def migrate(self) -> MigrationResult:
        """Perform configuration migration if needed.

        Returns:
            MigrationResult with details of the migration
        """
        needs, old_ver, new_ver = self.needs_migration()

        if not needs:
            return MigrationResult(
                migrated=False,
                old_version=old_ver,
                new_version=new_ver,
                backup_path=None,
                changes=[],
            )

        # Load configs
        template = self.load_template()
        user = self._read_json(self.config_file)

        # Create backup
        backup_path = self._create_backup(old_ver)

        # Perform merge
        merged = merge_configs(template, user)
        merged["version"] = new_ver

        # Detect changes
        changes = detect_changes(user, merged)

        # Write new config
        self._write_json(self.config_file, merged)

        return MigrationResult(
            migrated=True,
            old_version=old_ver,
            new_version=new_ver,
            backup_path=backup_path,
            changes=changes,
        )

    def force_migrate(
        self,
        target_version: str,
        transformer: Any = None,
    ) -> MigrationResult:
        """Force migration to specific version with custom transformation.

        Args:
            target_version: Target version string
            transformer: Optional function to transform config dict

        Returns:
            MigrationResult
        """
        user = self._read_json(self.config_file)
        old_ver = str(user.get("version", ""))

        # Create backup
        backup_path = self._create_backup(old_ver)

        # Apply transformation if provided
        if transformer:
            user = transformer(user)

        # Update version
        user["version"] = target_version

        # Detect changes
        changes = [f"Forced migration to {target_version}"]

        # Write config
        self._write_json(self.config_file, user)

        return MigrationResult(
            migrated=True,
            old_version=old_ver,
            new_version=target_version,
            backup_path=backup_path,
            changes=changes,
        )
