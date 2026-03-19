"""Configuration migrations module.

Handles configuration version migrations and upgrades.
"""

from .migrator import (
    ConfigMigrator,
    MigrationResult,
    is_newer_version,
    merge_configs,
    parse_semver,
)
from .template_merge import merge_template_defaults

__all__ = [
    "ConfigMigrator",
    "MigrationResult",
    "is_newer_version",
    "merge_configs",
    "merge_template_defaults",
    "parse_semver",
]
