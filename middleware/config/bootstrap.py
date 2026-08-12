# SPDX-License-Identifier: Apache-2.0
"""
Config subsystem bootstrap — orchestrates initialization of all config layers.

This module centralizes all config initialization logic that was previously
scattered across bootstrap.py. It should be called once at application startup.

Usage::

    from middleware.config.bootstrap import bootstrap_configs

    async def main():
        global_config, mcp_mgr, skill_mgr = await bootstrap_configs()
        # use config...
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from middleware.config.config_manager import ConfigManager, g_config
from middleware.config.mcp_config_manager import McpConfigManager, g_mcp_config_manager
from middleware.config.skill_config_manager import SkillConfigManager, skill_config_manager
from middleware.config.schemas.config_models import GlobalConfig

logger = logging.getLogger(__name__)


# ── Private helpers (moved from bootstrap.py) ─────────────────────────────────


def _init_directories(config: GlobalConfig) -> dict[str, Path]:
    """Initialize all required directory structures."""
    config.paths.workspace_dir.mkdir(parents=True, exist_ok=True)
    config.paths.skills_dir.mkdir(parents=True, exist_ok=True)
    config.paths.db_dir.mkdir(parents=True, exist_ok=True)
    config.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    if config.paths.context_dir is not None:
        config.paths.context_dir.mkdir(parents=True, exist_ok=True)

    return {
        "workspace": config.paths.workspace_dir,
        "skills": config.paths.skills_dir,
        "db": config.paths.db_dir,
        "logs": config.paths.logs_dir,
        "context": config.paths.context_dir,
    }


def _init_mcp_config(mcp_manager: McpConfigManager) -> dict[str, Any]:
    """Initialize MCP config: ensure mcp.json exists and load it."""
    mcp_manager.ensure_mcp_config_file()
    return mcp_manager.load()


def _init_skill_config(
    skill_manager: SkillConfigManager, config: GlobalConfig
) -> None:
    """Initialize skill config: ensure skill.json exists and load it."""
    skill_manager.load()


def _perform_config_migration(manager: ConfigManager) -> Any | None:
    """Execute config template merge (no version).

    Only runs at bootstrap:
    - New template fields are added to user config
    - Existing user fields are preserved
    - Fields marked x-managed-by: user are user-controlled
    - Never overwrites any existing user config
    """
    from middleware.config.migrations import MigrationResult
    from middleware.config.schema_meta import SchemaMetadata

    try:
        template = manager.load_user_template()
        user = manager.get_raw_user_config()
        schema = manager.load_schema()
    except FileNotFoundError:
        return None

    merged = SchemaMetadata.merge_respecting_metadata(template, user, schema)

    # Force gateway.enabled = True
    if merged.get("gateway", {}).get("enabled") is not True:
        merged.setdefault("gateway", {})["enabled"] = True

    # Provide default LLM profiles for new users
    if not merged.get("llm", {}).get("profiles"):
        template_llm = template.get("llm", {})
        if template_llm.get("profiles"):
            merged.setdefault("llm", {})
            merged["llm"]["profiles"] = template_llm["profiles"]
            merged["llm"]["active_profile"] = template_llm.get("active_profile", "default")

    # Provide default IM config for new users — deep merge each platform
    import copy

    def _deep_merge_template_onto(target: dict, template: dict) -> None:
        """Recursively fill in missing keys from template into target."""
        for key, template_value in template.items():
            if isinstance(template_value, dict):
                target_value = target.get(key, {})
                if isinstance(target_value, dict):
                    _deep_merge_template_onto(target_value, template_value)
                    target[key] = target_value
                else:
                    target[key] = copy.deepcopy(template_value)
            else:
                if key not in target:
                    target[key] = copy.deepcopy(template_value)

    template_im = template.get("im", {})
    if template_im:
        merged_im = merged.get("im", {})
        if merged_im:
            _deep_merge_template_onto(merged_im, template_im)
            merged["im"] = merged_im

    if json.dumps(user, sort_keys=True) == json.dumps(merged, sort_keys=True):
        return None

    manager.save_user_config_direct(merged)
    return MigrationResult(
        migrated=True,
        old_version=str(user.get("version", "")),
        new_version=str(merged.get("version", "")),
        backup_path=None,
        changes=["Config template merge completed"],
    )


def _ensure_config_version(manager: ConfigManager) -> None:
    """Ensure user config contains a version marker (informational)."""
    try:
        system_version = manager.load_system_config().get("version", "0.2.0")
        user_config = manager.get_raw_user_config()
        if "version" not in user_config:
            user_config["version"] = system_version
            manager.save_user_config_direct(user_config)
            logger.info(f"[config/bootstrap] Added version marker: {system_version}")
    except Exception as e:
        logger.warning(f"[config/bootstrap] Version marker addition failed: {e}")


# ── Public bootstrap function ──────────────────────────────────────────────────


async def bootstrap_configs() -> tuple[
    GlobalConfig,
    McpConfigManager,
    SkillConfigManager,
    Any | None,  # config_migration result
]:
    """Bootstrap all config subsystems in correct dependency order.

    Initialization sequence:
      1. System + User config (ConfigManager)
      2. Directory structure (depends on paths from config)
      3. MCP config (mcp.json)
      4. Skill config (skill.json)

    Returns:
        Tuple of (global_config, mcp_config_manager, skill_config_manager, config_migration)
    """
    logger.info("[config/bootstrap] Starting config subsystem bootstrap...")

    # 1. Ensure config directory and file exist
    g_config.ensure_user_config_dir()
    g_config.ensure_user_config_file()

    # 2. Config migration & version marker
    migration_result = _perform_config_migration(g_config)
    if migration_result:
        logger.info("[config/bootstrap] Config migration completed")

    _ensure_config_version(g_config)

    # 3. Load and validate GlobalConfig
    global_config = g_config.load()
    if global_config.paths.workspace_dir is None:
        raise ValueError("paths.workspace_dir should not be None; check config completion logic")

    # 4. Directory structure
    dirs = _init_directories(global_config)
    logger.info(f"[config/bootstrap] Directories initialized: {list(dirs.keys())}")

    # 5. MCP config
    mcp_config = _init_mcp_config(g_mcp_config_manager)
    logger.info(f"[config/bootstrap] MCP config loaded: enabled={mcp_config.get('enabled', False)}")

    # 6. Skill config
    _init_skill_config(skill_config_manager, global_config)
    logger.info("[config/bootstrap] Skill config loaded")

    logger.info("[config/bootstrap] All config subsystems bootstrapped successfully")
    return global_config, g_mcp_config_manager, skill_config_manager, migration_result