# SPDX-License-Identifier: Apache-2.0
"""Post-install handler — copies skill-creator output from workspace to skills_dir."""

from __future__ import annotations

import hashlib
from datetime import datetime
from shutil import copytree
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.skill.gateway import SkillGateway

logger = __import__("utils.logger", fromlist=["get_logger"]).get_logger(__name__)


class CreatedSkillInstaller:
    """Installs newly created skills from workspace into skills_dir and indexes them.

    After skill-creator succeeds, it writes the skill to:
      workspace/YYYY-MM-DD/{short_id}/{skill_name}/
    where short_id = md5(session_id)[:8].
    execute_skill only searches skills_dir, so this handler copies it there
    and syncs the DB + vector index.
    """

    def __init__(self, gateway: SkillGateway, session_id: str | None) -> None:
        self._gateway = gateway
        self._session_id = session_id

    async def install_from_workspace(self) -> None:
        """Copy newly created skill(s) from workspace to skills_dir and index them."""
        from core.skill.loader import load_from_dir

        workspace = self._gateway._config.workspace_dir

        date_str = datetime.now().strftime("%Y-%m-%d")
        raw_id = self._session_id or ""
        if raw_id:
            short_id = hashlib.md5(raw_id.encode()).hexdigest()[:8]
        else:
            short_id = "default"
        workspace_skill_root = workspace / date_str / short_id

        if not workspace_skill_root.exists():
            logger.warning(
                "_install_created_skill: workspace dir not found: {}",
                workspace_skill_root,
            )
            return

        skill_dirs = [
            d
            for d in workspace_skill_root.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        ]

        for skill_dir in skill_dirs:
            skill_name = skill_dir.name
            target_dir = self._gateway._config.skills_dir / skill_name

            if target_dir.exists():
                logger.debug(
                    "_install_created_skill: {} already in skills_dir, skipping",
                    skill_name,
                )
                continue

            try:
                copytree(skill_dir, target_dir)
                logger.info(
                    "_install_created_skill: copied {} -> {}",
                    skill_dir,
                    target_dir,
                )
            except Exception as e:
                logger.warning(
                    "_install_created_skill: failed to copy {}: {}", skill_dir, e
                )
                continue

            try:
                skill = load_from_dir(target_dir, full=True)
                if skill:
                    await self._gateway.skill_store.add_skill(skill)
                    logger.info("_install_created_skill: indexed skill: {}", skill.name)
            except Exception as e:
                logger.warning(
                    "_install_created_skill: failed to index {}: {}", skill_name, e
                )
