# SPDX-License-Identifier: Apache-2.0
"""Tool Dispatcher Gateway fixtures

Note: These tests need to be updated to use the new SkillProvider API
which no longer accepts store parameter.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest
import pytest_asyncio

from core.memento_s.skill_dispatch import SkillDispatcher
from core.skill.gateway import SkillGateway
from core.skill.registry import SkillRegistry
from core.skill.retrieval import LocalRecall, MultiRecall
from core.skill.store import SkillStorage
from core.skill.schema import SkillExecutionOutcome
from middleware.config.skill_config_manager import SkillConfigManager
from shared.schema import SkillConfig


class _FakeAgent:
    async def run(self, skill, query, params, run_dir, session_id, on_step):
        return (
            SkillExecutionOutcome(
                success=True,
                result={"request": query, "skill": skill.name},
                skill_name=skill.name,
            ),
            "",
        )


@pytest_asyncio.fixture
async def real_dispatcher(tmp_path: Path) -> SkillDispatcher:
    """Create a fully isolated gateway with copied builtin skills."""
    project_root = Path(__file__).resolve().parents[2]
    skills_dir = tmp_path / "skills"
    shutil.copytree(project_root / "builtin" / "skills", skills_dir)
    config = SkillConfig(
        skills_dir=skills_dir,
        builtin_skills_dir=project_root / "builtin" / "skills",
        workspace_dir=tmp_path / "workspace",
        cloud_catalog_url=None,
    )
    manager = SkillConfigManager(user_path=tmp_path / "skill.json")
    store = SkillStorage(skills_dir, SkillRegistry(manager))
    await store.init()
    await store.refresh_from_disk()
    gateway = SkillGateway(
        config=config,
        store=store,
        multi_recall=MultiRecall([LocalRecall(skills_dir)]),
        agent=_FakeAgent(),
    )
    dispatcher = SkillDispatcher(skill_gateway=gateway)
    yield dispatcher
    await store.close()
