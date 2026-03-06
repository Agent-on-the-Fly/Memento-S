
from __future__ import annotations

import logging
from typing import Any

from core.skills import SkillManager

logger = logging.getLogger(__name__)


class SkillStatusAdapter:

    def __init__(self, skill_manager: SkillManager) -> None:
        self._skill_manager = skill_manager

    @property
    def skill_manager(self) -> SkillManager:
        return self._skill_manager

    def has_skills(self) -> bool:
        return len(self._skill_manager.skill_names) > 0

    def get_skill_names(self) -> list[str]:
        return self._skill_manager.skill_names

    def get_skill_count(self) -> int:
        return len(self._skill_manager)

    def build_skills_summary(self) -> str:
        return self._skill_manager.build_skills_summary()
