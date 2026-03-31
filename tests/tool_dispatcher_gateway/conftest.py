"""Tool Dispatcher Gateway fixtures

Note: These tests need to be updated to use the new SkillProvider API
which no longer accepts store parameter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from builtin.tools.registry import configure as configure_builtin_tools
from core.memento_s.policies import PolicyManager
from core.memento_s.tool_dispatcher import ToolDispatcher


@pytest.fixture
def real_dispatcher() -> ToolDispatcher:
    """Fixture using the new init_skill_system API.

    TODO: Update to use init_skill_system() which returns the gateway
    """
    workspace = Path(__file__).resolve().parents[2]
    configure_builtin_tools(workspace)

    # TODO: Use new API
    # from core.skill import init_skill_system
    # from core.skill.config import SkillConfig
    # config = SkillConfig(...)
    # gateway = await init_skill_system(config)

    # For now, return None to skip tests
    return None
