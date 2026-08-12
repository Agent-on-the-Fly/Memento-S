# SPDX-License-Identifier: Apache-2.0
"""skill_dispatch — split handlers for SkillDispatcher responsibilities.

Handlers:
    - SkillSearchHandler: search local + cloud skills
    - SkillExecutionHandler: execute/download/create skills
    - HallucinationGuard: validate hallucinated skill calls
    - ContextRecallHandler: recall_context tool
    - CreatedSkillInstaller: post skill-creator install logic
"""

from .base import (
    AGENT_TOOL_SCHEMAS,
    SKILL_SEARCH_EXECUTE_SCHEMAS,
    TOOL_ASK_USER,
    TOOL_ASK_USER_SCHEMA,
    TOOL_CREATE_SKILL,
    TOOL_DOWNLOAD_SKILL,
    TOOL_EXECUTE_SKILL,
    TOOL_RECALL_CONTEXT,
    TOOL_SEARCH_SKILL,
    TOOL_RECALL_CONTEXT_SCHEMA,
)

from .execution import SkillExecutionHandler
from .hallucination_guard import HallucinationGuard
from .post_install import CreatedSkillInstaller
from .recall_context import ContextRecallHandler
from .search import SkillSearchHandler
from .dispatcher import SkillDispatcher

__all__ = [
    # base
    "AGENT_TOOL_SCHEMAS",
    "SKILL_SEARCH_EXECUTE_SCHEMAS",
    "TOOL_ASK_USER",
    "TOOL_ASK_USER_SCHEMA",
    "TOOL_CREATE_SKILL",
    "TOOL_DOWNLOAD_SKILL",
    "TOOL_EXECUTE_SKILL",
    "TOOL_RECALL_CONTEXT",
    "TOOL_SEARCH_SKILL",
    "TOOL_RECALL_CONTEXT_SCHEMA",
    # handlers
    "SkillExecutionHandler",
    "SkillSearchHandler",
    "HallucinationGuard",
    "ContextRecallHandler",
    "CreatedSkillInstaller",
    # facade
    "SkillDispatcher",
]
