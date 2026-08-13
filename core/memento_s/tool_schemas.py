# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for the agent-facing skill tool schemas."""

from core.memento_s.skill_dispatch.base import SKILL_SEARCH_EXECUTE_SCHEMAS

AGENT_TOOL_SCHEMAS = SKILL_SEARCH_EXECUTE_SCHEMAS[:2]

__all__ = ["AGENT_TOOL_SCHEMAS"]
