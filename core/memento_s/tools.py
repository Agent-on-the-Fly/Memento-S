"""Tool schemas and unified execution gateway.

Tool schemas define the function-calling interface exposed to the LLM agent.
ToolDispatcher routes all tool calls through consistent policy checking,
rate limiting, and logging.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from core.skill.gateway import SkillGateway
from utils.debug_logger import log_tool_start, log_tool_end

from .policies import PolicyManager

logger = logging.getLogger(__name__)

TOOL_SEARCH_SKILL = "search_skill"
TOOL_EXECUTE_SKILL = "execute_skill"


# ═══════════════════════════════════════════════════════════════════
# Tool schemas
# ═══════════════════════════════════════════════════════════════════

SKILL_SEARCH_EXECUTE_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_SEARCH_SKILL,
            "description": "Search relevant skills by natural language query, then choose one skill_name for execute_skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language intent to search skills for.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Max number of candidate skills to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_EXECUTE_SKILL,
            "description": "Execute one selected skill with skill-specific parameters. Each skill declares its own parameter schema in its manifest. Parameters are passed via the 'args' object. Common pattern: use 'request' field in args for natural language descriptions, or skill-specific fields like 'path', 'operation', etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Exact skill name to execute.",
                    },
                    "args": {
                        "type": "object",
                        "description": "Skill-specific parameters as declared in the skill's manifest. Use 'request' key for natural language descriptions (e.g., {'request': 'search for quantum computing'}), or structured parameters (e.g., {'operation': 'read', 'path': '/tmp/file.txt'}).",
                    },
                },
                "required": ["skill_name", "args"],
            },
        },
    },
]

AGENT_TOOL_SCHEMAS: list[dict[str, Any]] = SKILL_SEARCH_EXECUTE_SCHEMAS


# ═══════════════════════════════════════════════════════════════════
# Tool dispatcher
# ═══════════════════════════════════════════════════════════════════


class ToolDispatcher:
    """Unified entry point for executing all tools under policy guard.

    Handles:
    - search_skill / execute_skill via SkillGateway
    """

    def __init__(
        self,
        policy_manager: PolicyManager,
        skill_gateway: SkillGateway,
        session_id: str = "",
    ):
        self.policy_manager = policy_manager
        self._gateway = skill_gateway
        self._session_id = session_id
        self._last_skill_refresh_ts: float = 0.0
        self._refresh_interval_sec: float = 1.0
        self._session_searched: dict[str, bool] = {}

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id
        self._session_searched.setdefault(session_id, False)

    def set_skill_gateway(self, gateway: SkillGateway) -> None:
        self._gateway = gateway

    def mark_session_searched(self, session_id: str) -> None:
        """Mark a session as having performed a skill search."""
        self._session_searched[session_id] = True

    async def execute(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute an agent-exposed tool by name."""
        start_time = time.monotonic()
        call_id = f"{tool_name}_{int(start_time * 1000)}"

        log_tool_start(tool_name, args, call_id)

        try:
            if tool_name == TOOL_SEARCH_SKILL:
                result = await self._search_skill(args)
            elif tool_name == TOOL_EXECUTE_SKILL:
                result = await self._execute_skill(args)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            duration = time.monotonic() - start_time
            log_tool_end(tool_name, result, duration, success=True)
            return result

        except Exception as e:
            duration = time.monotonic() - start_time
            error_result = json.dumps({"ok": False, "error": str(e)})
            log_tool_end(tool_name, error_result, duration, success=False)
            raise

    async def _search_skill(self, args: dict[str, Any]) -> str:
        """Search for skills. Local/builtin skills are already in the system
        prompt, so this only returns *additional* cloud skills."""

        await self._refresh_skills_if_needed()

        query = str(args.get("query", "")).strip()
        k = int(args.get("k", 5) or 5)

        if not query:
            return json.dumps({
                "ok": False,
                "status": "failed",
                "error_code": "INVALID_INPUT",
                "summary": "query is required for search_skill",
            }, ensure_ascii=False)

        local_models = self._gateway.discover()
        local_count = len(local_models)
        logger.info(
            "search_skill: local_in_context=%d, local_names=%s",
            local_count,
            [m.name for m in local_models],
        )

        cloud_skills = []
        try:
            cloud_results = await self._gateway.search(query, k=k, cloud_only=True)
            cloud_skills = [m for m in cloud_results if m.governance.source == "cloud"]
        except Exception as e:
            logger.debug("Cloud search failed: {}", e)

        output = [
            {
                "name": m.name,
                "description": m.description,
                "source": m.governance.source,
                "parameters": m.parameters,
                "execution_mode": m.execution_mode,
            }
            for m in cloud_skills
        ]

        self._session_searched[self._session_id] = True

        payload: dict[str, Any] = {
            "ok": True,
            "status": "success",
            "summary": (
                f"{local_count} local skills already in context, "
                f"{len(output)} additional cloud skills found"
            ),
            "output": output,
            "diagnostics": {"query": query, "local_in_context": local_count},
        }
        return json.dumps(payload, ensure_ascii=False)

    async def _execute_skill(self, args: dict[str, Any]) -> str:
        await self._refresh_skills_if_needed()

        args = dict(args)
        skill_name = args.pop("skill_name", "").strip().rstrip(">").strip()
        logger.info(
            "ToolDispatcher._execute_skill: skill_name={}, query_preview={}",
            skill_name,
            str(args.get("request", ""))[:200],
        )
        if not skill_name:
            payload = {
                "ok": False,
                "status": "failed",
                "error_code": "INVALID_INPUT",
                "summary": "skill_name is required for execute_skill",
            }
            return json.dumps(payload, ensure_ascii=False)

        if not self._session_searched.get(self._session_id, False):
            local_models = self._gateway.discover()
            local_names = {m.name for m in local_models}
            logger.info(
                "execute_skill: session_searched=false, skill_name=%s, local_names=%s",
                skill_name,
                sorted(local_names),
            )
            if skill_name not in local_names:
                return json.dumps({
                    "ok": False,
                    "status": "failed",
                    "error_code": "SEARCH_REQUIRED",
                    "summary": "Call search_skill first, then execute_skill.",
                    "skill_name": skill_name,
                    "diagnostics": {"hint": "search_skill(query=...) -> execute_skill(skill_name=..., request=...)"},
                }, ensure_ascii=False)

        skill_args = args.get("args", {})
        if not isinstance(skill_args, dict):
            skill_args = {}

        if "request" not in skill_args:
            fallback_request = args.get("request")
            if isinstance(fallback_request, str) and fallback_request.strip():
                skill_args["request"] = fallback_request.strip()

        envelope = await self._gateway.execute(
            skill_name=skill_name,
            params=skill_args,
        )

        logger.info(
            "ToolDispatcher._execute_skill: skill_name={}, result_ok={}, summary={}",
            skill_name,
            envelope.ok,
            (envelope.summary or "")[:200],
        )

        payload: dict[str, Any] = {
            "ok": envelope.ok,
            "status": envelope.status.value,
            "summary": envelope.summary,
            "skill_name": envelope.skill_name,
            "output": envelope.output,
        }
        if envelope.error_code:
            payload["error_code"] = envelope.error_code.value
        if envelope.outputs:
            payload["outputs"] = envelope.outputs
        if envelope.artifacts:
            payload["artifacts"] = envelope.artifacts
        if envelope.diagnostics:
            payload["diagnostics"] = envelope.diagnostics
        return json.dumps(payload, ensure_ascii=False)

    async def _refresh_skills_if_needed(self) -> None:
        """Refresh local skills from disk with lightweight throttling."""
        now = time.monotonic()
        if now - self._last_skill_refresh_ts < self._refresh_interval_sec:
            return

        self._last_skill_refresh_ts = now

        library = getattr(self._gateway, "_library", None)
        refresh_fn = getattr(library, "refresh_from_disk", None)
        if refresh_fn is None:
            return

        try:
            await refresh_fn()
        except Exception as e:
            logger.debug("refresh_from_disk failed before search: {}", e)
