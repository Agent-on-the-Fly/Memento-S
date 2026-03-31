"""Tool schemas and unified execution gateway.

Tool schemas define the function-calling interface exposed to the LLM agent.
ToolDispatcher routes all tool calls through consistent policy checking,
rate limiting, and logging.
"""

from __future__ import annotations

import json
import logging
import time
from difflib import get_close_matches
from typing import Any, Callable

from core.skill.gateway import SkillGateway
from core.skill.schema import DiscoverStrategy
from middleware.config import g_config
from utils.debug_logger import log_tool_start, log_tool_end

logger = logging.getLogger(__name__)

SkillsChangedCallback = Callable[[], None]

TOOL_SEARCH_SKILL = "search_skill"
TOOL_EXECUTE_SKILL = "execute_skill"
TOOL_DOWNLOAD_SKILL = "download_skill"
TOOL_CREATE_SKILL = "create_skill"
TOOL_ASK_USER = "ask_user"


# ═══════════════════════════════════════════════════════════════════
# Tool schemas
# ═══════════════════════════════════════════════════════════════════

SKILL_SEARCH_EXECUTE_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_SEARCH_SKILL,
            "description": "Search for relevant skills by natural language query across BOTH local installed skills and the remote skill server. Use this first when you don't know which skill to use.",
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
            "description": "Execute a LOCAL installed skill. MUST NOT be used to execute remote or non-existent skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Exact logic name of the skill to execute (e.g., 'weather_fetcher').",
                    },
                    "request": {
                        "type": "string",
                        "description": "Natural language description of what you want the skill to do.",
                    },
                },
                "required": ["skill_name", "request"],
                "additionalProperties": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_DOWNLOAD_SKILL,
            "description": "Download and install a remote skill from the skill server to the local environment. Use this ONLY AFTER search_skill has found a matching remote skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The exact name of the remote skill found via search_skill.",
                    }
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_CREATE_SKILL,
            "description": "Create a NEW skill from scratch ONLY when search_skill returns NO results from both local and remote. This writes the skill to the local file system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "Natural language description of what skill to create, including name, purpose, language, and functionality details.",
                    },
                },
                "required": ["request"],
            },
        },
    },
]

TOOL_ASK_USER_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_ASK_USER,
        "description": "Ask the user a question when you need information that only the user can provide. The execution will pause until the user responds.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                },
            },
            "required": ["question"],
        },
    },
}

AGENT_TOOL_SCHEMAS: list[dict[str, Any]] = SKILL_SEARCH_EXECUTE_SCHEMAS + [
    TOOL_ASK_USER_SCHEMA
]


# ═══════════════════════════════════════════════════════════════════
# Tool dispatcher
# ═══════════════════════════════════════════════════════════════════


class ToolDispatcher:
    """Unified entry point for executing all tools under policy guard.

    Handles:
    - search_skill / execute_skill / download_skill / create_skill via SkillGateway
    """

    def __init__(
        self,
        skill_gateway: SkillGateway,
        session_id: str = "",
        on_skills_changed: SkillsChangedCallback | None = None,
        on_skill_step: Any | None = None,
    ):
        self._gateway = skill_gateway
        self._session_id = session_id
        self._on_skills_changed = on_skills_changed
        self._on_skill_step = on_skill_step

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id

    def set_on_skills_changed(self, callback: SkillsChangedCallback | None) -> None:
        """Set callback invoked after create_skill / download_skill succeeds."""
        self._on_skills_changed = callback

    def set_on_skill_step(self, callback: Any | None) -> None:
        """Set callback invoked during skill execution for each step."""
        self._on_skill_step = callback

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
            elif tool_name == TOOL_DOWNLOAD_SKILL:
                result = await self._download_skill(args)
            elif tool_name == TOOL_CREATE_SKILL:
                result = await self._create_skill(args)
            else:
                # Hallucination Interceptor: Auto-convert skill name to execute_skill
                result = await self._handle_hallucinated_skill_call(tool_name, args)

            duration = time.monotonic() - start_time
            log_tool_end(tool_name, result, duration, success=True)
            return result

        except Exception as e:
            duration = time.monotonic() - start_time
            error_result = json.dumps({"ok": False, "error": str(e)})
            log_tool_end(tool_name, error_result, duration, success=False)
            raise

    async def _handle_hallucinated_skill_call(
        self, tool_name: str, args: dict[str, Any]
    ) -> str:
        """Intercept hallucinated skill calls — validate existence before converting."""
        logger.warning(
            "Hallucination detected: LLM tried to call '{}' directly.",
            tool_name,
        )
        installed_names = await self._resolve_installed_skill_names()

        if tool_name not in installed_names:
            close = get_close_matches(tool_name, installed_names, n=3, cutoff=0.5)
            suggestion = (
                f" Did you mean one of: {', '.join(close)}?"
                if close
                else " Use search_skill to find available skills."
            )
            return json.dumps(
                {
                    "ok": False,
                    "status": "failed",
                    "error_code": "UNKNOWN_TOOL",
                    "summary": f"'{tool_name}' is not a valid tool or installed skill.{suggestion}",
                },
                ensure_ascii=False,
            )

        request = args.get("request", "")
        if not request:
            request = args.get("query", "") or "Execute the skill"
        converted_args = {
            "skill_name": tool_name,
            "request": str(request),
            **{
                k: v
                for k, v in args.items()
                if k not in ("skill_name", "request", "query")
            },
        }
        return await self._execute_skill(converted_args)

    async def _resolve_installed_skill_names(self) -> set[str]:
        """Return the set of currently installed skill names."""
        try:
            manifests = await self._gateway.discover()
            return {m.name for m in manifests}
        except Exception:
            logger.warning("Failed to discover installed skills", exc_info=True)
            return set()

    async def _search_skill(self, args: dict[str, Any]) -> str:
        """Search for skills across local and cloud sources with guided output."""
        query = str(args.get("query", "")).strip()
        k = int(args.get("k", 5) or 5)

        if not query:
            return json.dumps(
                {
                    "ok": False,
                    "status": "failed",
                    "error_code": "INVALID_INPUT",
                    "summary": "query is required for search_skill",
                },
                ensure_ascii=False,
                default=str,
            )

        all_skills = []
        try:
            all_skills = await self._gateway.search(query, k=k, cloud_only=False)
        except Exception as e:
            logger.warning("Skill search failed: {}", e)
            return json.dumps(
                {
                    "ok": False,
                    "status": "failed",
                    "error_code": "SEARCH_FAILED",
                    "summary": f"Skill search failed: {e}",
                    "diagnostics": {"query": query},
                },
                ensure_ascii=False,
                default=str,
            )

        # 分离本地和云端技能
        local_skills = [m for m in all_skills if m.governance.source == "local"]
        cloud_skills = [m for m in all_skills if m.governance.source == "cloud"]

        # 构建强引导性的输出
        output_lines = []

        for skill in local_skills:
            output_lines.append(
                f"Found [Local] skill: `{skill.name}`. Status: Installed. "
                f"You can `execute_skill(skill_name='{skill.name}', request='...')` directly."
            )

        for skill in cloud_skills:
            output_lines.append(
                f"Found [Remote] skill: `{skill.name}`. Status: Not Installed. "
                f"You MUST call `download_skill(skill_name='{skill.name}')` before executing."
            )

        if not output_lines:
            return json.dumps(
                {
                    "ok": True,
                    "status": "success",
                    "summary": f"No skills found for '{query}'.",
                    "output": "No skills found locally or remotely. You are authorized to use `create_skill` immediately.",
                    "diagnostics": {"query": query, "results_count": 0},
                },
                ensure_ascii=False,
                default=str,
            )

        payload: dict[str, Any] = {
            "ok": True,
            "status": "success",
            "summary": f"Found {len(all_skills)} skills matching '{query}'",
            "output": "\n".join(output_lines),
            "diagnostics": {
                "query": query,
                "results_count": len(all_skills),
                "local_count": len(local_skills),
                "cloud_count": len(cloud_skills),
            },
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    async def _execute_skill(self, args: dict[str, Any]) -> str:
        args = dict(args)
        skill_name = args.pop("skill_name", "").strip("> \t\n")
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
            return json.dumps(payload, ensure_ascii=False, default=str)

        # 扁平化参数：剩余的所有参数（包括 request）都传给 skill
        skill_args = args

        envelope = await self._gateway.execute(
            skill_name=skill_name,
            params=skill_args,
            session_id=self._session_id,
            on_step=self._on_skill_step,
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
        return json.dumps(payload, ensure_ascii=False, default=str)

    async def _download_skill(self, args: dict[str, Any]) -> str:
        """Download a cloud skill to local storage."""
        skill_name = str(args.get("skill_name", "")).strip()

        if not skill_name:
            return json.dumps(
                {
                    "ok": False,
                    "status": "failed",
                    "error_code": "INVALID_INPUT",
                    "summary": "skill_name is required for download_skill",
                },
                ensure_ascii=False,
                default=str,
            )

        try:
            skill = await self._gateway.install(skill_name)
            if skill:
                self._notify_skills_changed()
                return json.dumps(
                    {
                        "ok": True,
                        "status": "success",
                        "summary": f"Skill '{skill_name}' downloaded and installed successfully.",
                        "skill_name": skill.name,
                        "output": f"Skill '{skill_name}' is now installed and ready to use via `execute_skill(skill_name='{skill_name}', request='...')`",
                    },
                    ensure_ascii=False,
                    default=str,
                )
            else:
                return json.dumps(
                    {
                        "ok": False,
                        "status": "failed",
                        "error_code": "DOWNLOAD_FAILED",
                        "summary": f"Failed to download skill '{skill_name}'. It may not exist in the cloud or the download failed.",
                    },
                    ensure_ascii=False,
                    default=str,
                )
        except Exception as e:
            logger.exception("Failed to download skill")
            return json.dumps(
                {
                    "ok": False,
                    "status": "failed",
                    "error_code": "INTERNAL_ERROR",
                    "summary": f"Error downloading skill: {str(e)}",
                },
                ensure_ascii=False,
                default=str,
            )

    async def _create_skill(self, args: dict[str, Any]) -> str:
        """Create a new skill by delegating to skill-creator."""
        request = str(args.get("request", "")).strip()

        # Validate required fields
        if not request:
            return json.dumps(
                {
                    "ok": False,
                    "status": "failed",
                    "error_code": "INVALID_INPUT",
                    "summary": "request is required for create_skill - describe what skill you want to create",
                },
                ensure_ascii=False,
                default=str,
            )

        # Delegate to skill-creator
        execute_args = {
            "skill_name": "skill-creator",
            "request": request,
        }

        logger.info(
            "Delegating create_skill to skill-creator: request_preview={}",
            request[:100],
        )
        result = await self._execute_skill(execute_args)

        # Parse result and add skills directory info
        try:
            result_dict = json.loads(result)
            if result_dict.get("ok"):
                self._notify_skills_changed()
                return json.dumps(result_dict, ensure_ascii=False, default=str)
        except (json.JSONDecodeError, KeyError):
            pass

        return result

    def _notify_skills_changed(self) -> None:
        """Invoke the skills-changed callback if registered."""
        if self._on_skills_changed is not None:
            try:
                self._on_skills_changed()
            except Exception:
                logger.warning("on_skills_changed callback failed", exc_info=True)
