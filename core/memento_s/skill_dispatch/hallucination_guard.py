# SPDX-License-Identifier: Apache-2.0
"""Hallucination guard — validates skill names before execution."""

from __future__ import annotations

import json
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from core.skill.gateway import SkillGateway

logger = __import__("utils.logger", fromlist=["get_logger"]).get_logger(__name__)


class HallucinationGuard:
    """Intercepts and validates skill names that LLM attempts to call directly.

    If the name is a valid installed skill, converts the call to execute_skill.
    If not, returns an error with suggestions.
    """

    def __init__(self, gateway: SkillGateway) -> None:
        self._gateway = gateway

    async def intercept(
        self,
        tool_name: str,
        args: dict[str, Any],
        execute_fn: Callable[..., Any],
    ) -> str:
        """Validate tool_name, convert if valid, or return error."""
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
        return await execute_fn(converted_args)

    async def _resolve_installed_skill_names(self) -> set[str]:
        """Return the set of currently installed skill names."""
        try:
            manifests = await self._gateway.discover()
            return {m.name for m in manifests}
        except Exception:
            logger.opt(exception=True).warning("Failed to discover installed skills")
            return set()
