# SPDX-License-Identifier: Apache-2.0
"""Skill search handler — searches local and cloud skills with guided output."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.skill.gateway import SkillGateway

logger = __import__("utils.logger", fromlist=["get_logger"]).get_logger(__name__)


class SkillSearchHandler:
    """Handles search_skill: queries gateway.search across local and cloud sources."""

    def __init__(self, gateway: SkillGateway) -> None:
        self._gateway = gateway

    async def search(self, args: dict[str, Any]) -> str:
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

        local_skills = [m for m in all_skills if m.governance.source == "local"]
        cloud_skills = [m for m in all_skills if m.governance.source == "cloud"]

        output_lines: list[str] = []

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
