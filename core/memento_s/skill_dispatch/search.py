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

        output = [
            {
                "name": skill.name,
                "description": skill.description,
                "source": skill.governance.source,
                "execution_mode": skill.execution_mode.value,
            }
            for skill in cloud_skills
        ]

        if not all_skills:
            return json.dumps(
                {
                    "ok": True,
                    "status": "success",
                    "summary": f"No skills found for '{query}'.",
                    "output": [],
                    "metrics": {"k": k, "cloud_count": 0},
                    "diagnostics": {"query": query, "local_in_context": 0},
                },
                ensure_ascii=False,
                default=str,
            )

        payload: dict[str, Any] = {
            "ok": True,
            "status": "success",
            "summary": (
                f"Found {len(cloud_skills)} cloud skills matching '{query}'; "
                f"{len(local_skills)} local skills already in context."
            ),
            "output": output,
            "metrics": {"k": k, "cloud_count": len(cloud_skills)},
            "diagnostics": {
                "query": query,
                "local_in_context": len(local_skills),
            },
        }
        return json.dumps(payload, ensure_ascii=False, default=str)
