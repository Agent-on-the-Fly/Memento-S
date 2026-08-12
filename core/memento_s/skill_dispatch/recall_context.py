# SPDX-License-Identifier: Apache-2.0
"""Context recall handler — delegates to RecallEngine for session memory."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = __import__("utils.logger", fromlist=["get_logger"]).get_logger(__name__)


class ContextRecallHandler:
    """Handles recall_context: delegates to InfraService.recall_engine.recall_session_memory().

    This handler is intentionally thin — recall logic lives in infra/memory/recall_engine.py.
    """

    def __init__(self, infra: Any | None, session_id: str | None) -> None:
        self._infra = infra
        self._session_id = session_id

    async def recall(self, args: dict[str, Any]) -> str:
        """Execute recall_context: search session memory via RecallEngine."""
        query = args.get("query", "")
        if not query:
            return json.dumps(
                {"ok": False, "error": "query is required", "_no_persist": True}
            )

        if self._infra is None:
            return json.dumps(
                {"ok": False, "error": "infra not available", "_no_persist": True}
            )

        engine = self._infra.recall_engine
        result = engine.recall_session_memory(query)
        return json.dumps(
            {"ok": True, "result": result.to_display(query), "_no_persist": True}
        )
