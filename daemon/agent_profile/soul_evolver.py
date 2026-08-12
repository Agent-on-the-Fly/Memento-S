# SPDX-License-Identifier: Apache-2.0
"""SOUL.md evolution engine — SoulEvolver.

SoulManager 已移至 core/agent_profile/soul_manager.py。
此文件仅保留 SoulEvolver（异步进化引擎）。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING

from utils.logger import get_logger

from core.agent_profile.soul_manager import SoulManager
from daemon.agent_profile.constants import (
    EVOLVE_SOUL_INTERVAL_SECONDS,
    INPUT_MAX_CHARS,
    MSG_TRUNCATE_CHARS,
)
from daemon.agent_profile.soul_prompts import (
    PASS0_SYSTEM,
    PASS1_SYSTEM,
    pass0_user,
    pass1_user,
    pass2_system,
    pass2_user,
)

if TYPE_CHECKING:
    from middleware.llm.llm_client import LLMClient

logger = get_logger(__name__)


def _signal_sufficient(result: str) -> bool:
    """Check whether Pass 0 signal is strong enough to skip Pass 1."""
    if not result or len(result.strip()) < 100:
        return False
    if "\n" in result and (
        "##" in result
        or re.search(r"^[*-] ", result, re.MULTILINE)
        or re.search(r"^\s*\d+\. ", result, re.MULTILINE)
    ):
        return True
    return len(result.strip()) > 500


class SoulEvolver:
    """SOUL.md evolution engine — conservative: cross-session validation required."""

    def __init__(self, llm_client: "LLMClient", soul_manager: SoulManager | None = None) -> None:
        self._llm = llm_client
        self._soul = soul_manager or SoulManager()
        self._last_run: datetime | None = None
        self._recent_sessions: list[str] = []

    async def evolve_session(self, session_id: str) -> None:
        """Session-end trigger: record session ID, defer evolution for cross-session validation."""
        self._recent_sessions.append(session_id)
        if len(self._recent_sessions) > 10:
            self._recent_sessions = self._recent_sessions[-10:]

    async def evolve_timed(self) -> None:
        """Timed evolution: update SOUL.md after cross-session validation."""
        if len(self._recent_sessions) < 2:
            logger.debug("[SoulEvolver] not enough sessions for cross-validation, skipping")
            return

        from shared.chat import ChatManager

        sessions_to_analyze = self._recent_sessions[:5]
        transcripts = []
        for sid in sessions_to_analyze:
            history = await ChatManager.get_conversation_history(sid, limit=30)
            if len(history) >= 2:
                transcripts.append(self._format_transcript(history))

        if not transcripts:
            return

        combined = "\n\n====== SESSION ======\n\n".join(transcripts[:3])
        if len(combined) > INPUT_MAX_CHARS:
            combined = combined[:INPUT_MAX_CHARS]

        current_soul = self._soul.format_current()

        # Pass 0: Identity observation
        pass0 = await self._call_llm(PASS0_SYSTEM, pass0_user(combined))
        if not pass0:
            return

        # Pass 1: Cross-session validation
        pass1 = ""
        if not _signal_sufficient(pass0):
            pass1 = await self._call_llm(
                PASS1_SYSTEM,
                pass1_user(pass0, current_soul, len(sessions_to_analyze)),
            )
            if pass1 and pass1.strip().lower() in ("do not evolve", "no additions needed", "none", "n/a"):
                logger.info("[SoulEvolver] evolution signal insufficient, skipping")
                self._recent_sessions.clear()
                return

        analysis = pass0 + ("\n\n" + pass1 if pass1 else "")

        # Pass 2: Synthesis output
        updates = await self._call_llm_json(
            pass2_system(),
            pass2_user(analysis, current_soul),
        )
        if updates:
            changed = self._soul.merge_update(updates)
            if changed:
                logger.info("[SoulEvolver] SOUL.md evolved: {}", list(updates.keys()))
                self._last_run = datetime.now()
        self._recent_sessions.clear()

    async def seed_from_history(self, limit: int = 10) -> None:
        """Bootstrap _recent_sessions from database history (called at startup).

        Loads recent session IDs so that SOUL evolution can run without waiting
        for real session-end events to accumulate.
        """
        if self._recent_sessions:
            return
        from shared.chat import ChatManager

        try:
            sessions = await ChatManager.list_sessions(limit=limit)
            self._recent_sessions = [s.id for s in sessions if hasattr(s, "id")]
            if self._recent_sessions:
                logger.debug(
                    "[SoulEvolver] seeded {} sessions from history",
                    len(self._recent_sessions),
                )
        except Exception:
            logger.warning("[SoulEvolver] failed to seed sessions from history")

    def _format_transcript(self, history: list[dict]) -> str:
        lines = []
        for msg in history[-15:]:
            role = msg.get("role", "")
            content = msg.get("content", "")[:MSG_TRUNCATE_CHARS]
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            result = await self._llm.async_chat(messages)
            return (result.content if hasattr(result, "content") else str(result)).strip()
        except Exception as e:
            logger.warning("[SoulEvolver] LLM call failed: {}", e)
            return ""

    async def _call_llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            result = await self._llm.async_chat(messages)
            raw = (result.content if hasattr(result, "content") else str(result)).strip()
            code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if code_block_match:
                raw = code_block_match.group(1).strip()
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
            return {}
        except json.JSONDecodeError:
            logger.debug("[SoulEvolver] JSON parse failed for soul update")
            return {}
        except Exception as e:
            logger.warning("[SoulEvolver] LLM JSON call failed: {}", e)
            return {}
