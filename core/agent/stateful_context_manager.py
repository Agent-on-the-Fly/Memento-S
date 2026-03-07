
from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from core.skills import SkillManager
from .prompts.templates import (
    AGENT_IDENTITY_OPENING,
    BUILTIN_TOOLS_SECTION,
    EXECUTION_CONSTRAINTS_SECTION,
    IDENTITY_SECTION,
    IMPORTANT_DIRECT_REPLY,
    PROTOCOL_AND_FORMAT,
    SKILLS_SECTION,
    WORKSPACE_PATHS_NOTE,
)
from .session_manager import SessionManager, _approx_tokens_from_content
from .utils import format_user_content
from ..config import g_settings
from ..config.logging import get_logger

logger = get_logger(__name__)


class StatefulContextManager:

    def __init__(
        self,
        workspace: Path,
        skill_manager: SkillManager,
        session_manager: SessionManager | None = None,
    ) -> None:
        self.workspace = workspace or g_settings.workspace_path
        self.skill_manager = skill_manager
        self.session_manager = session_manager if session_manager is not None else SessionManager(self.workspace)

    def assemble_system_prompt(self, skill_names: list[str] | None = None) -> str:
        parts = []
        parts.append(self._identity_section())
        parts.append(PROTOCOL_AND_FORMAT)
        parts.append(BUILTIN_TOOLS_SECTION)

        skills_summary = self.skill_manager.build_skills_summary()
        if skills_summary:
            parts.append(SKILLS_SECTION.format(skills_summary=skills_summary))

        return "\n\n---\n\n".join(parts)

    def _identity_section(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        workspace_path = str(g_settings.workspace_path)
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        workspace_paths_note = WORKSPACE_PATHS_NOTE.format(workspace_path=workspace_path)
        return IDENTITY_SECTION.format(
            identity_opening=AGENT_IDENTITY_OPENING,
            current_time=now,
            runtime=runtime,
            workspace_paths_note=workspace_paths_note,
            execution_constraints=EXECUTION_CONSTRAINTS_SECTION.format(workspace_path=workspace_path),
            important_direct_reply=IMPORTANT_DIRECT_REPLY,
        )

    async def assemble_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | list[Path] | None = None,
    ) -> list[dict[str, Any]]:
        system_prompt = self.assemble_system_prompt(skill_names)
        user_content = await format_user_content(current_message, media)

        system_tokens = _approx_tokens_from_content(system_prompt)
        user_tokens = _approx_tokens_from_content(
            user_content if isinstance(user_content, str) else current_message
        )
        reply_reserve = 4096
        budget = g_settings.context_max_tokens - system_tokens - user_tokens - reply_reserve

        if budget > 0 and history:
            history = self._truncate_history(history, budget)

        return [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _truncate_history(
        history: list[dict[str, Any]],
        budget_tokens: int,
    ) -> list[dict[str, Any]]:
        total = sum(_approx_tokens_from_content(m.get("content")) for m in history)
        if total <= budget_tokens:
            return history

        kept_tokens = 0
        cut_index = len(history)
        for i in range(len(history) - 1, -1, -1):
            msg_tokens = _approx_tokens_from_content(history[i].get("content"))
            if kept_tokens + msg_tokens > budget_tokens:
                cut_index = i + 1
                break
            kept_tokens += msg_tokens
        else:
            return history

        while cut_index < len(history) and history[cut_index].get("role") == "tool":
            cut_index += 1

        truncated = history[cut_index:]
        if len(truncated) < len(history):
            logger.info(
                "History truncated: %d → %d messages (budget=%d tokens)",
                len(history), len(truncated), budget_tokens,
            )
        return truncated
