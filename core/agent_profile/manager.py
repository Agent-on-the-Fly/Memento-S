# SPDX-License-Identifier: Apache-2.0
"""AgentProfileManager — Facade（统一入口）。

agent.py 只对接这一个 Manager，底层委托给 SoulManager 和 UserManager。

职责边界：
  Facade   — agent 热路径入口、profile building、进化触发
  SoulManager   (core/agent_profile/soul_manager.py) — SOUL.md I/O + TTL cache
  UserManager   (core/agent_profile/user_manager.py)  — USER.md I/O + TTL cache
"""

from __future__ import annotations

import asyncio
from typing import Any

from core.agent_profile.models import AgentProfile
from core.agent_profile.soul_manager import SoulManager
from core.agent_profile.user_manager import UserManager


# 模块级单例
_soul_mgr: SoulManager | None = None
_user_mgr: UserManager | None = None


def _get_soul_manager() -> SoulManager:
    global _soul_mgr
    if _soul_mgr is None:
        _soul_mgr = SoulManager()
    return _soul_mgr


def _get_user_manager() -> UserManager:
    global _user_mgr
    if _user_mgr is None:
        _user_mgr = UserManager()
    return _user_mgr


class AgentProfileManager:
    """Facade: agent 的唯一入口，内部委托给 SoulManager + UserManager。"""

    # ── SOUL.md ────────────────────────────────────────────────────────

    def load_soul(self) -> dict:
        """Load SOUL.md (from SoulManager, TTL cached)."""
        return _get_soul_manager().load()

    def merge_update_soul(self, updates: dict) -> bool:
        """Merge partial field updates into SOUL.md."""
        return _get_soul_manager().merge_update(updates)

    def format_soul(self) -> str:
        """Return current SOUL.md as formatted string."""
        return _get_soul_manager().format_current()

    # ── USER.md ────────────────────────────────────────────────────────

    def load_user_context(self) -> str:
        """Load USER.md with TTL cache. Returns sectioned context for LLM."""
        return _get_user_manager().load_context()

    def append_user_facts(self, facts_by_section: dict[str, list[str]]) -> bool:
        """Merge facts into USER.md by section. Returns True if new facts were added."""
        return _get_user_manager().append_facts(facts_by_section)

    # ── lifecycle ─────────────────────────────────────────────────────

    def ensure_files(self) -> None:
        """Ensure SOUL.md and USER.md exist with default templates."""
        _get_soul_manager().ensure_exists()
        _get_user_manager().ensure_exists()

    # ── profile building ───────────────────────────────────────────────

    async def build_profile(
        self,
        skill_gateway: Any = None,
        config: Any = None,
    ) -> AgentProfile:
        """Build AgentProfile from SOUL.md + USER.md + runtime context."""
        from core.memento_s.skill_dispatch import AGENT_TOOL_SCHEMAS

        soul_mgr = _get_soul_manager()
        soul = soul_mgr.load()
        user_context = self.load_user_context()

        capabilities: list[str] = []
        available_tools = [
            item.get("function", {}).get("name", "")
            for item in AGENT_TOOL_SCHEMAS
            if item.get("type") == "function"
        ]
        available_tools = [t for t in available_tools if t]

        if skill_gateway is not None:
            try:
                manifests = await skill_gateway.discover()
                for m in manifests:
                    name = m.name.strip()
                    desc = (m.description or "").strip()
                    if desc:
                        capabilities.append(f"{name}: {desc[:100]}")
                    else:
                        capabilities.append(name)
            except Exception:
                pass

        model_info = ""
        if config is not None:
            try:
                model_info = config.llm.model or ""
            except Exception:
                pass

        return AgentProfile(
            name=soul["name"],
            role=soul["role"],
            core_truths=soul["core_truths"],
            boundaries=soul["boundaries"],
            vibe=soul["vibe"],
            tone_examples=soul["tone_examples"],
            capabilities=capabilities,
            model_info=model_info,
            available_tools=available_tools,
            user_context=user_context,
        )

    # ── evolution trigger ────────────────────────────────────────────

    def on_session_end(self, session_id: str) -> None:
        """Forward session-end to the evolver daemon (non-blocking)."""
        try:
            from daemon.agent_profile import AgentProfileEvolverDaemon
            evolver = AgentProfileEvolverDaemon.get_evolver()
            if evolver is not None:
                evolver.on_session_end(session_id)
        except Exception:
            pass


apm = AgentProfileManager()
