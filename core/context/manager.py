"""ContextManager — Agent 唯一的上下文管理接口。

Public API:
  Prompt & History:
    load_history()            — 从 DB 加载历史，token-aware 截止
    assemble_messages()       — 组装完整 message list（system + history + user）
    assemble_system_prompt()  — 构造 system prompt
    build_history_summary()   — 简短历史摘要（用于 intent 识别）
    invalidate_skills_cache() — 清除技能摘要缓存

  Context Runtime:
    init_budget()             — 设置 token 预算（input_budget）+ 派生 compress/compact 阈值
    append()                  — 追加消息 + 自动 compress / compact（compact 时归档到 scratchpad）
    persist_tool_result()     — 返回内联 tool message（不写盘、不截断）
    write_to_scratchpad()     — 手动写入 scratchpad
"""
from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

from core.prompts.templates import (
    AGENT_IDENTITY_OPENING,
    BUILTIN_TOOLS_SECTION,
    EXECUTION_CONSTRAINTS_SECTION,
    IDENTITY_SECTION,
    IMPORTANT_DIRECT_REPLY,
    PROTOCOL_AND_FORMAT,
    SKILLS_SECTION,
    WORKSPACE_PATHS_NOTE,
)
from core.skill.gateway import SkillGateway
from core.utils import format_user_content
from middleware.config import g_config
from utils.logger import get_logger
from utils.token_utils import count_tokens, count_tokens_messages

from .compaction import compact_messages, compress_message
from .schemas import ContextConfig
from .scratchpad import Scratchpad

logger = get_logger(__name__)

_HISTORY_LOAD_LIMIT = 80

HistoryLoader = Callable[[str, int], Coroutine[Any, Any, list[dict[str, Any]]]]


class ContextManager:
    """Session 级别的上下文管理器。

    生命周期: agent.reply_stream() 创建 → assemble_messages() → execution 使用。
    """

    def __init__(
        self,
        session_id: str,
        config: ContextConfig,
        *,
        skill_gateway: SkillGateway | None = None,
        history_loader: HistoryLoader | None = None,
    ) -> None:
        self.session_id = session_id
        self._cfg = config

        self._skill_gateway = skill_gateway
        self._history_loader = history_loader
        self._skills_summary_cache: str | None = None

        self.workspace = g_config.paths.workspace_dir

        # token 状态（init_budget 时设置）
        self._total_tokens: int = 0
        self._context_max_tokens: int = 0
        self._compress_threshold: int = 0
        self._compact_trigger: int = 0
        self._summary_tokens: int = 0

        # scratchpad
        ctx_dir: Path = g_config.paths.context_dir
        today_str = datetime.now().strftime("%Y-%m-%d")
        date_dir = ctx_dir / today_str
        date_dir.mkdir(parents=True, exist_ok=True)

        self._scratchpad = Scratchpad(session_id, date_dir)

    # ═══════════════════════════════════════════════════════════════
    # Token 状态 & append
    # ═══════════════════════════════════════════════════════════════

    def init_budget(self, context_max_tokens: int) -> None:
        """设置 token 预算，所有阈值直接从 input_budget * ratio 派生。"""
        self._context_max_tokens = context_max_tokens
        self._compress_threshold = max(
            int(context_max_tokens * self._cfg.compress_threshold_ratio), 512,
        )
        self._compact_trigger = max(
            int(context_max_tokens * self._cfg.compaction_trigger_ratio), 1024,
        )
        self._summary_tokens = max(
            int(context_max_tokens * self._cfg.summary_ratio), 200,
        )
        logger.info(
            "Budget: context_max={}, compact_trigger={}, "
            "compress_threshold={}, summary_tokens={}",
            context_max_tokens, self._compact_trigger,
            self._compress_threshold, self._summary_tokens,
        )

    def sync_tokens(self, messages: list[dict[str, Any]]) -> None:
        """用完整消息列表同步 _total_tokens（首次/重置时调用）。"""
        self._total_tokens = count_tokens_messages(messages)
        logger.debug("Token state synced: {}", self._total_tokens)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    async def append(
        self,
        messages: list[dict[str, Any]],
        new_msgs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """追加消息，自动执行 compress 和 compact。

        流程:
        1. compress: 单条新消息超过 _compress_threshold → LLM 摘要
        2. 追加到消息列表，更新 _total_tokens
        3. compact: 总 token 超过 _compact_trigger → 归档 + 全量合并
        """
        compressed = []
        for msg in new_msgs:
            if self._compact_trigger > 0:
                compressed.append(await compress_message(
                    msg,
                    max_msg_tokens=self._compress_threshold,
                    summary_tokens=self._summary_tokens,
                ))
            else:
                compressed.append(msg)

        result = list(messages) + compressed
        added_tokens = count_tokens_messages(compressed)
        self._total_tokens += added_tokens

        if self._compact_trigger > 0 and self._total_tokens > self._compact_trigger:
            logger.info("Compact trigger: {} > {} (max={})",
                        self._total_tokens, self._compact_trigger,
                        self._context_max_tokens)
            self._archive_to_scratchpad(result)
            result, self._total_tokens = await compact_messages(
                result,
                summary_tokens=self._summary_tokens,
                scratchpad_path=str(self._scratchpad.path),
            )

        return result

    # ═══════════════════════════════════════════════════════════════
    # History loading
    # ═══════════════════════════════════════════════════════════════

    async def load_history(self) -> list[dict[str, Any]]:
        """从 DB 加载历史，token-aware 截止。

        倒序遍历 DB 记录，累加 token 到 input_budget 即停。
        应在 reply_stream 入口调用一次，结果传给 intent + assemble。
        """
        if not self._history_loader:
            return []

        raw = await self._history_loader(self.session_id, _HISTORY_LOAD_LIMIT)
        if not raw:
            return []

        budget = g_config.llm.current_profile.input_budget
        if budget <= 0:
            return raw

        selected: list[dict[str, Any]] = []
        accumulated = 0
        for msg in reversed(raw):
            t = count_tokens(str(msg.get("content", "")))
            if accumulated + t > budget:
                break
            selected.append(msg)
            accumulated += t

        selected.reverse()
        if len(selected) < len(raw):
            logger.info(
                "Token-aware load: {} -> {} msgs ({}/{} tokens)",
                len(raw), len(selected), accumulated, budget,
            )
        return selected

    # ═══════════════════════════════════════════════════════════════
    # Prompt & History
    # ═══════════════════════════════════════════════════════════════

    def invalidate_skills_cache(self) -> None:
        """清除技能摘要缓存（安装新技能后调用）。"""
        self._skills_summary_cache = None

    async def force_compact_now(self) -> tuple[int, int, str]:
        """立即压缩历史上下文。

        Returns:
            (old_tokens, new_tokens, summary_preview)
        """
        history = await self.load_history()
        if not history or len(history) <= 2:
            return 0, 0, ""

        old_tokens = count_tokens_messages(history)
        target = max(self._summary_tokens, 200) if self._summary_tokens else 2000
        compacted, new_tokens = await compact_messages(history, summary_tokens=target)

        preview = ""
        for msg in compacted:
            if "[历史摘要" in (msg.get("content") or ""):
                preview = msg["content"]
                break

        self._total_tokens = new_tokens
        logger.info("Force compact: {} -> {} tokens", old_tokens, new_tokens)
        return old_tokens, new_tokens, preview

    def build_history_summary(
        self,
        history: list[dict[str, Any]] | None,
        max_rounds: int = 3,
        max_tokens: int = 800,
    ) -> str:
        """构建简短历史摘要（用于 intent 识别）。"""
        if not history:
            return "(no prior context)"

        meaningful = [
            m for m in history
            if m.get("role") in ("user", "assistant")
            and str(m.get("content", "")).strip()
        ]
        if not meaningful:
            return "(no prior context)"

        candidates = meaningful[-(max_rounds * 2):]
        selected: list[str] = []
        remaining_tokens = max_tokens

        for m in reversed(candidates):
            content = str(m.get("content", "")).strip()
            tokens = count_tokens(content)
            if tokens <= remaining_tokens:
                selected.append(f"{m['role']}: {content}")
                remaining_tokens -= tokens
            else:
                char_budget = remaining_tokens * 3
                if char_budget > 50:
                    selected.append(f"{m['role']}: {content[:char_budget]}...")
                break

        selected.reverse()
        return "\n".join(selected) if selected else "(no prior context)"

    async def assemble_system_prompt(
        self,
        *,
        mode: str = "agentic",
        intent_shifted: bool = False,
        matched_skills_context: str = "",
        agent_profile: Any = None,
        session_context: Any = None,
    ) -> str:
        """构造完整 system prompt。"""
        if agent_profile is not None and hasattr(agent_profile, "to_prompt_section"):
            identity = self._identity_section()
            identity += "\n\n" + agent_profile.to_prompt_section()
            parts = [identity]
        else:
            parts = [self._identity_section()]

        behavior = [
            "## runtime_behavior",
            "- Prefer direct concise reply for simple chit-chat; avoid unnecessary tool calls.",
            "- For task-oriented requests, use tools/skills step-by-step.",
        ]
        if intent_shifted:
            behavior.append(
                "- Current user intent has shifted from previous turns; prioritize latest user message."
            )
        if mode in ("direct", "interrupt"):
            behavior.append(
                "- This turn is classified as direct. Answer directly unless the user explicitly asks for tools."
            )

        parts.append("\n".join(behavior))
        parts.append(PROTOCOL_AND_FORMAT)
        parts.append(BUILTIN_TOOLS_SECTION)

        skills_summary = await self._build_skills_summary()
        if skills_summary and mode not in ("direct", "interrupt"):
            skills_section = SKILLS_SECTION.format(skills_summary=skills_summary)
            if matched_skills_context:
                skills_section += "\n\n" + matched_skills_context
            parts.append(skills_section)
        elif matched_skills_context and mode not in ("direct", "interrupt"):
            parts.append(matched_skills_context)

        ctx_section = self._get_context_section()
        if ctx_section:
            parts.append(ctx_section)

        if session_context is not None and hasattr(session_context, "to_prompt_section"):
            session_section = session_context.to_prompt_section()
            if session_section:
                parts.append(session_section)

        return "\n\n---\n\n".join(parts)

    async def assemble_messages(
        self,
        history: list[dict[str, Any]] | None,
        current_message: str,
        media: list[str] | list[Path] | None = None,
        matched_skills_context: str = "",
        agent_profile: Any = None,
        session_context: Any = None,
        mode: str = "agentic",
        intent_shifted: bool = False,
    ) -> list[dict[str, Any]]:
        """组装完整 message list 并初始化 token 状态。

        history 应由调用方通过 load_history() 预先加载并传入。
        """
        if history is None:
            history = []

        selected_history = self._select_history_for_intent(
            history, mode=mode, intent_shifted=intent_shifted,
        )
        logger.info("History: raw={}, selected={} (mode={}, shifted={})",
                     len(history), len(selected_history), mode, intent_shifted)

        system_prompt = await self.assemble_system_prompt(
            mode=mode, intent_shifted=intent_shifted,
            matched_skills_context=matched_skills_context,
            agent_profile=agent_profile, session_context=session_context,
        )
        user_content = await format_user_content(current_message, media)

        system_tokens = count_tokens(system_prompt)
        user_tokens = count_tokens(
            user_content if isinstance(user_content, str) else current_message
        )
        input_budget = g_config.llm.current_profile.input_budget
        history_budget = input_budget - system_tokens - user_tokens

        logger.info("Budget: input_budget={}, sys={}, user={}, history_budget={}",
                     input_budget, system_tokens, user_tokens, history_budget)

        if selected_history:
            history_tokens = count_tokens_messages(selected_history)
            if history_budget > 0 and history_tokens > history_budget:
                summary_target = max(
                    int(input_budget * self._cfg.summary_ratio), 200,
                )
                logger.info("History ({} tokens) exceeds budget ({}), compacting to ~{} tokens",
                            history_tokens, history_budget, summary_target)
                selected_history, _ = await compact_messages(
                    selected_history, summary_tokens=summary_target,
                )
            elif history_budget <= 0:
                selected_history = selected_history[-4:]
                logger.warning("Budget exhausted ({}), keeping last {} msgs",
                               history_budget, len(selected_history))

        if selected_history:
            last = selected_history[-1]
            current_str = user_content if isinstance(user_content, str) else str(user_content)
            if last.get("role") == "user" and last.get("content") == current_str:
                selected_history = selected_history[:-1]

        result = [
            {"role": "system", "content": system_prompt},
            *selected_history,
            {"role": "user", "content": user_content},
        ]

        self.sync_tokens(result)
        self.init_budget(input_budget)

        return result

    # ═══════════════════════════════════════════════════════════════
    # Context Runtime（scratchpad）
    # ═══════════════════════════════════════════════════════════════

    @property
    def scratchpad_path(self) -> Path:
        return self._scratchpad.path

    def persist_tool_result(
        self, tool_call_id: str, tool_name: str, result: str,
    ) -> dict[str, Any]:
        """返回内联 tool message（不写盘、不截断，compress/compact 统一管控）。"""
        return self._scratchpad.persist_tool_result(tool_call_id, tool_name, result)

    def write_to_scratchpad(self, section: str, content: str) -> str:
        """手动写入 scratchpad，返回引用标记。"""
        return self._scratchpad.write(section, content)

    def _archive_to_scratchpad(self, messages: list[dict[str, Any]]) -> None:
        """compact 触发时，将待压缩的消息批量归档到 scratchpad。"""
        system_start = 1 if messages and messages[0].get("role") == "system" else 0
        rest = messages[system_start:]
        if rest:
            self._scratchpad.archive_messages(rest)
            logger.info("Archived {} messages to scratchpad before compact", len(rest))

    def _get_context_section(self) -> str:
        """生成 system prompt 注入段: scratchpad ref（仅 compact 后有内容时注入）。"""
        ref = self._scratchpad.build_reference()
        return ref if ref else ""

    # ═══════════════════════════════════════════════════════════════
    # Internal
    # ═══════════════════════════════════════════════════════════════

    def _identity_section(self) -> str:
        now_dt = datetime.now()
        now = now_dt.strftime("%Y-%m-%d %H:%M (%A)")
        current_year = str(now_dt.year)
        workspace_path = str(self.workspace)
        skills_path = str(g_config.paths.skills_dir)
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        environment_section = (
            f"[Environment]\n"
            f"- Workspace: {workspace_path}\n"
            f"- Skills: {skills_path}\n"
            f"[/Environment]"
        )
        workspace_paths_note = WORKSPACE_PATHS_NOTE.format(workspace_path=workspace_path)

        return "\n\n".join([
            IDENTITY_SECTION.format(
                identity_opening=AGENT_IDENTITY_OPENING,
                current_time=now,
                current_year=current_year,
                runtime=runtime,
                workspace_paths_note=workspace_paths_note,
                execution_constraints=EXECUTION_CONSTRAINTS_SECTION,
                important_direct_reply=IMPORTANT_DIRECT_REPLY,
            ),
            environment_section,
        ])

    async def _build_skills_summary(self) -> str:
        if self._skills_summary_cache is not None:
            return self._skills_summary_cache
        if not self._skill_gateway:
            return ""
        manifests = self._skill_gateway.discover()
        if not manifests:
            return ""
        lines = []
        for m in manifests:
            name = m.name.strip()
            desc = (m.description or "").strip()
            if desc and len(desc) > 400:
                desc = desc[:397] + "..."
            lines.append(f"- **{name}**: {desc}")
        self._skills_summary_cache = "\n".join(sorted(lines))
        return self._skills_summary_cache

    @staticmethod
    def _select_history_for_intent(
        history: list[dict[str, Any]], *, mode: str, intent_shifted: bool,
    ) -> list[dict[str, Any]]:
        if not history:
            return []
        if mode in ("direct", "interrupt"):
            return history[-4:]
        if intent_shifted:
            candidate = history[-4:]
            return [m for m in candidate if m.get("role") in {"user", "assistant"}]
        return history

