"""Context memory — 长期记忆（MEMORY.md）+ 每日执行笔记（daily notes）。

TODO: memory 持久化功能暂未接入 ContextManager，后续扩展。
      当前 ContextManager 不依赖此模块。

NOTE: ContextMemory 内部引用的 ctx_cfg 字段（max_memory_prompt_chars、
      daily_notes_show_days 等）与当前 ContextConfig 不兼容。
      接入时需要补充对应的配置字段或使用独立的 MemoryConfig。

数据目录:
    {data_dir}/context/MEMORY.md              — 跨 session 稳定知识
    {data_dir}/context/{YYYY-MM-DD}/daily_note.md  — 当天执行细节
"""
from __future__ import annotations

from utils.logger import get_logger
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from middleware.llm.utils import chat_completions_async
from .scratchpad import Scratchpad

logger = get_logger(__name__)

_DEDUP_SYSTEM_PROMPT = (
    "You are a deduplication checker. Given existing memory content and a new entry, "
    "determine if the new entry is semantically redundant (already covered by existing content). "
    "Respond with ONLY 'YES' if redundant, or 'NO' if it contains new information."
)


class ContextMemory:
    """管理 MEMORY.md 和 daily notes 的读写。

    Attributes:
        memory_path: MEMORY.md 文件路径。
        daily_note_path: 今天的 daily_note.md 文件路径。
    """

    def __init__(self, ctx_dir: Path, date_dir: Path, ctx_cfg: Any) -> None:
        """初始化 memory 路径。

        Args:
            ctx_dir: context 根目录（{data_dir}/context/）。
            date_dir: 今天的日期目录（{data_dir}/context/YYYY-MM-DD/）。
            ctx_cfg: ContextConfig 实例。
        """
        self._ctx_dir = ctx_dir
        self._cfg = ctx_cfg
        self.memory_path = ctx_dir / "MEMORY.md"
        self.daily_note_path = date_dir / "daily_note.md"

    def get_context_section(self, scratchpad: Scratchpad) -> str:
        """生成供 system prompt 注入的上下文段落。

        包含: MEMORY.md + 最近 N 天 daily notes + scratchpad 引用。

        Args:
            scratchpad: 当前 session 的 Scratchpad 实例。

        Returns:
            格式化 markdown 字符串。空则返回 ""。
        """
        parts: list[str] = []

        # 1. MEMORY.md
        if self.memory_path.exists():
            content = self.memory_path.read_text(encoding="utf-8").strip()
            if content:
                max_chars = self._cfg.max_memory_prompt_chars
                if len(content) > max_chars:
                    content = content[:max_chars] + "\n...[memory truncated]"
                parts.append(f"## Long-term Memory\n{content}")

        # 2. Daily notes — 最近 N 天
        today = datetime.now().date()
        notes: list[str] = []
        for i in range(self._cfg.daily_notes_show_days):
            date = today - timedelta(days=i)
            path = self._ctx_dir / str(date) / "daily_note.md"
            if path.exists():
                size = path.stat().st_size
                if size > 0:
                    notes.append(f"- {date}: `{path}` ({size // 1024}KB)")
        if notes:
            parts.append(
                "## Recent Daily Notes (use filesystem skill to read)\n"
                + "\n".join(notes)
            )

        # 3. Scratchpad 引用
        ref = scratchpad.build_reference()
        if ref:
            parts.append(ref)

        return "\n\n".join(parts)

    async def write_to_memory(
        self, content: str, dedup: bool = True,
    ) -> bool:
        """向 MEMORY.md 追加稳定知识，支持 LLM 语义去重。

        Args:
            content: 要追加的文本。
            dedup: 是否启用语义去重（默认 True）。

        Returns:
            True 如果成功写入，False 如果被去重跳过。
        """
        if dedup:
            is_dup = await self._is_semantic_duplicate(content)
            if is_dup:
                logger.debug("Skipping semantically duplicate memory: {content}", content=content[:50])
                return False

        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, "a", encoding="utf-8") as f:
            f.write(f"\n{content}\n")
        return True

    def needs_compaction(self) -> bool:
        """检查 MEMORY.md 是否超过容量阈值，需要 LLM 归纳合并。

        Returns:
            True 如果需要压缩。
        """
        if not self.memory_path.exists():
            return False
        return self.memory_path.stat().st_size > self._cfg.max_memory_bytes

    def get_memory_content(self) -> str:
        """读取 MEMORY.md 全文。"""
        if not self.memory_path.exists():
            return ""
        return self.memory_path.read_text(encoding="utf-8")

    def replace_memory(self, new_content: str) -> None:
        """替换 MEMORY.md 全文（用于 LLM 归纳合并后的写回）。

        Args:
            new_content: 归纳后的新内容。
        """
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(new_content, encoding="utf-8")
        logger.info(
            "Memory compacted: {size} bytes", size=len(new_content.encode("utf-8"))
        )

    async def compact_memory(self) -> bool:
        """LLM 归纳合并 MEMORY.md 内容，减少文件体积。

        仅当 needs_compaction() 为 True 时才有效果。
        将全文发给 LLM 要求精炼归纳，然后用结果替换原文件。

        Returns:
            True 如果执行了压缩，False 如果不需要。
        """
        if not self.needs_compaction():
            return False

        existing = self.get_memory_content()
        if not existing.strip():
            return False

        try:
            compacted = await chat_completions_async(
                "You are a memory compactor. Given a collection of long-term memory entries, "
                "merge and deduplicate them into a concise, well-organized summary. "
                "Preserve all unique facts, preferences, and decisions. "
                "Remove redundancy. Use bullet points grouped by topic. "
                "Keep the result in the same language as the input.",
                [{"role": "user", "content": existing}],
            )
            if compacted.strip():
                self.replace_memory(compacted.strip())
                return True
        except Exception:
            logger.warning("Memory compaction via LLM failed", exc_info=True)

        return False

    def flush_session(self, session_id: str, summary: str) -> None:
        """Session 结束时写入 daily note。

        Args:
            session_id: 会话 ID。
            summary: 执行摘要（来自 session_ctx.to_summary()）。
        """
        self.daily_note_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.daily_note_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n### Session {session_id} ({datetime.now():%H:%M})\n"
                f"{summary}\n"
            )

    def cleanup_old_daily_notes(self) -> int:
        """清理超过 daily_notes_age_days 天的 daily notes 目录。

        Returns:
            删除的目录数量。
        """
        today = datetime.now().date()
        cutoff = today - timedelta(days=self._cfg.daily_notes_age_days)
        removed = 0

        if not self._ctx_dir.exists():
            return 0

        for child in self._ctx_dir.iterdir():
            if not child.is_dir():
                continue
            try:
                dir_date = datetime.strptime(child.name, "%Y-%m-%d").date()
                if dir_date < cutoff:
                    shutil.rmtree(child)
                    logger.debug("Cleaned up old daily notes: {name}", name=child.name)
                    removed += 1
            except ValueError:
                continue  # 非日期格式的目录跳过

        return removed

    # ── Internal helpers ──────────────────────────────────────────

    async def _is_semantic_duplicate(self, content: str) -> bool:
        """LLM 判断 content 是否与已有 MEMORY.md 内容语义重复。

        优化策略:
        1. 快速文本匹配: content 已是 existing 的子串 → 直接判重
        2. 截断 existing: 仅发送最后 max_memory_prompt_chars 给 LLM，控制成本
        3. 传递 memory_dedup_max_tokens 限制 LLM 输出

        Args:
            content: 待写入的文本。

        Returns:
            True 如果 LLM 判断为语义重复。
        """
        existing = self.get_memory_content()
        if not existing.strip():
            return False

        normalized_content = content.strip()
        if normalized_content in existing:
            logger.debug("Skipping exact-substring duplicate memory")
            return True

        max_chars = self._cfg.max_memory_prompt_chars
        if len(existing) > max_chars:
            existing = existing[-max_chars:]

        prompt = (
            f"## Existing Memory:\n{existing}\n\n"
            f"## New Entry:\n{content}\n\n"
            f"Is the new entry semantically redundant with existing memory?"
        )

        try:
            result = await chat_completions_async(
                _DEDUP_SYSTEM_PROMPT,
                [{"role": "user", "content": prompt}],
                max_tokens=self._cfg.memory_dedup_max_tokens,
            )
            answer = result.strip().upper()
            return answer.startswith("YES")
        except Exception:
            logger.warning(
                "LLM semantic dedup failed, allowing write", exc_info=True
            )
            return False
