# SPDX-License-Identifier: Apache-2.0
"""SessionMemory — L1 Session Memory 实现。

迁移自 core/context/session_memory.py。
无 core/ 依赖，使用 shared_prompts 中的模板和构建函数。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from utils.logger import get_logger
from utils.token_utils import count_tokens

logger = get_logger(__name__)

SESSION_MEMORY_DIR = ""
SUMMARY_FILE = "summary.md"
META_FILE = "meta.json"
MAX_SECTION_TOKENS = 2000
MAX_TOTAL_TOKENS = 12000
MAX_WORKLOG_LINES = 50

# LLM client type alias
LLMClient = Callable[..., Awaitable[str]]


def _default_llm_client(
    messages: list[dict[str, Any]],
    *,
    system: str = "",
    max_tokens: int = 3000,
) -> Awaitable[str]:
    from middleware.llm.llm_client import chat_completions_async

    return chat_completions_async(
        system=system,
        messages=messages,
        max_tokens=max_tokens,
    )


class SessionMemory:
    """L1 Session Memory — CC-style structured summary.md.

    Lifecycle: setup() → append_worklog_entry() / llm_update() → to_prompt_section()

    迁移自 core/context/session_memory.py，无 core/ 依赖。
    """

    def __init__(
        self,
        session_dir: Path,
        model: str = "",
        llm_update_interval: int = 5,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._dir = session_dir / SESSION_MEMORY_DIR
        self._path = self._dir / SUMMARY_FILE
        self._meta_path = self._dir / META_FILE
        self._model = model
        self._llm_update_interval = llm_update_interval
        self._template: str = ""
        self._react_since_llm_update: int = 0
        self._last_summarized_seq: int = 0
        self._cached_content: str | None = None
        self._llm_client = llm_client or _default_llm_client

    # ---- 生命周期 ----

    async def setup(self) -> None:
        """创建目录 + 初始化 template 文件 + 加载 meta。"""
        from infra.context.providers.shared_prompts import DEFAULT_SESSION_MEMORY_TEMPLATE

        self._dir.mkdir(parents=True, exist_ok=True)
        self._template = DEFAULT_SESSION_MEMORY_TEMPLATE.strip()
        if not self._path.exists():
            self._path.write_text(self._template, encoding="utf-8")
            self._cached_content = self._template
            logger.info("[SessionMemory] created summary.md at {}", self._path)
        else:
            logger.info("[SessionMemory] summary.md already exists at {}", self._path)
        self._load_meta()
        self._save_meta()

    def _load_meta(self) -> None:
        """从 .meta.json 恢复 last_summarized_seq。"""
        if self._meta_path.exists():
            try:
                data = json.loads(self._meta_path.read_text(encoding="utf-8"))
                self._last_summarized_seq = data.get("last_summarized_seq", 0)
            except (json.JSONDecodeError, OSError):
                logger.opt(exception=True).warning("Failed to load session memory meta")

    def _save_meta(self) -> None:
        """持久化 meta 到磁盘。"""
        try:
            self._meta_path.write_text(
                json.dumps({"last_summarized_seq": self._last_summarized_seq}),
                encoding="utf-8",
            )
        except OSError:
            logger.opt(exception=True).warning("Failed to save session memory meta")

    def get_content(self) -> str:
        """读取 summary.md 全文 (内存缓存，避免重复磁盘读)。"""
        if self._cached_content is not None:
            return self._cached_content
        if not self._path.exists():
            self._cached_content = self._template
            return self._template
        try:
            self._cached_content = self._path.read_text(encoding="utf-8")
            return self._cached_content
        except OSError:
            self._cached_content = self._template
            return self._template

    def _invalidate_cache(self) -> None:
        """Invalidate in-memory cache after disk write."""
        self._cached_content = None

    def is_empty(self) -> bool:
        """对比 template 判断是否有实际内容。"""
        content = self.get_content().strip()
        return content == self._template.strip() or not content

    # ---- 自适应更新 ----

    async def llm_update(
        self, recent_messages: list[dict[str, Any]], step_id: str
    ) -> None:
        """LLM 全文替换 + 校验兜底。"""
        from infra.context.providers.shared_prompts import (
            build_session_memory_update_prompt,
        )

        current = self.get_content()
        new_msgs = [
            m for m in recent_messages
            if m.get("_seq", 0) > self._last_summarized_seq
        ]
        msgs_text = _messages_to_text_brief(new_msgs)

        if not msgs_text.strip():
            self._react_since_llm_update = 0
            return

        prompt = build_session_memory_update_prompt(current, msgs_text)
        updated_ok = False
        try:
            new_content = await self._llm_client(
                [{"role": "user", "content": prompt}],
                system="You are a session notes updater. Output the complete updated notes file.",
                max_tokens=3000,
            )
            if _validate_summary(new_content, self._template, model=self._model):
                cleaned = new_content.strip()
                self._path.write_text(cleaned, encoding="utf-8")
                self._cached_content = cleaned
                updated_ok = True
                logger.info("SM LLM update succeeded for step {}", step_id)
            else:
                logger.warning("SM LLM update failed validation, keeping old content")
        except Exception:
            logger.opt(exception=True).warning("SM LLM update failed")

        self._react_since_llm_update = 0
        if updated_ok:
            self._update_last_summarized_seq(recent_messages)
            self._save_meta()

    def append_worklog_entry(self, entry: str) -> None:
        """每个 react iteration 后规则追加 Worklog 行 (零 LLM)。"""
        content = self.get_content()
        updated = _append_to_section(
            content, "# Worklog", f"- {entry}",
            max_lines=MAX_WORKLOG_LINES,
        )
        try:
            self._path.write_text(updated, encoding="utf-8")
            self._cached_content = updated
        except OSError:
            logger.opt(exception=True).warning("Failed to write worklog entry")
        self._react_since_llm_update += 1

    def should_llm_update(self) -> bool:
        """判断是否应触发 LLM 更新 (N react 阈值)。"""
        return self._react_since_llm_update >= self._llm_update_interval

    # ---- SM compact 支持 (零 LLM) ----

    def truncate_for_compact(self) -> tuple[str, bool]:
        """CC-style 按段截断。"""
        content = self.get_content()
        if not content or self.is_empty():
            return "", False

        max_chars = MAX_SECTION_TOKENS * 4
        if len(content) <= max_chars:
            return content, False

        lines = content.split("\n")
        kept: list[str] = []
        char_count = 0

        for line in lines:
            if char_count + len(line) + 1 > max_chars:
                break
            kept.append(line)
            char_count += len(line) + 1

        kept.append("\n[... session memory truncated for length ...]")
        return "\n".join(kept), True

    @property
    def last_summarized_seq(self) -> int:
        return self._last_summarized_seq

    # ---- System prompt 注入 ----

    def to_prompt_section(self) -> str:
        """生成注入 system prompt 的段落 (按段截断防溢出)。"""
        if self.is_empty():
            return ""
        truncated, _ = self.truncate_for_compact()
        if not truncated:
            return ""
        return f"## Session Memory\n{truncated}"

    # ---- Recall 支持 ----

    def recall(self, query: str) -> str:
        """关键词搜索 summary.md 各段落。"""
        content = self.get_content()
        if not content or self.is_empty():
            return f"No matches found for query: {query}"

        query_lower = query.lower()
        sections = content.split("\n# ")
        matches: list[str] = []

        for section in sections:
            if query_lower in section.lower():
                matches.append(section.strip()[:500])

        if not matches:
            return f"No matches found for query: {query}"

        return "\n\n---\n\n".join(matches)

    # ---- 内部 ----

    def _update_last_summarized_seq(self, messages: list[dict[str, Any]]) -> None:
        """Update last_summarized_seq from messages."""
        if not messages:
            return
        max_seq = 0
        for msg in messages:
            seq = msg.get("_seq", 0)
            if seq > max_seq:
                max_seq = seq
        if max_seq > self._last_summarized_seq:
            self._last_summarized_seq = max_seq

    def save(self) -> None:
        """No-op: summary.md is written on each update. Meta saved separately."""
        self._save_meta()

    def load(self) -> None:
        """No-op for backward compatibility: setup() handles loading."""
        pass

    def has_digests(self) -> bool:
        """Backward compat: True if summary has content."""
        return not self.is_empty()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _validate_summary(content: str, template: str, model: str = "") -> bool:
    """校验 LLM 输出的 summary 是否合法。"""
    if not content or not content.strip():
        return False

    required_headers = [
        line.strip() for line in template.split("\n")
        if line.strip().startswith("# ")
    ]
    for header in required_headers:
        if header not in content:
            return False

    if count_tokens(content, model=model) > MAX_TOTAL_TOKENS:
        return False

    return True


def _append_to_section(
    content: str, section_header: str, entry: str, *, max_lines: int = 50
) -> str:
    """在指定 section 末尾追加一行，超 max_lines 淘汰最旧行。"""
    lines = content.split("\n")
    section_start = -1
    section_end = len(lines)

    for i, line in enumerate(lines):
        if line.strip() == section_header:
            section_start = i
        elif section_start >= 0 and line.startswith("# ") and i > section_start:
            section_end = i
            break

    if section_start < 0:
        return content + f"\n{section_header}\n{entry}\n"

    desc_end = section_start + 1
    while desc_end < section_end:
        if lines[desc_end].startswith("_") and lines[desc_end].endswith("_"):
            desc_end += 1
        else:
            break

    section_lines = [
        l for l in lines[desc_end:section_end]
        if l.strip()
    ]
    section_lines.append(entry)

    if len(section_lines) > max_lines:
        section_lines = section_lines[-max_lines:]

    result = (
        lines[:desc_end]
        + section_lines
        + ([""] if section_end < len(lines) else [])
        + lines[section_end:]
    )
    return "\n".join(result)


def _messages_to_text_brief(messages: list[dict[str, Any]]) -> str:
    """Lightweight message serialization for SM update prompts."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)

        if not content and not msg.get("tool_calls"):
            continue

        if role == "tool":
            name = msg.get("name", "tool")
            if len(content) > 300:
                content = content[:300] + "..."
            parts.append(f"[TOOL:{name}] {content}")
        elif role == "assistant":
            tc = msg.get("tool_calls")
            if tc:
                calls = []
                for t in tc:
                    func = t.get("function", {})
                    calls.append(func.get("name", "?"))
                parts.append(f"[ASSISTANT calls: {', '.join(calls)}]")
            if content:
                parts.append(f"[ASSISTANT] {content[:500]}")
        else:
            parts.append(f"[{role.upper()}] {content[:500]}")
    return "\n".join(parts)
