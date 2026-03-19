"""Scratchpad — session 级别的 append-only 持久化文件。

Tool result 处理策略:
  - persist_tool_result: 全部内联返回，不做截断决策
  - 单条过长由 compress_message (token 阈值) 统一管控
  - compact 触发时: 将待压缩消息批量归档到 scratchpad

文件位置: {data_dir}/context/{YYYY-MM-DD}/scratchpad_{session_id}.md
"""
from __future__ import annotations

import json
from utils.logger import get_logger
from datetime import datetime
from pathlib import Path
from typing import Any

logger = get_logger(__name__)


def _format_skill_payload(data: dict) -> str:
    """Skill 执行结果 → markdown。"""
    skill = data.get("skill_name", "unknown")
    summary = data.get("summary", "")
    ok = data.get("ok")
    status = "OK" if ok else ("FAIL" if ok is False else "")

    parts = [f"**{skill}** {status}: {summary}" if summary else f"**{skill}** {status}"]

    output = data.get("output")
    if output is not None:
        parts.append(str(output))

    diag = data.get("diagnostics")
    if diag:
        parts.append(f"diagnostics: {json.dumps(diag, ensure_ascii=False)}")

    return "\n\n".join(parts)


def _format_batch_results(results: list) -> str:
    """批量 tool results → markdown。"""
    parts: list[str] = []
    for r in results:
        tool = r.get("tool", "unknown")
        args = r.get("args", {})
        label = args.get("path") or args.get("command", "") or args.get("query", "")
        parts.append(f"### {tool}: {label}")
        if "error" in r:
            parts.append(f"**ERROR**: {r['error']}")
        else:
            parts.append(str(r.get("result", "")))
    return "\n\n".join(parts)


class Scratchpad:
    """单个 session 的 scratchpad 文件管理。

    Attributes:
        path: scratchpad 文件的绝对路径。
    """

    _MIN_REF_BYTES = 100

    def __init__(self, session_id: str, date_dir: Path) -> None:
        """初始化 scratchpad 文件。

        Args:
            session_id: 当前会话 ID。
            date_dir: 日期目录路径（如 {workspace}/context/2026-03-17/）。
        """
        self.path = date_dir / f"scratchpad_{session_id}.md"
        self._section_count: int = 0

        try:
            if not self.path.exists():
                self.path.write_text(
                    f"# Session Scratchpad\n"
                    f"> session_id: {session_id}\n"
                    f"> created: {datetime.now():%Y-%m-%d %H:%M}\n\n",
                    encoding="utf-8",
                )
            else:
                existing = self.path.read_text(encoding="utf-8")
                self._section_count = existing.count("\n## [")
        except OSError:
            logger.warning("Failed to initialize scratchpad: {}", self.path, exc_info=True)

    def write(self, section: str, content: str) -> str:
        """向 scratchpad 追加一个带锚点的段落。

        Args:
            section: 段落标题（如 "Tool: search_file"）。
            content: 段落正文。

        Returns:
            引用标记，如 "[详见 scratchpad#section-3]"。
            I/O 失败时返回 "[scratchpad write failed]"。
        """
        self._section_count += 1
        anchor = f"section-{self._section_count}"
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"\n## [{anchor}] {section}\n")
                f.write(content)
                f.write("\n")
        except OSError:
            logger.warning("Failed to write to scratchpad: {}", self.path, exc_info=True)
            return "[scratchpad write failed]"
        return f"[详见 scratchpad#{anchor}]"

    def persist_tool_result(
        self, tool_call_id: str, tool_name: str, result: str
    ) -> dict[str, Any]:
        """返回内联 tool message（不写盘、不截断）。

        截断决策交给 compress_message / compact_messages 统一管控。
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }

    def archive_messages(self, messages: list[dict[str, Any]]) -> None:
        """将待压缩的消息批量归档到 scratchpad（compact 触发时调用）。

        跳过 system 消息。
        """
        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            if not content:
                continue
            formatted = self._format_for_scratchpad(content) if role == "tool" else content
            self.write(f"{role}", formatted)

    @staticmethod
    def _format_for_scratchpad(result: str) -> str:
        """将 tool result 格式化为易读 markdown。非 JSON 原样返回。

        支持两种 payload 结构:
        1. Skill payload: {"ok", "summary", "skill_name", "output", ...}
        2. 批量 results: {"results": [{"tool", "args", "result"}, ...]}
        """
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result

        if not isinstance(parsed, dict):
            return result

        # Skill payload: {"ok", "summary", "skill_name", "output", ...}
        if "skill_name" in parsed or "output" in parsed:
            return _format_skill_payload(parsed)

        # 批量 results: {"results": [...]}
        results = parsed.get("results")
        if isinstance(results, list) and results:
            return _format_batch_results(results)

        return result

    @property
    def has_archived_content(self) -> bool:
        """scratchpad 是否有归档内容（compact 发生过）。"""
        return self._section_count > 0

    def build_reference(self) -> str:
        """生成 scratchpad 的 system prompt 引用文本。

        只在 compact 归档后（scratchpad 有实质内容）才返回引用。
        """
        if not self._section_count:
            return ""
        if not self.path.exists():
            return ""
        size = self.path.stat().st_size
        if size < self._MIN_REF_BYTES:
            return ""
        return (
            f"## Scratchpad (archived context)\n"
            f"Path: `{self.path}`  ({size // 1024}KB)\n"
            f"Earlier conversation was compacted; full original data archived here.\n"
            f"To access: `execute_skill(skill_name=\"filesystem\", "
            f"args={{\"operation\": \"read\", \"path\": \"{self.path}\"}})`\n"
            f"To search: `execute_skill(skill_name=\"search_grep\", "
            f"args={{\"pattern\": \"<keyword>\", \"path\": \"{self.path}\"}})`"
        )
