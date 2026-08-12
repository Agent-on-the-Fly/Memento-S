# SPDX-License-Identifier: Apache-2.0
"""ArtifactRegistry — 跟踪所有创建的文件，防止幻觉。

防幻觉核心机制：记录所有创建的文件路径和内容摘要，
在 microcompact 和 auto_compact 后仍然完整注入到 prompt，
确保 LLM 每轮都知道"哪些文件已经被创建了"。

设计原则：
- 不落文件，完全在内存中存活
- 不依赖具体工具，所有工具执行后都可以注册 artifact
- 永远不会被 context 压缩掉（通过独立的 prompt 注入点）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ArtifactRecord:
    """单个文件的创建记录。"""

    path: str
    tool: str  # 创建该文件的工具名
    created_turn: int  # 创建时的 turn 编号
    size_bytes: int | None = None
    content_summary: str = ""  # 文件内容摘要（<= 200 chars）
    verified: bool = False  # 是否被 read_file 验证过
    updated_turn: int | None = None  # 最后更新时的 turn 编号
    source: str = "agent"  # 注册来源（agent / hook_detection）


class ArtifactRegistry:
    """
    记录所有创建的文件，确保 LLM 永远知道已创建的工件。

    注入方式：
    - 通过 SkillContext.build_llm_messages() 的 artifact_registry 参数注入
    - 不经过 microcompact 压缩
    - 不经过 auto_compact summarization
    """

    # 单个文件内容摘要的最大长度（防止 prompt 膨胀）
    MAX_CONTENT_SUMMARY: int = 200

    # 超过此大小的文件不读取内容摘要
    MAX_CONTENT_READ_SIZE: int = 50 * 1024  # 50KB

    def __init__(self) -> None:
        self._records: dict[str, ArtifactRecord] = {}

    # ── 写入 ──────────────────────────────────────────────────────────────

    def register(
        self,
        path: str,
        tool: str,
        turn: int,
        content_summary: str = "",
        size_bytes: int | None = None,
        source: str = "agent",
    ) -> None:
        """注册一个已创建的文件。

        路径自动规范化为绝对路径（resolve），
        同路径重复注册时，取 turn 更大的记录覆盖。

        Args:
            path: 文件路径（会自动 resolve 为绝对路径）
            tool: 创建该文件的工具名
            turn: 创建时的 turn 编号
            content_summary: 文件内容摘要
            size_bytes: 文件大小
            source: 注册来源（agent / hook_detection）
        """
        # 路径规范化：转为绝对路径
        try:
            normalized = str(Path(path).resolve())
        except Exception:
            normalized = path

        if normalized in self._records:
            # 已存在：比较 turn，只用更大的覆盖
            existing = self._records[normalized]
            if turn > existing.created_turn:
                existing.created_turn = turn
                existing.tool = tool
                existing.content_summary = content_summary[: self.MAX_CONTENT_SUMMARY]
                existing.size_bytes = size_bytes
                existing.source = source
            return

        self._records[normalized] = ArtifactRecord(
            path=normalized,
            tool=tool,
            created_turn=turn,
            size_bytes=size_bytes,
            content_summary=content_summary[: self.MAX_CONTENT_SUMMARY],
            source=source,
        )

    def mark_verified(self, path: str) -> None:
        """标记文件已被 read_file 验证过。"""
        if path in self._records:
            self._records[path].verified = True

    def is_registered(self, path: str) -> bool:
        """检查文件是否已注册。"""
        return path in self._records

    def get(self, path: str) -> ArtifactRecord | None:
        """获取文件记录。"""
        return self._records.get(path)

    # ── 查询 ──────────────────────────────────────────────────────────────

    def get_created_paths(self) -> list[str]:
        """返回所有创建的文件路径（按创建时间排序）。"""
        return [
            r.path for r in sorted(self._records.values(), key=lambda r: r.created_turn)
        ]

    def get_updated_paths(self) -> list[str]:
        """返回所有更新过的文件路径（按更新时间排序）。"""
        updated = [r for r in self._records.values() if r.updated_turn is not None]
        return [
            r.path
            for r in sorted(updated, key=lambda r: r.updated_turn or r.created_turn)
        ]

    def get_unverified_paths(self) -> list[str]:
        """返回所有未验证的文件路径。"""
        return [r.path for r in self._records.values() if not r.verified]

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def all_paths(self) -> list[str]:
        """返回所有文件路径（创建+更新）。"""
        return self.get_created_paths() + self.get_updated_paths()

    # ── Prompt 注入 ───────────────────────────────────────────────────────

    def to_prompt_section(self) -> str:
        """
        生成注入 prompt 的 artifact 清单。

        永远不会被压缩掉，格式：
        ## Created Files (authoritative)
        - /path/to/file.pdf  [verify if needed]
          └─ 文件内容摘要...
        """
        if not self._records:
            return ""

        lines = ["## Created Files (authoritative — these files exist)"]
        for r in sorted(self._records.values(), key=lambda x: x.created_turn):
            tag = " [verified]" if r.verified else " [verify if needed]"
            lines.append(f"- {r.path}{tag}")
            if r.content_summary:
                lines.append(f"  \u2514\u2500 {r.content_summary}")
            if r.size_bytes is not None:
                size_str = self._format_size(r.size_bytes)
                lines.append(f"  \u2514\u2500 size: {size_str}")

        return "\n".join(lines)

    # ── 静态辅助 ──────────────────────────────────────────────────────────

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes}B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f}KB"
        return f"{size_bytes / (1024 * 1024):.1f}MB"

    @staticmethod
    def read_content_summary(path: str, max_chars: int = 200) -> str:
        """
        读取文件内容摘要（用于注册 artifact 时）。

        安全读取：只读取小文件，捕获所有异常。
        """
        try:
            p = Path(path)
            if not p.exists():
                return ""
            size = p.stat().st_size
            if size > 50 * 1024:
                # 大文件只读取开头
                content = p.read_text(encoding="utf-8", errors="replace")[:max_chars]
                return content + " ... [truncated]"
            content = p.read_text(encoding="utf-8", errors="replace")
            return content[:max_chars]
        except Exception:
            return ""
