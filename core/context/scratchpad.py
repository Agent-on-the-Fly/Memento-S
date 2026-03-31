"""Scratchpad — session 级别的 append-only 持久化文件 + artifact 存储。

Tool result 处理策略:
  - persist_tool_result: 短内容内联返回；长内容存为 artifact 文件，返回 ref + preview
  - compact 触发时: 将待压缩消息批量归档到 scratchpad

文件位置:
  scratchpad: {data_dir}/context/{YYYY-MM-DD}/scratchpad_{session_id}.md
  artifacts:  {data_dir}/context/{YYYY-MM-DD}/artifacts_{session_id}/artifact_NNNN.md
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


def _make_preview(content: str, max_lines: int = 5, max_chars: int = 500) -> str:
    """从原始内容中提取 preview：取前 max_lines 行，截断到 max_chars 字符。"""
    lines = content.split("\n")
    preview_lines = lines[:max_lines]
    preview = "\n".join(preview_lines)
    if len(preview) > max_chars:
        preview = preview[:max_chars]
    return preview


class Scratchpad:
    """单个 session 的 scratchpad 文件管理 + artifact 存储。

    Attributes:
        path: scratchpad 文件的绝对路径。
        artifacts_dir: artifact 存储目录。
    """

    _MIN_REF_BYTES = 100

    def __init__(
        self,
        session_id: str,
        date_dir: Path,
        *,
        artifact_fold_char_limit: int = 4000,
        artifact_fold_line_limit: int = 120,
        artifact_preview_max_lines: int = 5,
        artifact_preview_max_chars: int = 500,
    ) -> None:
        """初始化 scratchpad 文件和 artifact 目录。

        Args:
            session_id: 当前会话 ID。
            date_dir: 日期目录路径（如 {workspace}/context/2026-03-17/）。
            artifact_fold_char_limit: tool result 超过此字符数时存为 artifact。
            artifact_fold_line_limit: tool result 超过此行数时存为 artifact。
            artifact_preview_max_lines: artifact preview 最大行数。
            artifact_preview_max_chars: artifact preview 最大字符数。
        """
        self.path = date_dir / f"scratchpad_{session_id}.md"
        self._section_count: int = 0

        # artifact 配置
        self._fold_char_limit = artifact_fold_char_limit
        self._fold_line_limit = artifact_fold_line_limit
        self._preview_max_lines = artifact_preview_max_lines
        self._preview_max_chars = artifact_preview_max_chars

        # artifact 目录
        self.artifacts_dir = date_dir / f"artifacts_{session_id}"
        self._artifact_count: int = 0

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

        # 恢复已有 artifact 计数
        if self.artifacts_dir.exists():
            self._artifact_count = sum(
                1 for f in self.artifacts_dir.iterdir()
                if f.name.startswith("artifact_") and f.is_file()
            )

    # ═══════════════════════════════════════════════════════════════
    # Scratchpad 写入
    # ═══════════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════════
    # Artifact fold
    # ═══════════════════════════════════════════════════════════════

    def _should_fold(self, content: str) -> bool:
        """判断 tool result 是否应该被 fold 到 artifact 文件。"""
        if len(content) > self._fold_char_limit:
            return True
        if content.count("\n") + 1 > self._fold_line_limit:
            return True
        return False

    def save_artifact(self, tool_name: str, content: str) -> tuple[str, str]:
        """将长内容存为 artifact 文件。

        Args:
            tool_name: 工具名称，用于 artifact 文件头部标注。
            content: 原始完整内容。

        Returns:
            (ref_path, preview):
                ref_path — artifact 文件的相对路径（相对于 date_dir）
                preview — 截取的 preview 文本
        """
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._artifact_count += 1
        filename = f"artifact_{self._artifact_count:04d}.md"
        artifact_path = self.artifacts_dir / filename

        header = (
            f"# Artifact: {tool_name}\n"
            f"> created: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"> size: {len(content)} chars\n\n"
        )
        try:
            artifact_path.write_text(header + content, encoding="utf-8")
        except OSError:
            logger.warning(
                "Failed to write artifact: {}", artifact_path, exc_info=True
            )

        ref_path = f"{self.artifacts_dir.name}/{filename}"
        preview = _make_preview(
            content,
            max_lines=self._preview_max_lines,
            max_chars=self._preview_max_chars,
        )

        logger.info(
            "Artifact saved: {} ({} chars) -> {}",
            tool_name,
            len(content),
            ref_path,
        )
        return ref_path, preview

    def persist_tool_result(
        self, tool_call_id: str, tool_name: str, result: str
    ) -> dict[str, Any]:
        """返回 tool message。短内容内联；长内容存 artifact，返回 ref + preview。

        判断规则:
          - 字符数 > artifact_fold_char_limit → fold
          - 行数 > artifact_fold_line_limit → fold
          - 否则 → 内联返回（行为与之前完全一致）
        """
        if not self._should_fold(result):
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
            }

        # fold: 存 artifact，返回 ref + preview
        ref_path, preview = self.save_artifact(tool_name, result)
        artifact_abs = str(self.artifacts_dir.parent / ref_path)
        folded_content = (
            f"[artifact_ref: {ref_path}]\n"
            f"{preview}\n"
            f"[{len(result)} chars, full content archived — "
            f"use read_file(path=\"{artifact_abs}\") to retrieve full text]"
        )
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": folded_content,
        }

    # ═══════════════════════════════════════════════════════════════
    # Compact 归档
    # ═══════════════════════════════════════════════════════════════

    def archive_messages(self, messages: list[dict[str, Any]]) -> None:
        """将待压缩的消息批量归档到 scratchpad（compact 触发时调用）。

        跳过 system 消息。保留 tool_calls 和 tool_call_id 元信息。
        """
        for msg in messages:
            role = msg.get("role", "")
            if role == "system":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)

            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            parts: list[str] = []
            if tool_calls:
                tc_lines = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", tc.get("name", "?"))
                    args = func.get("arguments", tc.get("arguments", ""))
                    tc_lines.append(f"  - {name}({args})")
                parts.append("**tool_calls:**\n" + "\n".join(tc_lines))
            if tool_call_id:
                parts.append(f"**tool_call_id:** {tool_call_id}")
            if content:
                formatted = self._format_for_scratchpad(content) if role == "tool" else content
                parts.append(formatted)

            if not parts:
                continue
            self.write(f"{role}", "\n".join(parts))

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
        artifacts_hint = ""
        if self.artifacts_dir.exists() and self._artifact_count > 0:
            artifacts_hint = (
                f"\nArtifacts dir: `{self.artifacts_dir}`  ({self._artifact_count} files)\n"
                f"Use `read_file(path=\"<artifact_absolute_path>\")` to retrieve full content of any archived artifact."
            )
        return (
            f"## Scratchpad (archived context)\n"
            f"Path: `$SCRATCHPAD`  ({size // 1024}KB)\n"
            f"Earlier conversation was compacted; full original data archived here.\n"
            f"To access: `execute_skill(skill_name=\"filesystem\", "
            f"args={{\"operation\": \"read\", \"path\": \"$SCRATCHPAD\"}})`\n"
            f"To search: `execute_skill(skill_name=\"search_grep\", "
            f"args={{\"pattern\": \"<keyword>\", \"path\": \"$SCRATCHPAD\"}})`"
            f"{artifacts_hint}"
        )
