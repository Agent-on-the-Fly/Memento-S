# SPDX-License-Identifier: Apache-2.0
"""压缩策略抽象层。

架构设计：
- TokenBudgetPolicy：预算策略（阈值、阶段、估算器）
- ToolResultSummarizer：工具结果摘要策略接口
- SummarizerRegistry：工具名 → 摘要器的注册表

所有压缩相关的 hardcode 均在此文件中配置。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


# ============================================================================
# Token 预算策略
# ============================================================================

@dataclass
class TokenBudgetPolicy:
    """压缩触发策略配置。

    所有阈值均可通过 config 覆盖，不在业务代码中 hardcode。
    """

    #: LLM 输入预算（context_window - max_output_tokens）
    budget: int = 100_000

    #: Stage 1 microcompact 触发阈值（budget 的百分比）
    warn_ratio: float = 0.80

    #: Stage 2 激进截断触发阈值（budget 的百分比）
    urgent_ratio: float = 0.90

    #: microcompact 保留最近 N 个 tool results
    microcompact_keep_recent: int = 3

    #: Stage 2 截断时保留最近 N 个（完整）
    truncate_keep_recent: int = 4

    #: 截断后内容上限字符数（工具名 + 内容）
    truncate_max_chars: int = 200

    @property
    def warn_threshold(self) -> int:
        return int(self.budget * self.warn_ratio)

    @property
    def urgent_threshold(self) -> int:
        return int(self.budget * self.urgent_ratio)

    @property
    def compact_threshold(self) -> int:
        """Layer 2 auto_compact 触发阈值（75% of budget）。"""
        return int(self.budget * 0.75)


# ============================================================================
# 工具结果摘要策略
# ============================================================================

@runtime_checkable
class ToolResultSummarizer(Protocol):
    """工具结果摘要策略协议。

    每个工具可以有自己专属的摘要逻辑，
    通过 SummarizerRegistry 注册。
    """

    def summarize(self, tool_name: str, content: str) -> str:
        """将 tool result 内容摘要为短字符串。

        Args:
            tool_name: 工具名（如 "bash", "read_file"）
            content: 原始输出内容

        Returns:
            摘要后的字符串。如果不需要摘要，返回原内容。
        """
        ...


class DefaultSummarizer:
    """默认摘要器：按工具类型分发到专用摘要器。"""

    def summarize(self, tool_name: str, content: str) -> str:
        if tool_name == "bash":
            return BashSummarizer().summarize(tool_name, content)
        if tool_name == "python_repl":
            return PythonReplSummarizer().summarize(tool_name, content)
        if tool_name == "file_create":
            return FileCreateSummarizer().summarize(tool_name, content)
        if tool_name in {"read_file", "edit_file_by_lines"}:
            return ReadFileSummarizer().summarize(tool_name, content)
        if tool_name == "grep":
            return GrepSummarizer().summarize(tool_name, content)
        if tool_name == "glob":
            return GlobSummarizer().summarize(tool_name, content)
        # 未知工具：保留前 truncate_max_chars 字符
        return content[:200]


class BashSummarizer:
    """bash 输出摘要：提取 exit_code、文件路径、关键摘要行。"""

    _KEY_INDICATORS = frozenset({
        "installed", "created", "saved", "generated",
        "error", "warning", "done", "complete",
        "success", "failed",
    })

    def summarize(self, tool_name: str, content: str) -> str:
        lines = content.split("\n")
        exit_line = ""
        file_paths = []
        summary_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 提取路径
            for seg in stripped.split():
                if "/" in seg and not seg.startswith("-"):
                    p = seg.rstrip(",;)]")
                    if len(p) > 3:
                        file_paths.append(Path(p).name)

            # 提取 exit code（仅匹配行首的 exit 格式，避免误捕获）
            # 格式1: "exit: 0"   格式2: "exit 0"   格式3: 纯数字 "0" / "1"
            if stripped.startswith("0") or stripped.startswith("1"):
                exit_line = stripped[:20].strip()
            elif stripped.lower().startswith("exit"):
                # "exit: N" 或 "exit N" 格式，取冒号/空格后的 exit code
                code_part = stripped[4:].strip()  # 跳过 "exit"
                if code_part and code_part[0] in ": ":
                    exit_line = code_part[1:].strip()[:10]
                else:
                    exit_line = code_part[:10]

            # 提取关键摘要行（排除 exit_code 行，避免重复）
            if (any(k in stripped.lower() for k in self._KEY_INDICATORS)
                    and len(stripped) < 120
                    and "exit" not in stripped.lower()[:20]):
                summary_lines.append(stripped[:100])

        parts = []
        if exit_line:
            parts.append(f"exit={exit_line}")
        if file_paths[:3]:
            parts.append(f"files={', '.join(file_paths[:3])}")
        if summary_lines[:2]:
            parts.append(" | ".join(summary_lines[:2]))
        if not parts:
            parts.append(content[:150])

        return f"[bash] {' | '.join(parts)}"


class PythonReplSummarizer:
    """python_repl 摘要：保留最后一行（通常是有意义的结果）。"""

    def summarize(self, tool_name: str, content: str) -> str:
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if lines:
            last = lines[-1]
            if len(last) < 200:
                return f"[python_repl] {last}"
        return f"[python_repl] {content[:150]}..."


class FileCreateSummarizer:
    """file_create 摘要：提取创建的文件路径。"""

    def summarize(self, tool_name: str, content: str) -> str:
        path_match = re.search(r"(/[^\s'\"`,;]+\.[a-zA-Z0-9]+)", content)
        if path_match:
            return f"[file_create] {path_match.group(1)}"
        return content[:200] + ("..." if len(content) > 200 else "")


class ReadFileSummarizer:
    """read_file / edit_file_by_lines 摘要：保留前 250 字符。"""

    def summarize(self, tool_name: str, content: str) -> str:
        return content[:250] + ("..." if len(content) > 250 else "")


class GrepSummarizer:
    """grep 摘要：保留匹配行数 + 前 3 条匹配片段。"""

    def summarize(self, tool_name: str, content: str) -> str:
        match_count = re.search(r"(\d+)\s+match", content.lower())
        count_str = f"({match_count.group(1)} matches)" if match_count else ""
        sample_lines = [
            l.strip() for l in content.split("\n")
            if l.strip() and not l.strip().startswith("--")
        ][:3]
        if sample_lines:
            return f"[grep] {count_str} {' | '.join(l[:80] for l in sample_lines)}"
        return content[:200]


class GlobSummarizer:
    """glob 摘要：保留前 5 个路径。"""

    def summarize(self, tool_name: str, content: str) -> str:
        paths = re.findall(r"(/[^\s'\"`,;]+)", content)
        if paths[:5]:
            return f"[glob] {', '.join(paths[:5])}"
        return content[:200]


class PipSummarizer:
    """pip install / uv pip install 摘要：提取已安装的包名。"""

    def summarize(self, tool_name: str, content: str) -> str:
        packages = re.findall(r"([a-zA-Z0-9_-]+)==?[0-9.]+", content)
        if packages:
            return f"[pip] installed: {', '.join(packages[:5])}"
        return content[:200]


class FFmpegSummarizer:
    """ffmpeg 输出摘要：提取时间进度和关键信息。"""

    def summarize(self, tool_name: str, content: str) -> str:
        time_match = re.search(r"time=(\d+:\d+:\d+)", content)
        if time_match:
            return f"[ffmpeg] processing at {time_match.group(1)}"
        return content[:200]


# ============================================================================
# 摘要器注册表
# ============================================================================

class SummarizerRegistry:
    """工具名 → 摘要器的注册表。

    默认注册常用工具，新增工具只需在此注册即可，无需改动业务代码。
    """

    def __init__(self, fallback: ToolResultSummarizer | None = None):
        self._map: dict[str, ToolResultSummarizer] = {}
        self._fallback = fallback or DefaultSummarizer()

    def register(self, tool_name: str, summarizer: ToolResultSummarizer) -> None:
        """注册一个工具的摘要器。"""
        self._map[tool_name] = summarizer

    def register_alias(self, alias: str, original: str) -> None:
        """为工具注册别名（指向已有的摘要器）。"""
        if original in self._map:
            self._map[alias] = self._map[original]
        elif original in _DEFAULT_REGISTRY._map:
            self._map[alias] = _DEFAULT_REGISTRY._map[original]

    def get(self, tool_name: str) -> ToolResultSummarizer:
        """获取工具对应的摘要器。"""
        return self._map.get(tool_name, self._fallback)

    def summarize(self, tool_name: str, content: str) -> str:
        """对工具结果执行摘要。"""
        return self.get(tool_name).summarize(tool_name, content)


# ============================================================================
# 默认注册表（全局）
# ============================================================================

def _build_default_registry() -> SummarizerRegistry:
    reg = SummarizerRegistry(fallback=DefaultSummarizer())
    reg.register("bash", BashSummarizer())
    reg.register("python_repl", PythonReplSummarizer())
    reg.register("file_create", FileCreateSummarizer())
    reg.register("read_file", ReadFileSummarizer())
    reg.register("edit_file_by_lines", ReadFileSummarizer())
    reg.register("grep", GrepSummarizer())
    reg.register("glob", GlobSummarizer())
    # 常见别名
    reg.register_alias("uv-pip-install", "bash")
    return reg


_DEFAULT_REGISTRY = _build_default_registry()


def get_default_registry() -> SummarizerRegistry:
    """获取全局默认摘要器注册表。"""
    return _DEFAULT_REGISTRY


# ============================================================================
# 工厂函数（供外部调用）
# ============================================================================

def make_budget_policy(
    context_window: int,
    max_output_tokens: int,
    warn_ratio: float = 0.80,
    urgent_ratio: float = 0.90,
) -> TokenBudgetPolicy:
    """根据 LLM 参数构造预算策略。

    Args:
        context_window: LLM context window
        max_output_tokens: LLM max output tokens
        warn_ratio: Stage 1 触发比例（默认 80%）
        urgent_ratio: Stage 2 触发比例（默认 90%）
    """
    budget = max(context_window - max_output_tokens, 8192)
    return TokenBudgetPolicy(
        budget=budget,
        warn_ratio=warn_ratio,
        urgent_ratio=urgent_ratio,
    )