# SPDX-License-Identifier: Apache-2.0
"""ReAct State — Skill 执行层状态管理。

语义分层：
- observation（观察层）: 每次 tool call 的原始结果，用于调试追踪
- context（上下文层）: 传递给 LLM 的处理后上下文
- result_cache（结果缓存层）: 从 observation 中提取的结构化数据
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .result_cache import ResultCache
from .artifact_registry import ArtifactRegistry

if TYPE_CHECKING:
    from middleware.llm.llm_client import LLMClient
    from .artifact_registry import ArtifactRegistry


# ============================================================================
# 推理日志压缩 — 全局常量
# ============================================================================

# 推理日志特征模式（这些 bash 输出是"进度报告"而非"文件操作"）
INFERENCE_LOG_PATTERNS: list[re.Pattern] = [
    # Faster-Whisper 进度
    re.compile(r"Processing batch\s*\d+/\d+", re.IGNORECASE),
    re.compile(r"Transcribing segment\s*\d+", re.IGNORECASE),
    re.compile(r"Processed\s*\d+\s*segments?", re.IGNORECASE),
    re.compile(r"faster_whisper|vad_filter|whisper_timestamp", re.IGNORECASE),
    # ffmpeg 进度
    re.compile(r"frame\s*\d+.*fps=", re.IGNORECASE),
    re.compile(r"time=\d+:\d+:\d+\.\d+", re.IGNORECASE),
    # 通用进度
    re.compile(r"\[\s*\d+%\]"),
    re.compile(r" ETA|elapsed|remaining|batch \d+/\d+", re.IGNORECASE),
    # Python 执行特征（而非文件操作结果）
    re.compile(r"^>>> ", re.MULTILINE),
    re.compile(r"^In \[\d+\]:", re.MULTILINE),
    re.compile(r"^out\[\d+\]:", re.MULTILINE),
]

# 纯推理工具（输出通常是推理过程，非实质结果）
INFERENCE_TOOLS: frozenset[str] = frozenset({
    "python_repl",
})


def is_inference_output(tool_name: str, content: str) -> str:
    """判断工具输出是否属于推理日志，返回分类标签。

    返回值：
    - "inference": 推理日志，应压缩为摘要
    - "preserve": 实质结果，应完整保留或替换为占位符
    - "summary": 短内容，可保留原样
    """
    if not content:
        return "summary"

    # PRESERVE 工具：完整保留
    _PRESERVE_RESULT_TOOLS = frozenset({
        "read_file", "grep", "glob", "list_dir",
        "file_create", "edit_file_by_lines",
        "fetch_webpage",
    })

    if tool_name in _PRESERVE_RESULT_TOOLS:
        return "preserve"

    # 短内容不压缩
    if len(content) <= 80:
        return "preserve"

    # 推理工具：压缩为摘要
    if tool_name in INFERENCE_TOOLS:
        return "inference"

    # bash：检测是否匹配推理��志模式
    if tool_name == "bash":
        match_count = 0
        for pattern in INFERENCE_LOG_PATTERNS:
            if pattern.search(content):
                match_count += 1

        line_count = content.count("\n") + 1
        if match_count >= 2:
            return "inference"
        if match_count >= 1 and line_count >= 10:
            return "inference"
        # 纯错误信息 → 保留
        if "error" in content.lower()[:200]:
            return "preserve"

    return "preserve"


def compress_to_summary(tool_name: str, content: str) -> str:
    """将长输出压缩为一句话摘要。"""
    content_lower = content.lower()

    # Faster-Whisper 摘要
    if "faster_whisper" in content_lower or "whisper" in content_lower:
        if "saved" in content_lower or "complete" in content_lower or "done" in content_lower:
            return f"[{tool_name}] Transcription completed. Result saved."
        match = re.search(r"processed\s+(\d+)\s+segments?", content_lower)
        if match:
            return f"[{tool_name}] Processing: {match.group(1)} segments done."
        return f"[{tool_name}] Inference in progress..."

    # ffmpeg 摘要
    if "ffmpeg" in content_lower or "frame" in content_lower:
        time_match = re.search(r"time=(\d+:\d+:\d+)", content)
        if time_match:
            return f"[{tool_name}] FFmpeg processing at {time_match.group(1)}"
        return f"[{tool_name}] FFmpeg in progress..."

    # python_repl 摘要
    if tool_name == "python_repl":
        if "error" in content_lower[:100]:
            return f"[{tool_name}] Error occurred. Check output."
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        last_line = lines[-1] if lines else ""
        if last_line and len(last_line) < 100:
            return f"[{tool_name}] Result: {last_line}"
        return f"[{tool_name}] Execution complete."

    # 默认：保留前 80 字符
    preview = content[:80].replace("\n", " ").strip()
    return f"[{tool_name}] {preview}..."


@dataclass
class ContextCompactor:
    """
    SkillAgent 专用内存压缩器。

    Layer 1 (microcompact): 每 turn 压缩旧 tool results，保留最近 N 个。
    Layer 2 (auto_compact): token 超阈值时用 LLM summarization。

    所有压缩操作在内存中完成，不写磁盘 transcript。

    增强项:
    - Priority 1: Tool pair integrity（压缩后清理 orphaned call/result）
    - Priority 2: Head protection + Pre-flight guard
    - Priority 3: Token budget tail protection（自适应 context window）
    - Priority 4: 结构化 13 字段摘要模板 + 迭代压缩 + Anti-thrashing
    - Priority 5: MD5 去重
    """

    #: 保留最近 N 个 tool results 不压缩
    KEEP_RECENT: int = 3

    #: Head 保护：固定保留前 N 条（system + 首次交互）
    protect_first_n: int = 2

    #: Token budget tail 保护（自适应模型 context window）
    tail_token_budget: int = 20000

    #: 预压缩保护：消息少于 N 条时拒绝压缩
    _min_messages_for_compact: int = 6

    # 信息密集型工具：完整保留结果
    READ_RESULT_TOOLS: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "read_file", "grep", "glob", "list_dir",
        })
    )

    # 写操作工具：完整保留（防止 LLM 忘记创建的文件）
    WRITE_RESULT_TOOLS: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "file_create", "edit_file_by_lines", "bash",
        })
    )

    @property
    def PRESERVE_RESULT_TOOLS(self) -> frozenset[str]:
        """读写工具全部保留，不被 microcompact 压缩。"""
        return self.READ_RESULT_TOOLS | self.WRITE_RESULT_TOOLS | frozenset({"fetch_webpage"})

    #: Token 阈值，超出时触发 Layer 2 auto_compact
    threshold: int = 50000

    #: LLM client（用于 Layer 2 summarization，可为 None）
    llm: "LLMClient | None" = None

    #: tool_call_id -> tool_name 映射（由 _bind_tool_name_map 构建）
    _tool_name_map: dict[str, str] = field(default_factory=dict, repr=False)

    # ── Priority 4b: 迭代压缩状态 ──────────────────────────────────────

    #: 上一次压缩生成的摘要（用于迭代增量更新）
    _previous_summary: str | None = None

    #: 连续无效压缩次数（Anti-thrashing）
    _ineffective_compression_count: int = 0

    #: 摘要失败 cooldown 截止时间
    _summary_failure_cooldown_until: float = 0.0

    # ── Priority 2: Pre-flight guard ──────────────────────────────────

    def can_compress(self, messages: list[dict[str, Any]]) -> bool:
        """Pre-flight：消息太少时不压缩。"""
        return len(messages) > self._min_messages_for_compact

    # ── Priority 5: MD5 去重 + microcompact ───────────────────────────

    def microcompact(self, messages: list[dict[str, Any]]) -> bool:
        """
        Layer 1 (每 turn): 保留最近 KEEP_RECENT 个 tool results，
        其余按类型压缩：
        - 推理日志（python_repl、bash 进度输出）→ 压缩为一句话摘要
        - 实质结果（read_file/file_create 等）→ 完整保留
        - 短内容 → 直接保留

        Pass 0（Priority 5）: MD5 去重，同 _prune_old_tool_results。
        返回是否做了压缩。
        """
        tool_results: list[tuple[int, str, str]] = []  # (msg_idx, content, tool_name)

        for msg_idx, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            content_list = [content] if isinstance(content, str) else content
            tool_call_id = msg.get("tool_call_id", "")
            tool_name = self._tool_name_map.get(tool_call_id, "unknown")
            for c in content_list:
                tool_results.append((msg_idx, c, tool_name))

        if len(tool_results) <= self.KEEP_RECENT:
            return False

        modified = False

        # ── Pass 0: MD5 去重（Priority 5）─────────────────
        content_hashes: dict = {}
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 200:
                continue
            h = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()[:12]
            if h in content_hashes:
                messages[i]["content"] = "[Duplicate tool output — same content as a more recent call]"
                modified = True
            else:
                content_hashes[h] = i

        # ── Pass 1: 推理日志 / 实质结果分类压缩 ────────────────────────
        cutoff = len(tool_results) - self.KEEP_RECENT
        for msg_idx, content, tool_name in tool_results[:cutoff]:
            if tool_name in self.PRESERVE_RESULT_TOOLS:
                continue
            if len(content) <= 80:
                continue
            category = is_inference_output(tool_name, content)
            if category == "inference":
                summary = compress_to_summary(tool_name, content)
                messages[msg_idx]["content"] = summary
                modified = True
            elif category == "preserve":
                messages[msg_idx]["content"] = f"[Previous: used {tool_name}]"
                modified = True

        return modified

    # ── Priority 3: Token budget tail ────────────────────────────────

    def _find_tail_cut_by_tokens(
        self,
        messages: list[dict[str, Any]],
        head_end: int,
    ) -> int:
        """从后向前累加 token，找到 tail 开始位置。

        Token budget 是主要依据。最少保留 3 条作为硬保底。
        不在 tool_call/result group 内部切割。
        """
        n = len(messages)
        min_tail = min(3, n - head_end - 1) if n - head_end > 1 else 0
        accumulated = 0
        cut_idx = n

        for i in range(n - 1, head_end - 1, -1):
            content = messages[i].get("content") or ""
            tokens = len(content) // 4 + 10
            if accumulated + tokens > self.tail_token_budget and (n - i) >= min_tail:
                break
            accumulated += tokens
            cut_idx = i

        fallback_cut = n - min_tail
        if cut_idx > fallback_cut:
            cut_idx = fallback_cut

        # 不做 backward alignment：tool result 在 tail 中是正常的，
        # 足够长的 tail 本来就可能以 tool result 开头。
        # truncate_keep_recent = 10 的降级策略保证 tool results 不被完全丢弃。
        return max(cut_idx, head_end + 1)

    # ── Priority 1: Tool pair integrity ──────────────────────────────

    def sanitize_tool_pairs(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """清理压缩后 orphaned 的 tool_call / tool_result 配对。

        两种失败模式:
        1. tool_result 引用的 call_id 在 assistant 消息中不存在 → 移除该 result
        2. assistant 消息有 tool_calls 但对应 result 丢失 → 插入 stub result
        """
        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        cid = tc.get("id", "")
                    else:
                        cid = getattr(tc, "id", "") or ""
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # Pass 1: 收集 orphaned results（call_id 不在 surviving_call_ids 中）
        orphaned_results = result_call_ids - surviving_call_ids

        # Pass 2: 在移除 orphaned results 之前，先为 orphaned calls 插入 stub。
        # stub 的 tool_call_id 在 surviving_call_ids 中，所以移除步骤不会误伤它。
        # 但如果 orphaned results 存在（压缩前 call/result 都在），stub 不应添加。
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: list[dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        if isinstance(tc, dict):
                            cid = tc.get("id", "")
                        else:
                            cid = getattr(tc, "id", "") or ""
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "content": "[Result from earlier conversation — see context summary above]",
                                "tool_call_id": cid,
                            })
            messages = patched

        # Pass 3: 移除 orphaned tool results
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]

        return messages

    # ── Priority 4: 结构化摘要 + 迭代压缩 ────────────────────────────

    def _serialize_for_summary(
        self,
        turns: list[dict[str, Any]],
        max_chars: int = 6000,
    ) -> str:
        """将消息列表序列化为纯文本供 LLM summarization。

        对 tool result 和 assistant 消息分别保留足够信息（tool call arguments、truncation head+tail 策略）。
        """
        parts = []
        HEAD = 4000
        TAIL = 1500

        for msg in turns:
            role = msg.get("role", "unknown")
            content = msg.get("content", "") or ""

            if len(content) > max_chars:
                content = content[:HEAD] + "\n...[truncated]...\n" + content[-TAIL:]

            if role == "tool":
                parts.append(f"[TOOL RESULT {msg.get('tool_call_id', '')}]: {content}")
            elif role == "assistant":
                tcs = msg.get("tool_calls", [])
                if tcs:
                    tc_parts = []
                    for tc in tcs:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = fn.get("arguments", "")
                            if len(args) > 1500:
                                args = args[:1200] + "..."
                            tc_parts.append(f"  {name}({args})")
                        else:
                            fn = getattr(tc, "function", None)
                            name = getattr(fn, "name", "?") if fn else "?"
                            tc_parts.append(f"  {name}(...)")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
            else:
                parts.append(f"[{role.upper()}]: {content}")

        text = "\n\n".join(parts)
        # 最多 30k chars，与原来行为一致
        return text[:30000]

    async def auto_compact(
        self,
        messages: list[dict[str, Any]],
        artifact_section: str = "",
        focus_topic: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Layer 2 (token 超阈值): 用 LLM summarization 替换整个历史为一条摘要消息。

        增强项:
        - Pre-flight guard（消息太少时不压缩）
        - 结构化 13 字段摘要模板（复用 infra/compact/prompts.py）
        - 迭代压缩（_previous_summary 增量更新）
        - Anti-thrashing（连续节省 <10% 则退避）
        - Token budget tail 保护
        - Tool pair integrity 后处理

        artifact_section 永远保留，不被 summarization 压缩掉。
        若 LLM summarization 失败，降级为保留 artifact_section + 最近 10 条消息。
        """
        if not self.can_compress(messages):
            return list(messages)

        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None

        # ── Anti-thrashing（先于 boundary 计算）────────────────────────
        now = time.monotonic()
        if self._ineffective_compression_count >= 2:
            artifact = f"\n\n## CRITICAL: Existing Files\n{artifact_section}" if artifact_section else ""
            return self._fallback_compact(system_msg, messages, artifact)

        if now < self._summary_failure_cooldown_until:
            artifact = f"\n\n## CRITICAL: Existing Files\n{artifact_section}" if artifact_section else ""
            return self._fallback_compact(system_msg, messages, artifact)

        # ── 边界计算：head 保护 + token budget tail ──────────────────
        compress_start = self.protect_first_n
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)

        if compress_start >= compress_end:
            return list(messages)

        # 防止 tool result 独立在 boundary 处（与 _align_boundary_forward 对齐）
        while compress_start < len(messages) and messages[compress_start].get("role") == "tool":
            compress_start += 1

        turns_to_summarize = messages[compress_start:compress_end]

        # artifact_section 永远保留
        if artifact_section:
            artifact_section = f"\n\n## CRITICAL: Existing Files\n{artifact_section}"

        # ── 构建结构化摘要 prompt（Priority 4a）────────────────────────
        from infra.compact.prompts import (
            build_compression_prompt,
            SUMMARIZER_PREAMBLE,
            SUMMARY_PREFIX,
        )

        content_to_summarize = self._serialize_for_summary(turns_to_summarize)
        summary_budget = 2000
        prompt = build_compression_prompt(
            turns_to_summarize=content_to_summarize,
            summary_budget=summary_budget,
            previous_summary=self._previous_summary,
            focus_topic=focus_topic,
        )

        # ── LLM summarization ──────────────────────────────────────────
        tokens_before = self._estimate_tokens(messages)

        if not self.llm:
            return self._fallback_compact(system_msg, messages, artifact_section)

        try:
            summary_response = await self.llm.async_chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=int(summary_budget * 1.3),
            )
            if hasattr(summary_response, "content"):
                summary_text = summary_response.content.strip()
            else:
                summary_text = str(summary_response).strip()
        except Exception:
            self._summary_failure_cooldown_until = now + 60.0
            return self._fallback_compact(system_msg, messages, artifact_section)

        if not summary_text:
            self._summary_failure_cooldown_until = now + 60.0
            return self._fallback_compact(system_msg, messages, artifact_section)

        # ── 迭代摘要：存储供下次使用（Priority 4b）──────────────────────
        self._previous_summary = summary_text
        self._summary_failure_cooldown_until = 0.0

        # 加上 SUMMARY_PREFIX
        summary_with_prefix = f"{SUMMARY_PREFIX}\n{summary_text}"

        # ── 构建压缩后的消息列表 ──────────────────────────────────────
        compressed: list[dict[str, Any]] = []

        for i in range(compress_start):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system":
                note = "[Note: Some earlier conversation turns have been compacted into a handoff summary to preserve context space. The current session state may still reflect earlier work, so build on that summary and state rather than re-doing work.]"
                if note not in (msg.get("content") or ""):
                    msg["content"] = (msg.get("content") or "") + "\n\n" + note
            compressed.append(msg)

        # 智能选择摘要 role：避免与 head/tail 邻接消息同 role
        last_head_role = messages[compress_start - 1].get("role", "user") if compress_start > 0 else "user"
        first_tail_role = messages[compress_end].get("role", "user") if compress_end < len(messages) else "user"

        if last_head_role in ("assistant", "tool"):
            summary_role = "user"
        else:
            summary_role = "assistant"
        if summary_role == first_tail_role:
            flipped = "assistant" if summary_role == "user" else "user"
            if flipped != last_head_role:
                summary_role = flipped

        compressed.append({"role": summary_role, "content": summary_with_prefix + artifact_section})
        compressed.extend([msg.copy() for msg in messages[compress_end:]])

        # ── Anti-thrashing 状态更新 ───────────────────────────────────
        tokens_after = self._estimate_tokens(compressed)
        saved = tokens_before - tokens_after
        savings_pct = (saved / tokens_before * 100) if tokens_before > 0 else 0
        if savings_pct < 10:
            self._ineffective_compression_count += 1
        else:
            self._ineffective_compression_count = 0

        # ── Tool pair integrity 后处理（Priority 1）────────────────────
        compressed = self.sanitize_tool_pairs(compressed)

        return compressed

    def _fallback_compact(
        self,
        system_msg: dict[str, Any] | None,
        messages: list[dict[str, Any]],
        artifact_section: str,
    ) -> list[dict[str, Any]]:
        """Layer 2 降级策略：不做 summarization，保留 artifact + 最近消息。"""
        recent = messages[-10:] if len(messages) > 10 else messages

        result: list[dict[str, Any]] = []
        if system_msg:
            result.append(system_msg)

        if artifact_section:
            result.append({
                "role": "user",
                "content": (
                    f"[Conversation too large — showing recent {len(recent)} messages and all artifacts]"
                    f"{artifact_section}"
                ),
            })

        result.extend([msg.copy() for msg in recent])
        return result

    def _bind_tool_name_map(self, messages: list[dict[str, Any]]) -> None:
        """从 assistant 消息中构建 tool_call_id -> tool_name 映射。

        兼容两种 tool_calls 格式：
        - dict: {"id": ..., "function": {"name": ...}}
        - ToolCall dataclass: .id / .name 属性
        """
        self._tool_name_map.clear()
        for msg in messages:
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    tc_id = tc.get("id") or tc.get("tool_call_id") or ""
                    func = tc.get("function", {})
                    tool_name = func.get("name", "unknown") if isinstance(func, dict) else ""
                else:
                    tc_id = getattr(tc, "id", "") or ""
                    tool_name = getattr(tc, "name", "unknown")
                self._tool_name_map[tc_id] = tool_name

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """估算消息列表 token 数（简化版：字符数 / 4）。"""
        return sum(
            len(str(c))
            for msg in messages
            for c in (
                [msg.get("content", "")] if isinstance(msg.get("content"), str) else msg.get("content", [])
            )
        )



@dataclass
class SkillContext:
    """Skill 执行层的上下文管理。

    职责：
    1. 管理 observation → messages 的转换
    2. 提供 token 感知的上下文压缩
    3. 注入 result_cache 和 scratchpad
    """
    # 配置
    max_history_tokens: int = 4000
    compact_threshold_tokens: int = 3000
    keep_recent_messages: int = 12

    # 原始消息历史
    _raw_messages: list[dict[str, Any]] = field(default_factory=list)

    def append_message(self, role: str, content: Any, **kwargs) -> None:
        """追加消息到历史"""
        msg = {"role": role, "content": str(content) if content else ""}
        msg.update(kwargs)
        self._raw_messages.append(msg)

    def append_assistant(self, text: str = "", tool_calls: Any = None, **kwargs) -> None:
        """追加 assistant 消息"""
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        msg.update(kwargs)
        self._raw_messages.append(msg)

    def append_user(self, role: str = "user", content: Any = "", **kwargs) -> None:
        """追加 user 消息（role 默认为 user）"""
        msg: dict[str, Any] = {"role": role, "content": str(content) if content else ""}
        msg.update(kwargs)
        self._raw_messages.append(msg)

    def append_tool_result(self, tool_call_id: str, tool_name: str, content: str) -> None:
        """追加 tool 结果"""
        self._raw_messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": content,
        })

    def build_llm_messages(
        self,
        system_prompt: str,
        scratchpad: str = "",
        result_cache: "ResultCache | None" = None,
        artifact_registry: "ArtifactRegistry | None" = None,
    ) -> list[dict[str, Any]]:
        """构建发送给 LLM 的消息（带上下文优化和 artifact 注入）。

        注意：不维护 tool_calls 数组，tool_calls 附加在 assistant 消息的字典结构中。
        """
        messages = [{"role": "system", "content": system_prompt}]

        # 注入 scratchpad（永远可见）
        if scratchpad:
            messages.append({
                "role": "user",
                "content": f"[Scratchpad]\n{scratchpad}",
                "_metadata": {"is_scratchpad": True},
            })

        # 注入 artifact 清单（防幻觉核心，永远不会被压缩）
        if artifact_registry:
            artifact_section = artifact_registry.to_prompt_section()
            if artifact_section:
                messages.append({
                    "role": "user",
                    "content": artifact_section,
                    "_metadata": {"is_artifact_registry": True},
                })

        # 注入 result_cache 摘要
        if result_cache:
            cache_section = result_cache.to_prompt_section()
            if cache_section and "no results cached" not in cache_section:
                messages.append({
                    "role": "user",
                    "content": cache_section,
                    "_metadata": {"is_result_cache": True},
                })

        # 处理历史消息（microcompact 在 agent.py 的 tool_call 循环末尾调用）
        if len(self._raw_messages) <= self.keep_recent_messages:
            messages.extend(self._raw_messages)
        else:
            processed = self._compact_history()
            messages.extend(processed)

        return messages

    def _compact_history(self) -> list[dict[str, Any]]:
        """压缩历史消息（保留关键信息）"""
        head = self._raw_messages[:-self.keep_recent_messages]
        tail = self._raw_messages[-self.keep_recent_messages:]

        compacted = []
        for msg in head:
            role = msg.get("role")
            if role == "tool":
                content = str(msg.get("content", ""))[:500]
                if len(str(msg.get("content", ""))) > 500:
                    content += "\n... [truncated]"
                compacted.append({**msg, "content": content})
            else:
                compacted.append(msg)

        compacted.extend(tail)
        return compacted


@dataclass
class ReActState:
    """重构后的 ReAct 状态管理。

    语义分层：
    - observation_log: 观察日志（只用于调试追踪）
    - context: 上下文层（传递给 LLM）
    - result_cache: 结果缓存层（结构化数据存储）
    - scratchpad: 跨轮次的备忘录
    """
    # 核心输入
    query: str
    params: dict[str, Any] | None = None
    max_turns: int = 10
    preferred_core_extension: str | None = None

    # Scratchpad（轻量备忘录，跨轮次保留）
    scratchpad: str = ""

    # 结果缓存（持久层）
    result_cache: ResultCache = field(default_factory=ResultCache)

    # Artifact 注册表（防幻觉核心）
    artifact_registry: ArtifactRegistry = field(default_factory=ArtifactRegistry)

    # 上下文层
    context: SkillContext = field(default_factory=SkillContext)

    # 消息压缩器（独立实现，不依赖 core/context/compaction.py）
    compact_threshold_tokens: int = 50000
    _compactor: ContextCompactor | None = field(default=None, repr=False)

    # 观察日志（只用于调试追踪，不用于 LLM 推理）
    observation_log: list[dict[str, Any]] = field(default_factory=list)

    # 执行统计
    turn_count: int = 0
    tool_calls_count: int = 0

    # 错误追踪
    last_error: str | None = None
    error_history: list[dict[str, Any]] = field(default_factory=list)
    last_error_hash: str | None = None
    repeated_error_count: int = 0
    last_recovery_hint_turn: int = 0

    # 循环检测（legacy — 已由 Hook 系统接管，保留以兼容序列化）
    # 检测逻辑现统一在 LoopSupervisionHook / StallSupervisionHook 中
    last_action_signature: str | None = None
    repeated_action_count: int = 0
    max_repeated_actions: int = 2
    last_state_fingerprint: str | None = None
    repeated_state_fingerprint_count: int = 0
    max_repeated_state_fingerprint: int = 2

    # 工件追踪
    core_artifacts: dict[str, str] = field(default_factory=dict)
    all_artifacts: list[str] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    updated_files: list[str] = field(default_factory=list)
    installed_deps: list[str] = field(default_factory=list)

    # 其他元数据
    seen_urls: set[str] = field(default_factory=set)
    seen_entities: set[str] = field(default_factory=set)

    # Legacy compatibility（保持向后兼容）
    messages: list[dict] = field(default_factory=list)
    no_progress_count: int = 0
    stall_warning_count: int = 0
    last_error_hash: str | None = None

    # Consecutive Final Answer counter（连续 Final Answer 无工具调用）
    consecutive_final_answer_count: int = 0

    # ── progress_projection 缓存 ──────────────────────────────────────────
    _cached_projection: str = ""
    _projection_dirty: bool = True

    CORE_ARTIFACT_EXTENSIONS: set[str] = field(
        default_factory=lambda: {
            ".pptx",
            ".docx",
            ".xlsx",
            ".pdf",
            ".py",
            ".js",
            ".ts",
            ".java",
            ".go",
            ".rs",
            ".html",
            ".css",
            ".md",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".db",
            ".sqlite",
        }
    )

    # ========================================================================
    # 工件管理
    # ========================================================================

    def is_core_artifact(self, path: str) -> bool:
        return Path(path).suffix.lower() in self.CORE_ARTIFACT_EXTENSIONS

    def get_primary_artifact(self) -> str | None:
        if (
            self.preferred_core_extension
            and self.preferred_core_extension in self.core_artifacts
        ):
            return self.core_artifacts[self.preferred_core_extension]
        return next(iter(self.core_artifacts.values()), None)

    def _is_similar_filename(self, name1: str, name2: str) -> bool:
        base1 = re.sub(
            r"[_-]?(v\d+|final|draft|copy|backup|new|old)[_-]?", "", name1.lower()
        )
        base2 = re.sub(
            r"[_-]?(v\d+|final|draft|copy|backup|new|old)[_-]?", "", name2.lower()
        )
        return Path(base1).stem == Path(base2).stem

    def lock_artifact(self, path: str) -> tuple[bool, str | None]:
        path_name = Path(path).name
        suffix = Path(path).suffix.lower()

        if path in self.all_artifacts:
            return True, None

        self.all_artifacts.append(path)

        if not self.is_core_artifact(path):
            return True, None

        if self.preferred_core_extension and suffix != self.preferred_core_extension:
            if self.preferred_core_extension in self.core_artifacts:
                preferred = self.core_artifacts[self.preferred_core_extension]
                return (
                    False,
                    f"Primary artifact is locked to {preferred}. Avoid creating new core artifact type '{suffix}'.",
                )

        if suffix in self.core_artifacts:
            existing = self.core_artifacts[suffix]
            existing_name = Path(existing).name
            if self._is_similar_filename(path_name, existing_name):
                return (
                    False,
                    f"Similar core artifact already exists: {existing_name}. Edit that file instead of creating a new version.",
                )
            return True, None

        self.core_artifacts[suffix] = path
        return True, None

    # ========================================================================
    # 状态更新
    # ========================================================================

    def update_from_observation(self, observation: dict[str, Any]) -> None:
        """从观察更新状态。"""
        self.observation_log.append(observation)
        self._projection_dirty = True  # 状态变化，投影缓存失效

        # NOTE: append_tool_result is NOT called here — it is the caller's
        # responsibility (SkillAgent). Having it here AND in agent.py caused
        # duplicate tool messages (same tool_call_id appearing twice), which
        # results in orphaned tool messages and server-side TemplateError.

        # 更新文件追踪
        delta = observation.get("state_delta") or {}
        for p in delta.get("created_files", []):
            if p not in self.created_files:
                self.created_files.append(p)
        for p in delta.get("updated_files", []):
            if p not in self.updated_files:
                self.updated_files.append(p)
        for d in delta.get("installed_deps", []):
            if d not in self.installed_deps:
                self.installed_deps.append(d)

    def update_scratchpad(self, content: str) -> None:
        """更新备忘录内容（追加模式，带时间戳）"""
        entry = f"[Turn {self.turn_count}] {content.strip()}"
        if self.scratchpad:
            self.scratchpad = f"{self.scratchpad}\n{entry}"
        else:
            self.scratchpad = entry
        # 防止无限增长，保留最近 2000 字符
        if len(self.scratchpad) > 2000:
            self.scratchpad = self.scratchpad[-2000:]
        self._projection_dirty = True  # 状态变化，投影缓存失效

    # ========================================================================
    # 投影生成
    # ========================================================================

    def build_progress_projection(self, use_cache: bool = True) -> str:
        """生成进度投影（带缓存）。"""
        if use_cache and not self._projection_dirty:
            return self._cached_projection

        primary = self.get_primary_artifact() or "<none>"

        lines = [
            "## Current Execution Progress",
            f"- Turns: {self.turn_count}",
            f"- Tool calls: {self.tool_calls_count}",
            f"- Primary artifact: {primary}",
            f"- Created files: {[Path(p).name for p in self.created_files[-5:]]}",
            f"- Updated files: {[Path(p).name for p in self.updated_files[-5:]]}",
            f"- Installed deps: {self.installed_deps[-5:]}",
        ]

        # Scratchpad 注入
        if self.scratchpad:
            lines.extend([
                "",
                "## Scratchpad (persistent notes - NEVER ignore these)",
                self.scratchpad,
            ])

        # 注入 result_cache 摘要
        cache_section = self.result_cache.to_prompt_section()
        if cache_section and "no results cached" not in cache_section:
            lines.extend(["", cache_section])

        self._cached_projection = "\n".join(lines)
        self._projection_dirty = False
        return self._cached_projection

    def invalidate_projection(self) -> None:
        """当状态变化时调用，使投影缓存失效。"""
        self._projection_dirty = True

    def build_outcome_projection(self) -> dict[str, Any]:
        """生成执行结果投影"""
        success_count = sum(
            1 for o in self.observation_log if o.get("exec_status") == "success"
        )
        error_count = sum(
            1 for o in self.observation_log if o.get("exec_status") == "error"
        )

        return {
            "turn_count": self.turn_count,
            "tool_calls": self.tool_calls_count,
            "primary_artifact": self.get_primary_artifact(),
            "created_files": self.created_files,
            "updated_files": self.updated_files,
            "installed_deps": self.installed_deps,
            "observation_stats": {
                "total": len(self.observation_log),
                "success": success_count,
                "error": error_count,
            },
            # 包含 result_cache 状态
            "result_cache": self.result_cache.to_structured_output(),
            "recent_observations": self.observation_log[-5:],
        }

    # ========================================================================
    # 错误追踪
    # ========================================================================

    def record_error(
        self, error: str, tool_name: str, hint_injected: bool = False
    ) -> None:
        """Record an error for pattern detection."""
        error_record = {
            "turn": self.turn_count,
            "tool": tool_name,
            "error": error[:500],
            "timestamp": time.time(),
            "was_recovery_hint_injected": hint_injected,
        }
        self.error_history.append(error_record)
        self.error_history = self.error_history[-20:]

        # Update error hash and count
        current_hash = self._compute_error_fingerprint(error)
        if current_hash == self.last_error_hash:
            self.repeated_error_count += 1
        else:
            self.repeated_error_count = 0
            self.last_error_hash = current_hash

    def _compute_error_fingerprint(self, error: str) -> str | None:
        """Generate normalized error fingerprint."""
        if not error:
            return None

        normalized = error.lower()
        normalized = re.sub(r"line\s+\d+", "line <n>", normalized)
        normalized = re.sub(r"/[^\s:\"']+", "<path>", normalized)
        normalized = re.sub(r"'[^']+'", "'<var>'", normalized)
        normalized = re.sub(r'"[^"]+"', '"<str>"', normalized)
        normalized = re.sub(r"\b\d+\b", "<n>", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return sha1(normalized.encode("utf-8")).hexdigest()[:16]

    def should_inject_recovery_hint(self, min_interval: int = 2) -> bool:
        """Check if we should inject a recovery hint."""
        return (self.turn_count - self.last_recovery_hint_turn) >= min_interval

    def mark_recovery_hint_injected(self) -> None:
        """Mark that a recovery hint was injected this turn."""
        self.last_recovery_hint_turn = self.turn_count
        if self.error_history:
            self.error_history[-1]["was_recovery_hint_injected"] = True


# =============================================================================
# 辅助函数
# =============================================================================

def infer_preferred_extension(query: str, params: dict[str, Any] | None) -> str | None:
    text = (query or "").lower()
    if params:
        try:
            text += "\n" + json.dumps(params, ensure_ascii=False).lower()
        except Exception:
            text += "\n" + str(params).lower()

    extension_hints = [
        (".pptx", ["pptx", "ppt", "slides", "presentation"]),
        (".docx", ["docx", "word", "document"]),
        (".xlsx", ["xlsx", "excel", "spreadsheet"]),
        (".pdf", ["pdf"]),
        (".md", ["markdown", ".md", "readme"]),
        (".py", ["python", ".py", "script"]),
    ]
    for ext, keywords in extension_hints:
        if any(k in text for k in keywords):
            return ext
    return None


def action_signature(tool_name: str, arguments: Any) -> str:
    if isinstance(arguments, dict):
        normalized = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    else:
        normalized = str(arguments)
    sig_raw = f"{tool_name}|{normalized}"
    return sha1(sig_raw.encode("utf-8")).hexdigest()


def state_fingerprint(
    observation: dict[str, Any],
    tool_args: dict[str, Any] | None = None,
) -> str:
    """计算状态指纹，用于检测重复执行。

    改进：包含执行有效性维度，区分"成功但无产物"和"执行失败"。

    只包含"新状态"而非累积历史，避免历史累积导致指纹不变。

    summary 使用完整摘要的 digest（有上限），避免长路径下仅用前 80 字符时
    截断在 ``(Lines X to Y)`` 之前，导致分块 read_file 被误判为等价状态。

    当提供 ``tool_args`` 时并入 ``action_signature``：python_repl 等工具若每次返回
    相同包装 JSON（例如固定 "Slide 1 created" + ``__runner__.py``），仅凭 summary
    会误判为等价；不同代码应视为不同状态。
    """
    delta = observation.get("state_delta") or {}
    exec_status = observation.get("exec_status", "unknown")
    summary = observation.get("summary", "") or ""
    _max_summary_fp = 256 * 1024
    _snip = summary if len(summary) <= _max_summary_fp else summary[:_max_summary_fp]
    summary_digest = sha1(_snip.encode("utf-8")).hexdigest()

    # 关键：区分"成功但无产物"和"执行失败"
    # 只有成功执行且产生了新 artifact 才算有效
    has_artifact = bool(
        delta.get("created_files") or
        delta.get("updated_files") or
        delta.get("result_entities")
    )
    effective = 1 if (exec_status == "success" and has_artifact) else 0

    _tool = observation.get("tool")
    payload: dict[str, Any] = {
        "tool": _tool,
        "exec_status": exec_status,
        "effective": effective,                          # 新增：执行有效性
        "summary_digest": summary_digest,
        "new_files_count": len(delta.get("created_files") or []),
        "updated_files_count": len(delta.get("updated_files") or []),
        "deps_count": len(delta.get("installed_deps") or []),
        "entities_count": len(delta.get("result_entities") or []),
        "error_kind": (observation.get("raw") or {}).get("error_type"),
    }
    if tool_args and isinstance(tool_args, dict) and _tool:
        payload["action_sig"] = action_signature(str(_tool), tool_args)
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return sha1(normalized.encode("utf-8")).hexdigest()
