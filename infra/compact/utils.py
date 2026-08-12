# SPDX-License-Identifier: Apache-2.0
"""工具函数 — 压缩模块的共享工具。"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from infra.compact.models import MessageGroup


# ── 常量 ──────────────────────────────────────────────────────────────

_CHARS_PER_TOKEN = 4

CLEARED_MESSAGE: str = "[Old tool result content cleared]"
"""旧 tool result 清除后的占位符。"""

SKIP_PREFIXES: tuple[str, ...] = (
    "[Output persisted",
    "[Extracted from",
    "[summarized]",
)
"""跳过已处理的 tool result 标记。"""

PTL_TRUNCATE_RATIO: float = 0.2
"""每次 PTL 重试丢弃的最旧消息组比例。"""

_MAX_PTL_RETRIES: int = 3


# ── Token 计数 ────────────────────────────────────────────────────────

def count_messages_tokens(
    messages: list[dict[str, Any]],
    config: Any,
) -> int:
    """计算消息列表的 token 数。

    优先使用注入的 token_counter，否则使用粗略估算。
    """
    if config.token_counter is not None:
        return config.token_counter.count_messages(
            messages, model=config.model
        )
    return estimate_message_tokens_rough(messages)


def count_text_tokens(text: str, config: Any) -> int:
    """计算文本的 token 数。"""
    if config.token_counter is not None:
        return config.token_counter.count_text(text, model=config.model)
    return estimate_tokens_fast(text)


def estimate_tokens_fast(text: str) -> int:
    """O(1) 粗略 token 估算。

    基于 chars / 4 * 4/3 overhead padding。
    """
    if not text:
        return 0
    return len(text) // 3 + 1


def estimate_message_tokens_rough(messages: list[dict[str, Any]]) -> int:
    """粗略估算消息 token 数 (不调用 tokenizer，用于快速判断)。
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    total_chars += len(part)
                elif isinstance(part, dict):
                    total_chars += len(str(part.get("text", "")))
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            total_chars += len(func.get("name", ""))
            args = func.get("arguments", "")
            total_chars += len(args) if isinstance(args, str) else len(str(args))
    return total_chars // 3 + 1


# ── 消息分组 ─────────────────────────────────────────────────────────

def group_messages_by_round(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """按 API 轮次分组。

    每当遇到一条新的 assistant 消息（不同于上一条 assistant 的来源）
    就开始新的一组。同组内 tool_use / tool_result 天然配对。
    """
    if not messages:
        return []

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") == "assistant" and current:
            groups.append(current)
            current = [msg]
        else:
            current.append(msg)

    if current:
        groups.append(current)

    return groups


def adjust_index_to_preserve_invariants(
    messages: list[dict[str, Any]], start_index: int
) -> int:
    """调整切分索引保证 API 不变量。

    从 start_index 开始向前扩展，确保:
    1. 保留范围内的 tool_result 消息，其对应的 tool_use（在 assistant 消息中）
       也必须在保留范围内。
    2. 不拆分同一 assistant 回复的多条消息（tool_calls 链）。
    """
    if start_index <= 0 or start_index >= len(messages):
        return start_index

    adjusted = start_index

    needed_tool_use_ids: set[str] = set()
    present_tool_use_ids: set[str] = set()

    for i in range(adjusted, len(messages)):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "tool":
            tcid = msg.get("tool_call_id", "")
            if tcid:
                needed_tool_use_ids.add(tcid)

        if role == "assistant":
            for tc in msg.get("tool_calls", []):
                tc_id = tc.get("id", "")
                if tc_id:
                    present_tool_use_ids.add(tc_id)

    missing = needed_tool_use_ids - present_tool_use_ids

    if missing:
        for i in range(adjusted - 1, -1, -1):
            if not missing:
                break
            msg = messages[i]
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    tc_id = tc.get("id", "")
                    if tc_id in missing:
                        adjusted = i
                        missing.discard(tc_id)

    while adjusted > 0 and messages[adjusted].get("role") == "tool":
        adjusted -= 1

    return adjusted


# ── 内容处理 ─────────────────────────────────────────────────────────

def _compact_content(content: str) -> str:
    """Replace tool result with structured digest if possible, else empty placeholder."""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            output_val = parsed.get("output")
            if isinstance(output_val, dict):
                return build_digest(parsed, output_val)
    except Exception:
        pass
    return CLEARED_MESSAGE


def build_digest(payload: dict, output_val: dict) -> str:
    """从 skill 结果中提取关键事实 — 做了什么 + 产生了什么。

    提取特定字段组装为文本，产出大小取决于内容本身（典型 3-10 行）。
    不做截断 — 大小控制交给上层的上下文管道。
    """
    skill = payload.get("skill_name", "")
    ok = payload.get("ok", False)
    summary = payload.get("summary", "")

    lines: list[str] = []
    if skill:
        lines.append(f"[{skill}: {'OK' if ok else 'FAIL'}] {summary}")

    raw_output = _extract_skill_output_for_context(output_val)
    if raw_output:
        lines.append(raw_output)

    exec_sum = output_val.get("execution_summary", {})
    if isinstance(exec_sum, dict):
        for key in ("created_files", "updated_files", "primary_artifact", "installed_deps"):
            val = exec_sum.get(key)
            if val:
                lines.append(f"{key}: {val}")
    return "\n".join(lines)


def _extract_skill_output_for_context(output_val: Any) -> str:
    """从 skill 的 output 字段中提取适合放入上下文的原始数据。"""
    if isinstance(output_val, str) and output_val.strip():
        return output_val.strip()
    if isinstance(output_val, dict):
        for key in ("result", "results", "data", "output", "text", "content"):
            val = output_val.get(key)
            if val is not None:
                if isinstance(val, str) and val.strip():
                    return val.strip()
                if isinstance(val, list):
                    return json.dumps(val, ensure_ascii=False, indent=2)
        return json.dumps(output_val, ensure_ascii=False, indent=2)
    if isinstance(output_val, list):
        return json.dumps(output_val, ensure_ascii=False, indent=2)
    return ""


def extract_key_content(content: str, max_tokens: int, model: str = "") -> str:
    """Token-aware progressive content extraction.

    Stages:
      1. Fast estimate: if obviously fits → return as-is
      2. Normalize whitespace (collapse blank lines)
      3. For JSON: extract key fields
      4. Keep head + tail, remove middle (char-based, O(1))
    """
    if not content:
        return content

    fast_est = estimate_tokens_fast(content)
    if fast_est <= max_tokens:
        return content

    original_tokens = fast_est
    working = _normalize_whitespace(content)
    if estimate_tokens_fast(working) <= max_tokens:
        return working

    structured = _extract_structured(working)
    if structured:
        if estimate_tokens_fast(structured) <= max_tokens:
            return _tag_extraction(structured, original_tokens)
        working = structured

    head_tail = _head_tail_extract(working, max_tokens)
    return _tag_extraction(head_tail, original_tokens)


def _normalize_whitespace(text: str) -> str:
    """Collapse 3+ consecutive blank lines to 2, strip trailing whitespace."""
    lines = text.split("\n")
    result: list[str] = []
    blank_count = 0
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            blank_count += 1
            if blank_count <= 2:
                result.append("")
        else:
            blank_count = 0
            result.append(stripped)
    return "\n".join(result)


def _extract_structured(content: str) -> str | None:
    """For JSON content, extract key fields to reduce size."""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(parsed, dict):
        key_fields: dict[str, Any] = {}
        priority_keys = (
            "ok", "status", "error", "error_code", "summary",
            "output", "result", "skill_name", "diagnostics",
        )
        for k in priority_keys:
            if k in parsed:
                val = parsed[k]
                if isinstance(val, str) and len(val) > 500:
                    val = val[:500] + "..."
                key_fields[k] = val

        remaining = {k: v for k, v in parsed.items() if k not in key_fields}
        for k, v in remaining.items():
            sv = str(v)
            if len(sv) > 200:
                key_fields[k] = sv[:200] + "..."
            else:
                key_fields[k] = v

        return json.dumps(key_fields, ensure_ascii=False, indent=1)

    if isinstance(parsed, list) and len(parsed) > 10:
        truncated = (
            parsed[:5]
            + [f"... ({len(parsed) - 10} items omitted)"]
            + parsed[-5:]
        )
        return json.dumps(truncated, ensure_ascii=False, indent=1)

    return None


def _head_tail_extract(content: str, max_tokens: int) -> str:
    """Keep head and tail of content, remove middle.

    Uses char-based slicing (O(1)) instead of per-line tokenize.
    Approximate: 1 token ~ 4 chars.
    """
    target_chars = max(80, max_tokens * 4)
    if len(content) <= target_chars:
        return content

    head_chars = int(target_chars * 0.6)
    tail_chars = max(0, target_chars - head_chars - 40)

    if tail_chars == 0:
        return content[:head_chars] + "\n... [truncated] ..."

    head_end = content.rfind("\n", 0, head_chars)
    if head_end <= 0:
        head_end = head_chars

    tail_start = content.find("\n", len(content) - tail_chars)
    if tail_start < 0 or tail_start <= head_end:
        tail_start = len(content) - tail_chars

    head = content[:head_end]
    tail = content[tail_start:]
    omitted_chars = tail_start - head_end
    omitted_lines = content[head_end:tail_start].count("\n")

    return (
        head
        + f"\n... [{omitted_lines} lines / ~{omitted_chars} chars omitted] ...\n"
        + tail
    )


def _tag_extraction(text: str, original_tokens: int) -> str:
    """Prepend extraction metadata tag."""
    result_tokens = estimate_tokens_fast(text)
    return f"[Extracted from {original_tokens} → {result_tokens} tokens]\n{text}"


# ── PTL 重试 ─────────────────────────────────────────────────────────

def truncate_for_ptl_retry(
    messages: list[dict[str, Any]],
    truncate_ratio: float = PTL_TRUNCATE_RATIO,
) -> list[dict[str, Any]]:
    """PTL 重试时丢弃最旧的 API round groups。"""
    groups = group_messages_by_round(messages)
    if len(groups) <= 2:
        return messages

    drop_count = max(1, int(len(groups) * truncate_ratio))
    kept = groups[drop_count:]
    result: list[dict[str, Any]] = []
    for g in kept:
        result.extend(g)
    return result


# ── 序列化 ───────────────────────────────────────────────────────────

def serialize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Serialize tool_calls to a compact textual representation."""
    parts: list[str] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", tc.get("name", "unknown"))
        args = func.get("arguments", tc.get("arguments", ""))
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        parts.append(f"{name}({args})")
    return "; ".join(parts)


def messages_to_text(messages: list[dict[str, Any]]) -> str:
    """将消息列表序列化为纯文本，供 LLM 压缩使用。"""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        tag = "TOOL_RESULT" if role == "tool" else role.upper()
        tc = msg.get("tool_calls")
        if tc and role == "assistant":
            parts.append(f"[TOOL_CALLS]: {serialize_tool_calls(tc)}")
        if content:
            parts.append(f"[{tag}]: {content}")
    return "\n".join(parts)


# ── Tail Token Budget ────────────────────────────────────────────────

def _align_boundary_backward(
    messages: list[dict[str, Any]], idx: int
) -> int:
    """Pull compress-end boundary backward to avoid splitting tool_call/result group.

    If the boundary falls in the middle of a tool-result group (consecutive tool
    messages before idx), walk backward past all of them to find the parent
    assistant message. Include the whole group in the summarised region rather
    than splitting it (which causes silent data loss when sanitize_tool_pairs
    removes orphaned tail results).
    """
    if idx <= 0 or idx >= len(messages):
        return idx
    check = idx - 1
    while check >= 0 and messages[check].get("role") == "tool":
        check -= 1
    if (
        check >= 0
        and messages[check].get("role") == "assistant"
        and messages[check].get("tool_calls")
    ):
        idx = check
    return idx


def _ensure_last_user_message_in_tail(
    messages: list[dict[str, Any]],
    cut_idx: int,
    head_end: int,
) -> int:
    """Ensure the last user message is in the protected tail.

    Prevents active task (latest user request) from being compressed into
    the middle region, which would cause the agent to lose the current
    task and repeat work.
    """
    n = len(messages)
    # Find the last user message at or after head_end
    last_user_idx = -1
    for i in range(n - 1, head_end - 1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx < 0:
        # No user message found beyond head — nothing to anchor
        return cut_idx

    if last_user_idx >= cut_idx:
        # Already in tail — nothing to do
        return cut_idx

    # The last user message is in the middle (compressed) region.
    # Pull cut_idx back to it.  A user message is a clean boundary
    # (no tool_call/result splitting risk), so no need to re-align backward.
    # Safety floor: never go back into the head region.
    return max(last_user_idx, head_end + 1)


def _align_boundary_forward(
    messages: list[dict[str, Any]], idx: int
) -> int:
    """Push compress-start boundary forward past any orphan tool results.

    If messages[idx] is a tool result, slide forward until we hit a non-tool
    message so we don't start the summarised region mid-group.
    """
    while idx < len(messages) and messages[idx].get("role") == "tool":
        idx += 1
    return idx


def find_tail_cut_by_tokens(
    messages: list[dict[str, Any]],
    head_end: int,
    tail_token_budget: int,
    min_tail_messages: int = 3,
) -> int:
    """Walk backward from end, accumulating tokens until budget is reached.

    Returns the index where the tail starts.  Token budget is the primary
    criterion.  A hard minimum of min_tail_messages is always protected, but
    the budget is allowed to exceed by up to 1.5x to avoid cutting inside an
    oversized message (tool output, file read, etc.).  If even the minimum
    exceeds 1.5x the budget the cut is placed right after the head so
    compression can still remove middle turns.

    Never cuts inside a tool_call/result group — uses _align_boundary_backward.
    """
    n = len(messages)
    min_tail = min(min_tail_messages, n - head_end - 1) if n - head_end > 1 else 0
    soft_ceiling = int(tail_token_budget * 1.5)
    accumulated = 0
    cut_idx = n

    for i in range(n - 1, head_end - 1, -1):
        msg = messages[i]
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(str(p.get("text", "")) for p in content)
        msg_tokens = len(content) // _CHARS_PER_TOKEN + 10
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                args = tc.get("function", {}).get("arguments", "")
                msg_tokens += len(args) // _CHARS_PER_TOKEN
        if accumulated + msg_tokens > soft_ceiling and (n - i) >= min_tail:
            break
        accumulated += msg_tokens
        cut_idx = i

    fallback_cut = n - min_tail
    if cut_idx > fallback_cut:
        cut_idx = fallback_cut

    cut_idx = _align_boundary_backward(messages, cut_idx)

    # Anchor last user message in tail (prevents active task loss)
    cut_idx = _ensure_last_user_message_in_tail(messages, cut_idx, head_end)

    return max(cut_idx, head_end + 1)


# ── Tool Result Informative Summaries ─────────────────────────────────

_DUPLICATE_PLACEHOLDER = "[Duplicate tool output — same content as a more recent call]"
_PRUNED_PLACEHOLDER = "[Old tool output cleared to save context space]"


def summarize_tool_result(
    tool_name: str, tool_args: str, tool_content: str,
) -> str:
    """Generate an informative 1-line summary of a tool call + result.

    Purely generic — no hard-coded tool names. Extracts semantic key fields
    from tool_args and result scale from tool_content to produce a summary
    like::

        [bash] command='npm test' — exit 0, 47 lines
        [read_file] path='config.py' — 1,200 chars
        [grep] pattern='compress' — 12 matches
    """
    # ── Step 1: Extract semantic key fields from tool_args ──────────────
    try:
        args: dict[str, Any] = json.loads(tool_args) if tool_args else {}
    except (json.JSONDecodeError, TypeError):
        args = {}

    # Ordered priority list: first non-empty key wins as the primary label
    _SEMANTIC_KEYS = (
        "path", "file", "url", "command", "cmd", "script",
        "pattern", "query", "goal", "name", "action",
        "content", "target", "model", "skill_name",
    )
    primary_key, primary_val = "", ""
    for key in _SEMANTIC_KEYS:
        val = args.get(key)
        if val and str(val).strip():
            primary_key, primary_val = key, str(val)
            break

    # Truncate long values for readability
    if len(primary_val) > 80:
        primary_val = primary_val[:77] + "..."

    # ── Step 2: Extract result scale from tool_content ───────────────────
    content = tool_content or ""
    content_len = len(content)
    line_count = content.count("\n") + 1 if content.strip() else 0

    # Try to extract exit code from JSON-structured output
    exit_match = re.search(r'"exit_code"\s*:\s*(-?\d+)', content)
    exit_code = exit_match.group(1) if exit_match else None

    # Try to extract match count
    count_match = re.search(
        r'"(total_count|total|count|match_count|found|ok)"\s*:\s*(\d+)',
        content, re.IGNORECASE
    )
    match_count = count_match.group(2) if count_match else None

    # Build scale string
    if exit_code is not None:
        scale = f"exit {exit_code}, {line_count} lines"
    elif match_count is not None:
        scale = f"{match_count} matches, {content_len:,} chars"
    else:
        scale = f"{content_len:,} chars"

    # ── Step 3: Compose output ───────────────────────────────────────────
    if primary_key and primary_val:
        return f"[{tool_name}] {primary_key}='{primary_val}' — {scale}"
    return f"[{tool_name}] {scale}"


def deduplicate_and_summarize_tool_results(
    messages: list[dict[str, Any]],
    protect_tail_tokens: int | None = None,
    protect_tail_count: int = 5,
) -> tuple[list[dict[str, Any]], int]:
    """Replace old tool results with informative summaries and deduplicate.

    Walks backward from the end, protecting the most recent messages that fall
    within protect_tail_tokens (when provided) OR protect_tail_count messages
    (backward-compatible default).  When both are given, the token budget
    takes priority and the message count acts as a hard minimum floor.

    Returns (pruned_messages, pruned_count).
    """
    if not messages:
        return messages, 0

    result = [m.copy() for m in messages]
    pruned = 0

    # Build index: tool_call_id -> (tool_name, arguments_json)
    call_id_to_tool: dict[str, tuple[str, str]] = {}
    for msg in result:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    cid = tc.get("id", "")
                    fn = tc.get("function", {})
                    call_id_to_tool[cid] = (fn.get("name", "unknown"), fn.get("arguments", ""))
                else:
                    cid = getattr(tc, "id", "") or ""
                    fn = getattr(tc, "function", None)
                    name = getattr(fn, "name", "unknown") if fn else "unknown"
                    args_str = getattr(fn, "arguments", "") if fn else ""
                    call_id_to_tool[cid] = (name, args_str)

    # Determine the prune boundary
    if protect_tail_tokens is not None and protect_tail_tokens > 0:
        accumulated = 0
        boundary = len(result)
        min_protect = min(protect_tail_count, len(result) - 1)
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            raw_content = msg.get("content") or ""
            if isinstance(raw_content, list):
                content_len = sum(len(p.get("text", "")) for p in raw_content)
            else:
                content_len = len(raw_content)
            msg_tokens = content_len // _CHARS_PER_TOKEN + 10
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    msg_tokens += len(args) // _CHARS_PER_TOKEN
            if accumulated + msg_tokens > protect_tail_tokens and (len(result) - i) >= min_protect:
                boundary = i
                break
            accumulated += msg_tokens
            boundary = i
        prune_boundary = max(boundary, len(result) - min_protect)
    else:
        prune_boundary = len(result) - protect_tail_count

    # Pass 1: Deduplicate identical tool results
    content_hashes: dict[str, tuple[int, str]] = {}
    for i in range(len(result) - 1, -1, -1):
        msg = result[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content") or ""
        if isinstance(content, list):
            continue
        if len(content) < 200:
            continue
        h = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()[:12]
        if h in content_hashes:
            result[i] = {**msg, "content": _DUPLICATE_PLACEHOLDER}
            pruned += 1
        else:
            content_hashes[h] = (i, msg.get("tool_call_id", "?"))

    # Pass 2: Replace old tool results with informative summaries
    for i in range(prune_boundary):
        msg = result[i]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            continue
        if not content or content in (_PRUNED_PLACEHOLDER, _DUPLICATE_PLACEHOLDER):
            continue
        if len(content) > 200:
            call_id = msg.get("tool_call_id", "")
            tool_name, tool_args = call_id_to_tool.get(call_id, ("unknown", ""))
            summary = summarize_tool_result(tool_name, tool_args, content)
            result[i] = {**msg, "content": summary}
            pruned += 1

    # Pass 3: Truncate large tool_call arguments in assistant messages
    for i in range(prune_boundary):
        msg = result[i]
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        new_tcs = []
        modified = False
        for tc in msg["tool_calls"]:
            if isinstance(tc, dict):
                args = tc.get("function", {}).get("arguments", "")
                if len(args) > 500:
                    tc = {**tc, "function": {**tc["function"], "arguments": args[:200] + "...[truncated]"}}
                    modified = True
            new_tcs.append(tc)
        if modified:
            result[i] = {**msg, "tool_calls": new_tcs}

    return result, pruned


# ── Tool Pair Integrity ────────────────────────────────────────────────

def sanitize_tool_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fix orphaned tool_call / tool_result pairs after compression.

    Two failure modes:
    1. A tool *result* references a call_id whose assistant tool_call was
       removed (summarized/truncated).  The API rejects this.
    2. An assistant message has tool_calls whose results were dropped.
       The API rejects this because every tool_call must be followed by
       a tool result with the matching call_id.

    This method removes orphaned results and inserts stub results for
    orphaned calls so the message list is always well-formed.
    """
    surviving_call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                cid: str = ""
                if isinstance(tc, dict):
                    cid = tc.get("id", "")
                else:
                    cid = getattr(tc, "id", "") or ""
                if cid:
                    surviving_call_ids.add(cid)

    result_call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            cid = msg.get("tool_call_id")
            if cid:
                result_call_ids.add(cid)

    # 1. Remove tool results whose call_id has no matching assistant tool_call
    orphaned_results = result_call_ids - surviving_call_ids
    if orphaned_results:
        messages = [
            m for m in messages
            if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
        ]

    # 2. Add stub results for assistant tool_calls whose results were dropped
    surviving_call_ids_after = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                cid: str = ""
                if isinstance(tc, dict):
                    cid = tc.get("id", "")
                else:
                    cid = getattr(tc, "id", "") or ""
                if cid:
                    surviving_call_ids_after.add(cid)

    result_call_ids_after: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            cid = msg.get("tool_call_id")
            if cid:
                result_call_ids_after.add(cid)

    missing_results = surviving_call_ids_after - result_call_ids_after
    if missing_results:
        patched: list[dict[str, Any]] = []
        for msg in messages:
            patched.append(msg)
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid: str = ""
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

    return messages


# ── Compression Summary Prefix ────────────────────────────────────────

SUMMARY_PREFIX: str = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. Respond ONLY to the latest user message "
    "that appears AFTER this summary. The current session state (files, "
    "config, etc.) may reflect work described here — avoid repeating it:"
)

LEGACY_SUMMARY_PREFIX: str = "[CONTEXT SUMMARY]:"


def with_summary_prefix(summary: str) -> str:
    """Normalize summary text to the current compaction handoff format."""
    text = (summary or "").strip()
    for prefix in (LEGACY_SUMMARY_PREFIX, SUMMARY_PREFIX):
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
            break
    return f"{SUMMARY_PREFIX}\n{text}" if text else SUMMARY_PREFIX


def extract_summary_from_messages(messages: list[dict[str, Any]]) -> str | None:
    """Extract the summary text from a compressed message list."""
    for msg in messages:
        content = msg.get("content", "")
        if "CONTEXT COMPACTION" in content or ("Goal" in content and "Completed Actions" in content):
            return content
    return None
