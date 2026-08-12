# SPDX-License-Identifier: Apache-2.0
"""Shared message utilities — 无 core/ 依赖的纯函数。

迁移自 core/context/message_utils.py 和 core/context/compaction.py。
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from infra.context.providers.shared_prompts import (
    COMPACT_SYSTEM_PROMPT,
    COMPRESS_EMERGENCY_PROMPT,
    COMPRESS_SM_COMPACT_PROMPT,
    COMPRESS_TOOL_RESULT_SYSTEM,
    get_compact_user_summary_message,
)
from utils.token_utils import count_tokens, count_tokens_messages

from infra.compact.constants import DEFAULT_MAX_PTL_RETRIES, DEFAULT_PTL_TRUNCATE_RATIO

# 向后兼容别名
MAX_PTL_RETRIES = DEFAULT_MAX_PTL_RETRIES
PTL_TRUNCATE_RATIO = DEFAULT_PTL_TRUNCATE_RATIO

# LLM client type alias
LLMClient = Callable[..., Awaitable[str]]


def _default_llm_client(
    messages: list[dict[str, Any]],
    *,
    system: str = "",
    max_tokens: int = 4000,
) -> Awaitable[str]:
    """Lazy default: tries to import from middleware.llm.llm_client."""
    from middleware.llm.llm_client import chat_completions_async

    return chat_completions_async(
        system=system,
        messages=messages,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# 消息分组
# ---------------------------------------------------------------------------

def group_messages_by_round(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """按 API 轮次分组。"""
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
    """调整切分索引保证 API 不变量（tool_use/tool_result 配对）。"""
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


def group_aware_trim(
    messages: list[dict[str, Any]], keep_tail: int
) -> list[dict[str, Any]]:
    """从尾部保留 keep_tail 个完整消息组，始终保留 system 消息。"""
    if not messages:
        return messages

    system_msgs: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            system_msgs.append(msg)
        else:
            rest.append(msg)

    if not rest:
        return system_msgs

    groups = group_messages_by_round(rest)

    if len(groups) <= keep_tail:
        return system_msgs + rest

    kept_groups = groups[-keep_tail:]
    kept_msgs: list[dict[str, Any]] = []
    for g in kept_groups:
        kept_msgs.extend(g)

    return system_msgs + kept_msgs


def estimate_message_tokens_rough(messages: list[dict[str, Any]]) -> int:
    """粗略估算消息 token 数（不调用 tokenizer）。"""
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
                    total_chars += len(str(part.get("text", "") or ""))
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            total_chars += len(func.get("name", ""))
            args = func.get("arguments", "")
            total_chars += len(args) if isinstance(args, str) else len(str(args))

    return int(total_chars / 3)


# ---------------------------------------------------------------------------
# 内容提取
# ---------------------------------------------------------------------------

def extract_key_content(content: str, max_tokens: int, model: str = "") -> str:
    """Token-aware progressive content extraction (零 LLM)。"""
    if not content:
        return content

    from utils.token_utils import estimate_tokens_fast

    fast_est = estimate_tokens_fast(content)
    if fast_est <= max_tokens:
        return content

    tokens = count_tokens(content, model=model)
    if tokens <= max_tokens:
        return content

    original_tokens = tokens

    working = _normalize_whitespace(content)
    if estimate_tokens_fast(working) <= max_tokens:
        return working
    tokens = count_tokens(working, model=model)
    if tokens <= max_tokens:
        return working

    structured = _extract_structured(working)
    if structured:
        if estimate_tokens_fast(structured) <= max_tokens:
            return _tag_extraction(
                structured, original_tokens, count_tokens(structured, model=model)
            )
        tokens = count_tokens(structured, model=model)
        if tokens <= max_tokens:
            return _tag_extraction(structured, original_tokens, tokens)
        working = structured

    head_tail = _head_tail_extract(working, max_tokens)
    return _tag_extraction(
        head_tail, original_tokens, estimate_tokens_fast(head_tail)
    )


def _normalize_whitespace(text: str) -> str:
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
        return json.dumps(
            parsed[:5]
            + [f"... ({len(parsed) - 10} items omitted)"]
            + parsed[-5:],
            ensure_ascii=False,
            indent=1,
        )

    return None


def _head_tail_extract(content: str, max_tokens: int) -> str:
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


def _tag_extraction(text: str, original_tokens: int, result_tokens: int) -> str:
    return f"[Extracted from {original_tokens} → {result_tokens} tokens]\n{text}"


# ---------------------------------------------------------------------------
# build_digest — skill 结果结构化总结
# ---------------------------------------------------------------------------

def _extract_skill_output_for_context(output_val: Any) -> str:
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


def build_digest(payload: dict, output_val: dict) -> str:
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


# ---------------------------------------------------------------------------
# 压缩函数
# ---------------------------------------------------------------------------

def _serialize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", tc.get("name", "unknown"))
        args = func.get("arguments", tc.get("arguments", ""))
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        parts.append(f"{name}({args})")
    return "; ".join(parts)


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        tag = "TOOL_RESULT" if role == "tool" else role.upper()
        tc = msg.get("tool_calls")
        if tc and role == "assistant":
            parts.append(f"[TOOL_CALLS]: {_serialize_tool_calls(tc)}")
        if content:
            parts.append(f"[{tag}]: {content}")
    return "\n".join(parts)


def _fallback_extract(messages: list[dict[str, Any]], max_tokens: int = 1500) -> str:
    text = _messages_to_text(messages)
    return extract_key_content(text, max_tokens)


def _truncate_for_ptl_retry(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups = group_messages_by_round(messages)
    if len(groups) <= 2:
        return messages

    drop_count = max(1, int(len(groups) * PTL_TRUNCATE_RATIO))
    kept = groups[drop_count:]
    result: list[dict[str, Any]] = []
    for g in kept:
        result.extend(g)
    return result


async def compress_message(
    msg: dict[str, Any],
    max_msg_tokens: int,
    summary_tokens: int = 800,
    model: str = "",
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """单条消息压缩。"""
    content = msg.get("content", "")
    if not isinstance(content, str) or not content:
        return msg

    tokens = count_tokens(content, model=model)
    if tokens <= max_msg_tokens:
        return msg

    role = msg.get("role", "user")
    client = llm_client or _default_llm_client

    try:
        summary = await client(
            [{"role": "user", "content": content}],
            system=COMPRESS_TOOL_RESULT_SYSTEM,
            max_tokens=summary_tokens,
        )
        result = dict(msg)
        result["content"] = f"[compressed from {role}]\n{summary.strip()}"
        return result
    except Exception:
        result = dict(msg)
        result["content"] = extract_key_content(content, max_msg_tokens, model=model)
        return result


async def compress_for_sm_compact(
    messages_to_summarize: list[dict[str, Any]],
    model: str = "",
    transcript_path: str | None = None,
    summary_tokens: int = 3000,
    llm_client: LLMClient | None = None,
) -> str:
    """SM compact fallback — only called when summary.md is empty。"""
    context = _messages_to_text(messages_to_summarize)
    client = llm_client or _default_llm_client

    try:
        raw = await client(
            [{"role": "user", "content": COMPRESS_SM_COMPACT_PROMPT + "\n\n" + context}],
            system=COMPACT_SYSTEM_PROMPT,
            max_tokens=summary_tokens,
        )
        summary = raw.strip()
    except Exception:
        summary = _fallback_extract(messages_to_summarize, max_tokens=summary_tokens)

    return get_compact_user_summary_message(
        summary,
        transcript_path=transcript_path,
        recent_preserved=True,
    )


async def compress_emergency(
    messages: list[dict[str, Any]],
    model: str = "",
    summary_tokens: int = 4000,
    custom_instructions: str = "",
    llm_client: LLMClient | None = None,
) -> list[dict[str, Any]]:
    """全量 9 段摘要 (含 PTL 重试)。"""
    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    start = 1 if system_msg else 0
    rest = messages[start:]

    if not rest:
        return messages

    prompt_suffix = ""
    if custom_instructions and custom_instructions.strip():
        prompt_suffix = f"\n\nAdditional Instructions:\n{custom_instructions}"

    working_rest = list(rest)
    client = llm_client or _default_llm_client

    for attempt in range(MAX_PTL_RETRIES + 1):
        context = _messages_to_text(working_rest)
        full_prompt = COMPRESS_EMERGENCY_PROMPT + prompt_suffix + "\n\n" + context

        try:
            raw = await client(
                [{"role": "user", "content": full_prompt}],
                system=COMPACT_SYSTEM_PROMPT,
                max_tokens=summary_tokens,
            )
            summary_text = raw.strip()
            summary_msg: dict[str, Any] = {
                "role": "user",
                "content": get_compact_user_summary_message(summary_text),
                "_metadata": {"is_compact_summary": True},
            }
            result = ([system_msg] if system_msg else []) + [summary_msg]
            return result

        except Exception as exc:
            exc_str = str(exc).lower()
            is_ptl = "prompt" in exc_str and "long" in exc_str

            if is_ptl and attempt < MAX_PTL_RETRIES:
                before = len(working_rest)
                working_rest = _truncate_for_ptl_retry(working_rest)
                continue

            raise

    return messages
