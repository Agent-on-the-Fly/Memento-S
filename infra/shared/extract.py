# SPDX-License-Identifier: Apache-2.0
"""共享内容提取工具 — smart_extract。

从 infra/context/providers/artifact_impl.py 和
infra/compact/extract.py 合并而来。
无 core/ 依赖。
"""

from __future__ import annotations

import json as _json
import re as _re
from typing import Any, Awaitable, Callable

from utils.logger import get_logger
from utils.token_utils import count_tokens, estimate_tokens_fast

logger = get_logger(__name__)

# LLM client type alias
LLMClient = Callable[..., Awaitable[str]]


# ---------------------------------------------------------------------------
# 结构化提取
# ---------------------------------------------------------------------------

def _extract_structured(content: str) -> str | None:
    """For JSON content, extract key fields to reduce size."""
    try:
        parsed = _json.loads(content)
    except (_json.JSONDecodeError, TypeError):
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

        return _json.dumps(key_fields, ensure_ascii=False, indent=1)

    if isinstance(parsed, list) and len(parsed) > 10:
        return _json.dumps(
            parsed[:5]
            + [f"... ({len(parsed) - 10} items omitted)"]
            + parsed[-5:],
            ensure_ascii=False,
            indent=1,
        )

    return None


# ---------------------------------------------------------------------------
# build_digest — skill 结果结构化总结
# ---------------------------------------------------------------------------

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
                    return _json.dumps(val, ensure_ascii=False, indent=2)
        return _json.dumps(output_val, ensure_ascii=False, indent=2)
    if isinstance(output_val, list):
        return _json.dumps(output_val, ensure_ascii=False, indent=2)
    return ""


def _build_digest(payload: dict, output_val: dict) -> str:
    """从 skill 结果中提取关键事实 — 做了什么 + 产生了什么。"""
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


# 公开别名（兼容旧的从 shared_utils 导入）
build_digest = _build_digest


# ---------------------------------------------------------------------------
# extract_key_content — Token-aware progressive content extraction
# ---------------------------------------------------------------------------

def extract_key_content(content: str, max_tokens: int, model: str = "") -> str:
    """Token-aware progressive content extraction (零 LLM)。"""
    if not content:
        return content

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
            return _tag_extraction(structured, original_tokens, count_tokens(structured, model=model))
        tokens = count_tokens(structured, model=model)
        if tokens <= max_tokens:
            return _tag_extraction(structured, original_tokens, tokens)
        working = structured

    head_tail = _head_tail_extract(working, max_tokens)
    return _tag_extraction(head_tail, original_tokens, estimate_tokens_fast(head_tail))


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


def _head_tail_extract(content: str, max_tokens: int) -> str:
    """Keep head and tail of content, remove middle (char-based O(1))."""
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
# smart_extract_content — 分层内容提取
# ---------------------------------------------------------------------------

_COMPRESS_TOOL_RESULT_SYSTEM: str = (
    "You are a precise summarizer. Compress the following tool result "
    "while preserving all key facts, data paths, error messages, and actionable output. "
    "Return ONLY the compressed text."
)


async def smart_extract_content(
    content: str,
    budget_tokens: int,
    *,
    model: str = "",
    tool_name: str = "",
    llm_client: LLMClient | None = None,
) -> str:
    """分层内容提取: 结构化 → LLM 总结 → 截断。

    Layer 0: skill 结果结构化提取 (零 LLM)
    Layer 0.5: JSON 结构化提取 (零 LLM)
    Layer 1: LLM 总结 (按需)
    Layer 2: 截断 (fallback)
    """
    # Layer 0: skill 结果结构化提取 (零 LLM)
    if tool_name == "execute_skill":
        try:
            parsed = _json.loads(content)
            if isinstance(parsed, dict):
                output_val = parsed.get("output")
                if isinstance(output_val, dict):
                    extracted = _build_digest(parsed, output_val)
                    if estimate_tokens_fast(extracted) <= budget_tokens:
                        return extracted
                    content = extracted
        except (ValueError, TypeError):
            pass

    if estimate_tokens_fast(content) <= budget_tokens:
        return content

    # Layer 0.5: JSON 结构化提取
    structured = _extract_structured(content)
    if structured and estimate_tokens_fast(structured) <= budget_tokens:
        return structured
    if structured:
        content = structured

    if estimate_tokens_fast(content) <= budget_tokens:
        return content

    # Layer 1: LLM 总结 (按需)
    client = llm_client
    if client is not None:
        try:
            summary = await client(
                [{"role": "user", "content": content}],
                system=_COMPRESS_TOOL_RESULT_SYSTEM,
                max_tokens=min(budget_tokens, 800),
            )
            return f"[summarized]\n{summary.strip()}"
        except Exception:
            logger.warning(
                "smart_extract_content LLM summarize failed, falling back to truncation"
            )

    # Layer 2: 截断 (fallback)
    return extract_key_content(content, budget_tokens, model=model)
