"""LLM 工具函数。

提供 raw tool call token 检测与清理。
覆盖主流模型格式: KIMI / DeepSeek / Qwen / GPT / Mistral / MiniMax / 通用 JSON。

注意: chat_completions / chat_completions_async 已移至 llm_client.py,
      避免 utils ↔ llm_client 循环引用。
"""
from __future__ import annotations

import re

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Detection regex — 覆盖所有主流模型 raw token 格式 ─────────────

# 通用控制 token: <|xxx|> — 覆盖 DeepSeek ▁ (U+2581)
_RE_CONTROL_TOKEN = re.compile(r"<\|[a-zA-Z_0-9\u2581]+\|>")

# GPT function tag: <function=name> 或 </function>
_RE_FUNCTION_TAG = re.compile(r"</?function(?:=[a-zA-Z_]\w*)?>")

# Qwen / ChatGLM: <tool_call> 或 </tool_call>  (angle brackets)
_RE_TOOL_CALL_TAG = re.compile(r"</?tool_call>")

# Bracket-style tool call tags:  [TOOL_CALL] [/TOOL_CALL] [TOOL_CALLS] etc.
# Covers MiniMax M2.7, Mistral, and other bracket-delimited formats
_RE_BRACKET_TOOL_TAG = re.compile(r"\[/?TOOL_CALLS?\]", re.IGNORECASE)

# General XML tool call tag (detection + per-tag sanitization):
# Covers all XML-like tool call variants across models, with optional namespace prefix.
#   <tool_call>, </tool_call>, <minimax:tool_call>, <function_calls>, </function_calls>,
#   <invoke name="...">, </invoke>, <parameter=name>, </parameter>,
#   <arg name="...">, </arg>, <function=name>, </function>,
#   <execute_skill>, <skill_name> (DeepSeek)
_RE_XML_TOOL_TAG = re.compile(
    r"</?(?:\w+:)?(?:tool_calls?|function_calls?|function|invoke|parameters?|arg|execute_skill|skill_name)\b[^>]*>",
    re.IGNORECASE,
)

# KIMI-style function reference: functions.xxx:N
_RE_KIMI_FUNC_REF = re.compile(r"functions\.[a-zA-Z_]\w*:\d+")

# Mistral: [TOOL_CALLS] followed by JSON array
_RE_MISTRAL_TOOL = re.compile(r"\[TOOL_CALLS\]\s*\[.*?\]", re.DOTALL)

# Qwen legacy: ✿FUNCTION✿...✿RESULT✿
_RE_QWEN_LEGACY = re.compile(r"✿FUNCTION✿.*?(?:✿RESULT✿|$)", re.DOTALL)

# JSON tool call structure (detection): covers OpenAI / StepFun / Anthropic variants
#   {"name": "xxx", "arguments": ...}
#   {"tool": "xxx", "parameters": ...}
#   {"function": "xxx", "arguments": ...}
#   {"tool_name": "xxx", "tool_input": ...}
_RE_TOOL_CALL_JSON = re.compile(
    r'\{\s*"(?:name|tool|function|tool_name)"\s*:\s*"[a-zA-Z_]\w*"\s*,\s*'
    r'"(?:arguments|parameters|tool_input|input)"\s*:'
)

# JSON tool call block (sanitization): match and remove entire JSON object
# Supports up to two levels of nested braces in the arguments/parameters value
_RE_TOOL_CALL_JSON_BLOCK = re.compile(
    r'\{\s*"(?:name|tool|function|tool_name)"\s*:\s*"[^"]*"\s*,'
    r'[^{}]*'
    r'(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*'
    r'\}',
    re.DOTALL,
)

# Full tool call section block (KIMI / DeepSeek style)
_RE_FULL_TOOL_BLOCK = re.compile(
    r"<\|tool_calls_section_begin\|>.*?(?:<\|tool_calls_section_end\|>|$)",
    re.DOTALL,
)

# General XML tool call full block (sanitization):
# Matches opening tag with optional namespace → content → matching close tag or EOF.
# Covers: <tool_call>...</tool_call>, <minimax:tool_call>...</minimax:tool_call>,
#          <function_calls>...</function_calls>, <execute_skill>...</execute_skill>, etc.
_RE_XML_TOOL_BLOCK = re.compile(
    r"<(?:\w+:)?(?:tool_calls?|function_calls?|execute_skill)\b[^>]*>"
    r".*?"
    r"(?:</(?:\w+:)?(?:tool_calls?|function_calls?|execute_skill)\s*>|$)",
    re.DOTALL | re.IGNORECASE,
)

# Bracket-style tool call full block:
#   [TOOL_CALL]...[/TOOL_CALL]  (MiniMax M2.7)
#   [TOOL_CALLS]...[/TOOL_CALLS]
_RE_BRACKET_TOOL_BLOCK = re.compile(
    r"\[TOOL_CALLS?\].*?(?:\[/TOOL_CALLS?\]|$)",
    re.DOTALL | re.IGNORECASE,
)

# KIMI single tool call: functions.xxx:N ... {json} ...
_RE_KIMI_FUNC_CALL = re.compile(
    r"functions\.[a-zA-Z_]\w*:\d+\s*"
    r"(?:<\|tool_call_argument_begin\|>)?\s*"
    r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*"
    r"(?:<\|tool_call_argument_end\|>)?",
    re.DOTALL,
)


# ── Detection ──────────────────────────────────────────────────────


def looks_like_tool_call_text(content: str) -> bool:
    """检测 LLM 文本输出是否像未成功解析的 tool call。

    覆盖所有主流模型的控制 token / raw tool call 格式。
    """
    if not content:
        return False
    return bool(
        _RE_CONTROL_TOKEN.search(content)
        or _RE_FUNCTION_TAG.search(content)
        or _RE_TOOL_CALL_TAG.search(content)
        or _RE_BRACKET_TOOL_TAG.search(content)
        or _RE_XML_TOOL_TAG.search(content)
        or _RE_TOOL_CALL_JSON.search(content)
        or _RE_KIMI_FUNC_REF.search(content)
        or _RE_MISTRAL_TOOL.search(content)
        or _RE_QWEN_LEGACY.search(content)
    )


# ── Content sanitization ──────────────────────────────────────────


def sanitize_content(content: str) -> str:
    """从文本中移除所有模型的控制 token 和 raw tool call 残留。

    清除顺序: 完整块 → 单条格式 → 标签 → 通用 token。
    注意: 不做 strip()，保留有意义的空白字符（换行等），
    避免流式 chunk 中的换行被吞掉导致输出格式错乱。
    需要 trim 的调用方应自行 .strip()。
    """
    if not content:
        return content
    result = _RE_FULL_TOOL_BLOCK.sub("", content)
    result = _RE_XML_TOOL_BLOCK.sub("", result)
    result = _RE_BRACKET_TOOL_BLOCK.sub("", result)
    result = _RE_TOOL_CALL_JSON_BLOCK.sub("", result)
    result = _RE_MISTRAL_TOOL.sub("", result)
    result = _RE_QWEN_LEGACY.sub("", result)
    result = _RE_KIMI_FUNC_CALL.sub("", result)
    result = _RE_FUNCTION_TAG.sub("", result)
    result = _RE_TOOL_CALL_TAG.sub("", result)
    result = _RE_BRACKET_TOOL_TAG.sub("", result)
    result = _RE_XML_TOOL_TAG.sub("", result)
    result = _RE_CONTROL_TOKEN.sub("", result)
    return result


strip_control_tokens = sanitize_content
