"""
Memento-S 调试日志模块

提供带明显视觉标志的调试日志输出，便于追踪：
- Agent 运行流程
- LLM 输入输出（自动截断）
- Tool 调用详情
- Skills 执行过程（pip安装、代码运行等）

使用方式：
    from utils.debug_logger import (
        log_agent_start, log_agent_end,
        log_llm_request, log_llm_response,
        log_tool_start, log_tool_end,
        log_skill_exec, log_sandbox_exec,
        log_pip_install, log_python_exec,
    )
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

from utils.logger import get_logger
from utils.token_utils import count_tokens_messages, count_tokens

logger = get_logger(__name__)

# ============================================================================
# 视觉标志常量
# ============================================================================

MARKER_AGENT = "🤖"
MARKER_LLM = "🧠"
MARKER_TOOL = "🔧"
MARKER_SKILL = "⚡"
MARKER_SANDBOX = "📦"
MARKER_PIP = "📥"
MARKER_PYTHON = "🐍"
MARKER_SUCCESS = "✅"
MARKER_ERROR = "❌"
MARKER_WARNING = "⚠️"
MARKER_INFO = "ℹ️"

SEPARATOR_THICK = "=" * 80
SEPARATOR_THIN = "-" * 80
SEPARATOR_DOTTED = "·" * 80

# ============================================================================
# 截断工具函数
# ============================================================================


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """截断文本，避免日志过长"""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + suffix


def truncate_dict(data: dict, max_str_length: int = 200) -> dict:
    """截断字典中的字符串值"""
    result = {}
    for key, value in data.items():
        if isinstance(value, str) and len(value) > max_str_length:
            result[key] = truncate_text(value, max_str_length)
        else:
            result[key] = value
    return result


def format_json(data: Any, indent: int = 2) -> str:
    """格式化 JSON，自动截断长字符串"""
    try:
        if isinstance(data, dict):
            data = truncate_dict(data)
        return json.dumps(data, ensure_ascii=False, indent=indent, default=str)
    except Exception:
        return str(data)


# ============================================================================
# Agent 流程日志
# ============================================================================


def log_agent_start(session_id: str, model: str, message: str) -> None:
    """记录 Agent 开始运行"""
    logger.debug("")
    logger.debug(f"{SEPARATOR_THICK}")
    logger.debug(f"{MARKER_AGENT} AGENT START")
    logger.debug(f"{SEPARATOR_THICK}")
    logger.debug(f"  Session ID: {session_id}")
    logger.debug(f"  Model: {model}")
    logger.debug(f"  Message: {truncate_text(message, 300)}")
    logger.debug(f"{SEPARATOR_THIN}")


def log_agent_end(session_id: str, duration: float, success: bool = True) -> None:
    """记录 Agent 运行结束"""
    marker = MARKER_SUCCESS if success else MARKER_ERROR
    logger.debug(f"{SEPARATOR_THIN}")
    logger.debug(f"{marker} AGENT END")
    logger.debug(f"  Session ID: {session_id}")
    logger.debug(f"  Duration: {duration:.2f}s")
    logger.debug(f"{SEPARATOR_THICK}")
    logger.debug("")


def log_agent_phase(phase_name: str, step_id: str = "", details: str = "") -> None:
    """记录 Agent 进入某个阶段"""
    logger.debug("")
    logger.debug(f"{MARKER_AGENT} PHASE: {phase_name}")
    if step_id:
        logger.debug(f"  Step: {step_id}")
    if details:
        logger.debug(f"  Details: {truncate_text(details, 200)}")
    logger.debug(f"{SEPARATOR_DOTTED}")


# ============================================================================
# LLM 日志
# ============================================================================


def log_llm_request(
    messages: list[dict] | None, tools: list[dict] | None = None, model: str = "gpt-4"
) -> None:
    """记录 LLM 请求（自动截断）"""
    logger.debug("")
    logger.debug(f"{MARKER_LLM} LLM REQUEST")
    logger.debug(f"{SEPARATOR_THIN}")

    # 计算并打印输入 token 数
    if messages:
        input_tokens = count_tokens_messages(messages, model)
        logger.debug(f"  Input Tokens: {input_tokens}")
        logger.debug(f"{SEPARATOR_DOTTED}")

    # 记录消息
    if not messages:
        logger.debug("  (no messages)")
    else:
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            truncated = truncate_text(content, 800)
            logger.debug(f"  [{i}] {role}:")
            for line in truncated.split("\n"):
                logger.debug(f"      {line}")

    # 记录 tools
    if tools:
        logger.debug(f"  Tools: {len(tools)} available")
        for tool in tools:
            name = (
                tool.get("function", {}).get("name", "unknown")
                if isinstance(tool, dict)
                else "unknown"
            )
            logger.debug(f"    - {name}")

    logger.debug(f"{SEPARATOR_DOTTED}")


def log_llm_response(response: Any, truncate_length: int = 1000) -> None:
    """记录 LLM 响应（自动截断）"""
    logger.debug("")
    logger.debug(f"{MARKER_LLM} LLM RESPONSE")
    logger.debug(f"{SEPARATOR_THIN}")

    # 提取关键信息
    finish_reason = getattr(response, "finish_reason", "unknown")
    content = getattr(response, "text", "") or ""
    tool_calls = getattr(response, "tool_calls", []) or []

    # 打印输出 token 数（从 usage 获取）
    usage = getattr(response, "usage", None)

    # 检查 usage 是否有实际的 token 数据（处理空字典 {} 的情况）
    has_real_usage = False
    if usage:
        if isinstance(usage, dict):
            # 检查字典是否有任何非空的 token 字段
            has_real_usage = any(
                usage.get(key) is not None and usage.get(key) != 0
                for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
            )
        else:
            # 对象类型，检查是否有这些属性且不为 None
            has_real_usage = any(
                getattr(usage, key, None) is not None
                for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
            )

    if has_real_usage:
        try:
            # 兼容对象和字典两种格式
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")
            else:
                # Usage 对象，使用 getattr
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

            logger.debug(
                f"  Output Tokens: {completion_tokens or 'N/A'} (Prompt: {prompt_tokens or 'N/A'}, Total: {total_tokens or 'N/A'})"
            )
        except Exception as e:
            logger.debug(f"  Output Tokens: (error reading usage: {e})")
    else:
        # 备用：手动计算输出token
        output_tokens = count_tokens(content)
        if usage == {}:
            logger.debug(
                f"  Output Tokens: {output_tokens} (estimated, API returned empty usage)"
            )
        else:
            logger.debug(f"  Output Tokens: {output_tokens} (estimated)")

    logger.debug(f"  Finish Reason: {finish_reason}")
    logger.debug(f"  Tool Calls: {len(tool_calls)}")

    if tool_calls:
        for i, tc in enumerate(tool_calls):
            name = getattr(tc, "name", "unknown")
            args = getattr(tc, "arguments", "{}")
            logger.debug(f"    [{i}] {name}")
            try:
                args_dict = json.loads(args) if isinstance(args, str) else args
                args_truncated = truncate_dict(args_dict)
                for key, val in args_truncated.items():
                    val_str = str(val)[:100]
                    logger.debug(f"        {key}: {val_str}")
            except Exception:
                logger.debug(f"        args: {truncate_text(str(args), 100)}")

    if content:
        logger.debug(f"  Content:")
        for line in truncate_text(content, truncate_length).split("\n"):
            logger.debug(f"      {line}")

    logger.debug(f"{SEPARATOR_DOTTED}")


def log_llm_stream_chunk(chunk: str, chunk_num: int) -> None:
    """记录 LLM 流式响应块（仅记录前几个和最后几个）"""
    if chunk_num <= 3 or chunk_num % 20 == 0:
        truncated = truncate_text(chunk, 200)
        logger.debug(f"{MARKER_LLM} STREAM CHUNK #{chunk_num}: {truncated}")


# ============================================================================
# Tool 调用日志
# ============================================================================


def log_tool_start(tool_name: str, args: dict, call_id: str = "") -> None:
    """记录 Tool 调用开始"""
    logger.debug("")
    logger.debug(f"{MARKER_TOOL} TOOL CALL START: {tool_name}")
    logger.debug(f"{SEPARATOR_THIN}")
    if call_id:
        logger.debug(f"  Call ID: {call_id}")

    # 截断参数显示
    args_truncated = truncate_dict(args)
    for key, value in args_truncated.items():
        value_str = str(value)
        if len(value_str) > 150:
            value_str = value_str[:150] + "..."
        logger.debug(f"  {key}: {value_str}")

    logger.debug(f"{SEPARATOR_DOTTED}")


def log_tool_end(
    tool_name: str, result: str, duration: float, success: bool = True
) -> None:
    """记录 Tool 调用结束"""
    marker = MARKER_SUCCESS if success else MARKER_ERROR
    logger.debug(f"{SEPARATOR_DOTTED}")
    logger.debug(f"{marker} TOOL CALL END: {tool_name}")
    logger.debug(f"  Duration: {duration:.2f}s")
    logger.debug(f"  Result Preview:")

    # 截断结果显示
    result_truncated = truncate_text(str(result), 800)
    for line in result_truncated.split("\n")[:20]:  # 最多显示20行
        logger.debug(f"    {line}")

    if len(result_truncated) > 800:
        logger.debug(f"    ... (truncated)")

    logger.debug(f"{SEPARATOR_THIN}")


# ============================================================================
# Skill 执行日志
# ============================================================================


def log_skill_exec(skill_name: str, query: str, phase: str = "start") -> None:
    """记录 Skill 执行过程"""
    if phase == "start":
        logger.debug("")
        logger.debug(f"{MARKER_SKILL} SKILL EXEC START: {skill_name}")
        logger.debug(f"{SEPARATOR_THIN}")
        logger.debug(f"  Query: {truncate_text(query, 300)}")
        logger.debug(f"{SEPARATOR_DOTTED}")
    elif phase == "end":
        logger.debug(f"{SEPARATOR_DOTTED}")
        logger.debug(f"{MARKER_SKILL} SKILL EXEC END: {skill_name}")
        logger.debug(f"{SEPARATOR_THIN}")


def log_sandbox_exec(command: str, cwd: str = "", env: dict | None = None) -> None:
    """记录沙箱执行命令"""
    logger.debug("")
    logger.debug(f"{MARKER_SANDBOX} SANDBOX EXEC")
    logger.debug(f"{SEPARATOR_THIN}")
    logger.debug(f"  Command: {command[:200]}")
    if cwd:
        logger.debug(f"  Working Dir: {cwd}")
    if env:
        env_keys = list(env.keys())[:5]  # 只显示前5个环境变量
        logger.debug(f"  Env Keys: {env_keys}")
    logger.debug(f"{SEPARATOR_DOTTED}")


# ============================================================================
# Python/Pip 日志
# ============================================================================


def log_pip_install(packages: list[str] | None, python_path: str) -> None:
    """记录 Pip 安装"""
    logger.debug("")
    logger.debug(f"{MARKER_PIP} PIP INSTALL")
    logger.debug(f"{SEPARATOR_THIN}")
    logger.debug(f"  Python: {python_path}")
    if not packages:
        logger.debug(f"  Packages: (none)")
    else:
        logger.debug(f"  Packages ({len(packages)}):")
        for pkg in packages[:10]:  # 最多显示10个包
            logger.debug(f"    - {pkg}")
        if len(packages) > 10:
            logger.debug(f"    ... and {len(packages) - 10} more")
    logger.debug(f"{SEPARATOR_DOTTED}")


def log_python_exec(code: str | None, script_path: str | None = None) -> None:
    """记录 Python 代码执行"""
    logger.debug("")
    logger.debug(f"{MARKER_PYTHON} PYTHON EXEC")
    logger.debug(f"{SEPARATOR_THIN}")

    if script_path:
        logger.debug(f"  Script: {script_path}")
    elif code:
        logger.debug(f"  Code:")
        # 显示代码的前30行
        code_str = str(code)
        lines = code_str.split("\n")[:30]
        for i, line in enumerate(lines, 1):
            logger.debug(f"    {i:3}: {line}")
        if len(code_str.split("\n")) > 30:
            logger.debug(f"    ... ({len(code_str.split(chr(10))) - 30} more lines)")
    else:
        logger.debug(f"  Code: (none)")

    logger.debug(f"{SEPARATOR_DOTTED}")


# ============================================================================
# 通用调试日志
# ============================================================================


def log_debug_marker(title: str, data: Any = None, level: str = "info") -> None:
    """通用调试标记日志"""
    log_func = getattr(logger, level, logger.info)
    log_func("")
    log_func(f"{SEPARATOR_THIN}")
    log_func(f"{MARKER_INFO} {title}")
    if data is not None:
        formatted = format_json(data)
        for line in formatted.split("\n")[:50]:  # 最多50行
            log_func(f"  {line}")
    log_func(f"{SEPARATOR_THIN}")


def log_error_context(error: Exception, context: str = "") -> None:
    """记录错误上下文"""
    logger.error("")
    logger.error(f"{SEPARATOR_THICK}")
    logger.error(f"{MARKER_ERROR} ERROR OCCURRED")
    logger.error(f"{SEPARATOR_THICK}")
    if context:
        logger.error(f"Context: {context}")
    logger.error(f"Exception: {type(error).__name__}: {error}")
    logger.error(f"{SEPARATOR_THICK}")


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 常量
    "MARKER_AGENT",
    "MARKER_LLM",
    "MARKER_TOOL",
    "MARKER_SKILL",
    "MARKER_SANDBOX",
    "MARKER_PIP",
    "MARKER_PYTHON",
    "MARKER_SUCCESS",
    "MARKER_ERROR",
    "MARKER_WARNING",
    "MARKER_INFO",
    "SEPARATOR_THICK",
    "SEPARATOR_THIN",
    "SEPARATOR_DOTTED",
    # 工具函数
    "truncate_text",
    "truncate_dict",
    "format_json",
    # Agent 日志
    "log_agent_start",
    "log_agent_end",
    "log_agent_phase",
    # LLM 日志
    "log_llm_request",
    "log_llm_response",
    "log_llm_stream_chunk",
    # Tool 日志
    "log_tool_start",
    "log_tool_end",
    # Skill 日志
    "log_skill_exec",
    "log_sandbox_exec",
    # Python/Pip 日志
    "log_pip_install",
    "log_python_exec",
    # 通用
    "log_debug_marker",
    "log_error_context",
]
