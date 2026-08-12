# SPDX-License-Identifier: Apache-2.0
"""
异步 LLM 客户端 — 统一调用接口。

特性：
- 基于 litellm 的多提供商支持
- 自动重试机制（指数退避）
- 超时控制
- 熔断保护（Circuit Breaker）
- 统一响应格式
- Raw token 自愈层（检测并修复模型 raw tool call token 泄漏，上层无感知）
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator

os.environ["LITELLM_LOG"] = "WARNING"

import litellm
from litellm import acompletion

from middleware.config.config_manager import ConfigManager
from utils.logger import get_logger
from utils.debug_logger import log_llm_request, log_llm_response
from utils.token_utils import count_tokens_messages

from .exceptions import (
    LLMException,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMConnectionError,
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMContextWindowError,
)
from .schema import (
    FINISH_TOOL_CALLS,
    LLMResponse,
    LLMStreamChunk,
    ToolCall,
    Message,
)
from .utils import (
    looks_like_tool_call_text,
    sanitize_content,
)
from .circuit import CircuitBreaker, CircuitBreakerConfig
from .retry import RetryConfig, parse_llm_exception, retry_with_backoff

logger = get_logger()

MIN_OUTPUT_TOKENS: int = 256


@dataclass
class RawTokenConfig:
    """Raw token 自愈配置。"""

    retry_on_raw_tokens: bool = True
    max_raw_token_retries: int = 1


# ── Module-level helpers ───────────────────────────────────────────


async def _run_acompletion(completion_kwargs: dict[str, Any]) -> Any:
    """Module-level async wrapper for litellm acompletion (避免嵌套 def)。"""
    clean_kwargs = {k: v for k, v in completion_kwargs.items() if v is not None}
    return await acompletion(**clean_kwargs)


def _try_parse_json_tool_call(text: str) -> ToolCall | None:
    """Try parsing a JSON string as a tool call object."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    name = (
        obj.get("name")
        or obj.get("tool")
        or obj.get("function")
        or obj.get("tool_name")
        or ""
    )
    if not name:
        return None
    args = (
        obj.get("arguments")
        or obj.get("parameters")
        or obj.get("input")
        or obj.get("tool_input")
        or {}
    )
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = {}
    return ToolCall(
        id=obj.get("id") or f"tc_{uuid.uuid4().hex[:12]}",
        name=name,
        arguments=args if isinstance(args, dict) else {},
    )


def _try_parse_json_tool_call_at(content: str, start: int) -> ToolCall | None:
    """Extract a balanced JSON object starting at *start* and try parsing it."""
    depth = 0
    for i in range(start, len(content)):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                return _try_parse_json_tool_call(content[start : i + 1])
    return None


def _sanitize_response(response: LLMResponse) -> LLMResponse:
    """检测 raw tokens 并清理 content（不做提取）。"""
    if response.tool_calls or not response.content:
        return response
    if not looks_like_tool_call_text(response.content):
        return response

    cleaned = sanitize_content(response.content).strip()
    return LLMResponse(
        content=cleaned or None,
        tool_calls=[],
        usage=response.usage,
        model=response.model,
        finish_reason=response.finish_reason,
        raw_response=response.raw_response,
    )


# ── LLM Client ─────────────────────────────────────────────────────


class LLMClient:
    """
    统一的 LLM 异步客户端。

    用法::
        client = LLMClient()

        # 非流式调用
        response = await client.async_chat(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[...],
        )

        # 流式调用
        async for chunk in client.async_stream_chat(
            messages=[{"role": "user", "content": "Hello"}],
        ):
            print(chunk.delta_content)
    """

    def __init__(
        self,
        config_manager: ConfigManager | None = None,
        retry_config: RetryConfig | None = None,
        circuit_config: CircuitBreakerConfig | None = None,
        raw_token_config: RawTokenConfig | None = None,
    ):
        # 使用传入的 config_manager 或 g_config 全局单例
        from middleware.config import g_config

        self.config_manager = config_manager or g_config
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker(circuit_config)
        self.raw_token_config = raw_token_config or RawTokenConfig()

        self._load_config()

    def _load_config(self):
        """从 ConfigManager 加载 LLM 配置。

        始终从磁盘重新加载以获取最新配置，确保模型切换生效。
        context_window 优先从 litellm 自动检测，检测不到则用 profile 默认值。
        """
        try:
            # 优先使用全局 g_config 以获取最新配置
            from middleware.config import g_config

            if g_config.is_loaded():
                config = g_config
            else:
                config = self.config_manager.load()

            llm_config = config.llm
            profile = llm_config.current_profile

            if profile is None:
                raise ValueError(
                    f"No active LLM profile found. Active profile: '{llm_config.active_profile}'. "
                    f"Available profiles: {list(llm_config.profiles.keys())}"
                )

            self.model = profile.model or ""
            self.api_key = profile.api_key
            self.base_url = profile.base_url
            self.litellm_provider = getattr(profile, "litellm_provider", "")

            # 源头守卫：确保从 profile 取出的值不是 None
            # 兜底值与 LLMProfile schema 默认值一致
            _detected_cw = self._detect_context_window(profile)
            _profile_cw = profile.context_window if isinstance(profile.context_window, int) else 0
            effective_cw = min(_detected_cw, _profile_cw) if _profile_cw > 0 else _detected_cw

            # 写入 profile 和实例变量
            profile.context_window = effective_cw
            self.context_window = effective_cw
            self.max_tokens = profile.max_tokens if isinstance(profile.max_tokens, int) else 4096
            profile.max_tokens = self.max_tokens  # 回写，保证一致性
            self.temperature = profile.temperature
            self.timeout = profile.timeout
            self.extra_headers = profile.extra_headers
            self.extra_body = profile.extra_body

            logger.info(
                f"LLM client initialized with model: {self.model}, "
                f"context_window: {self.context_window}, max_tokens: {self.max_tokens}"
            )
        except Exception as e:
            logger.error(f"Failed to load LLM config: {e}")
            raise

    def reload_config(self) -> None:
        """重新加载配置，用于模型切换或删除后刷新客户端配置。

        调用此方法会从 ConfigManager 重新加载最新的 LLM 配置，
        更新所有实例变量（model, api_key, base_url 等）。
        """
        logger.info("Reloading LLM configuration...")
        self._load_config()
        logger.info(f"LLM configuration reloaded: model={self.model}")

    @staticmethod
    def _detect_context_window(profile) -> int:
        """尝试通过 litellm 自动检测模型 context window，失败则用 profile 默认值（保底 100000）。"""
        if not profile.model:
            return profile.context_window if isinstance(profile.context_window, int) else 100000
        candidates = [profile.model]
        model_name = profile.model.split("/", 1)[-1] if "/" in profile.model else ""
        if model_name and model_name != profile.model:
            candidates.append(model_name)

        for candidate in candidates:
            try:
                info = litellm.get_model_info(candidate)
                detected = info.get("max_input_tokens") or info.get("max_tokens")
                if detected and isinstance(detected, int) and detected > 0:
                    logger.info(
                        "Auto-detected context_window={} for model {}",
                        detected,
                        candidate,
                    )
                    return detected
            except Exception:
                continue
        _fallback = profile.context_window if isinstance(profile.context_window, int) else 100000
        return _fallback

    @staticmethod
    def _is_known_model(model: str) -> bool:
        """检查 litellm 注册表是否认识该模型。"""
        try:
            litellm.get_model_info(model)
            return True
        except Exception:
            return False

    def _build_model_string(self) -> str:
        """构建 litellm 模型字符串。

        三层策略：
        1. litellm_provider 显式配置 → "{provider}/{model}"
        2. 代理类 URL 特殊处理（openrouter）→ 添加代理前缀
        3. litellm 注册表已知模型 → 原样传递
        4. 未知模型 + 有 base_url → "openai/{model}"
        5. 无 base_url → 原样传递
        """
        model = self.model or ""

        provider = getattr(self, "litellm_provider", "")
        if provider:
            prefix = f"{provider}/"
            if model.startswith(prefix):
                return model
            return f"{provider}/{model}"

        if self.base_url:
            url = self.base_url.lower()
            if "openrouter" in url:
                if not model.startswith("openrouter/"):
                    return f"openrouter/{model}"
                return model

        if self._is_known_model(model):
            return model

        if self.base_url:
            if "/" not in model:
                return f"openai/{model}"
            return model

        return model

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """确保 system message 只出现在第一位。

        非首位 system 消息转为 user 消息以兼容所有模型。
        同时兼容历史数据里的 content=None。
        """
        if not messages:
            return messages

        def _normalize_tool_calls(tool_calls: Any) -> Any:
            if not isinstance(tool_calls, list):
                return tool_calls

            fixed_calls: list[dict[str, Any]] = []
            for tc in tool_calls:
                # ToolCall dataclass对象 → 转为 dict
                if hasattr(tc, "as_dict"):
                    tc = tc.as_dict()
                elif hasattr(tc, "__dataclass_fields__") and not isinstance(tc, dict):
                    tc = {
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": getattr(tc, "name", ""),
                            "arguments": json.dumps(getattr(tc, "arguments", {}))
                            if isinstance(getattr(tc, "arguments", None), dict)
                            else str(getattr(tc, "arguments", "{}")),
                        },
                    }
                if not isinstance(tc, dict):
                    continue

                function_obj = tc.get("function")
                if not isinstance(function_obj, dict):
                    fixed_calls.append(tc)
                    continue

                fixed_function = dict(function_obj)
                args = fixed_function.get("arguments", "{}")

                if isinstance(args, dict):
                    fixed_function["arguments"] = json.dumps(args, ensure_ascii=False)
                elif args is None:
                    fixed_function["arguments"] = "{}"
                elif isinstance(args, str):
                    s = args.strip()
                    if not s:
                        fixed_function["arguments"] = "{}"
                    else:
                        try:
                            parsed = json.loads(s)
                            if isinstance(parsed, dict):
                                fixed_function["arguments"] = json.dumps(
                                    parsed, ensure_ascii=False
                                )
                            else:
                                fixed_function["arguments"] = "{}"
                        except Exception:
                            fixed_function["arguments"] = "{}"
                else:
                    fixed_function["arguments"] = "{}"

                fixed_tc = dict(tc)
                fixed_tc["function"] = fixed_function
                fixed_calls.append(fixed_tc)

            return fixed_calls

        normalized = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if content is None:
                content = ""

            if isinstance(content, list):
                text_parts = [
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                ]
                content = " ".join(text_parts)

            normalized_msg = {**msg, "content": content}
            if "tool_calls" in normalized_msg:
                normalized_msg["tool_calls"] = _normalize_tool_calls(
                    normalized_msg.get("tool_calls")
                )

            if i > 0 and normalized_msg.get("role") == "system":
                normalized.append({"role": "user", "content": f"[System]: {content}"})
            else:
                normalized.append(normalized_msg)
        return normalized

    @staticmethod
    def _fix_empty_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fix empty / orphaned messages that strict APIs reject.

        Handles two cases:
        1. Assistant message with empty content but valid tool_calls →
           set content to None (Gemini / DeepSeek require non-empty-string).
        2. Assistant message with no content AND no tool_calls, followed by
           tool-role messages → the original tool_calls were lost (e.g. raw
           XML tool calls that got sanitized).  Drop the empty assistant and
           its trailing tool messages to avoid orphaned tool messages, which
           cause "Message has tool role, but there was no previous assistant
           message with a tool call!" from the server.
        """
        fixed: list[dict[str, Any]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")
            content = msg.get("content")
            has_tool_calls = bool(msg.get("tool_calls"))

            if role == "assistant" and not content and not has_tool_calls:
                # tool_calls were already stripped by _normalize_messages.
                # Drop the empty assistant and its trailing tool messages —
                # they are orphaned and would cause server TemplateError.
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    j += 1
                i = j
                continue

            if role == "assistant" and not content and has_tool_calls:
                msg = dict(msg)
                msg["content"] = None

            fixed.append(msg)
            i += 1
        return fixed

    def _build_completion_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        """构建 litellm 调用参数。"""
        model_str = self._build_model_string()
        messages = self._normalize_messages(messages)
        messages = self._fix_empty_messages(messages)

        if not tools:
            converted = []
            for msg in messages:
                if msg.get("role") == "tool":
                    content = msg.get("content", "")
                    if content:
                        converted.append(
                            {
                                "role": "user",
                                "content": f"[Tool Result]: {content}",
                            }
                        )
                else:
                    stripped = {k: v for k, v in msg.items() if k != "tool_calls"}
                    if stripped.get("role") == "assistant" and not stripped.get(
                        "content"
                    ):
                        continue
                    converted.append(stripped)
            messages = converted

        input_tokens = count_tokens_messages(messages, model=self.model, tools=tools)
        _context_window = max(0, self.context_window) if isinstance(self.context_window, int) else 0
        _configured_max = self.max_tokens if isinstance(self.max_tokens, int) else (_context_window // 2)
        effective_max = min(_configured_max, _context_window - input_tokens)
        effective_max = max(effective_max, MIN_OUTPUT_TOKENS)

        if input_tokens >= (self.context_window or 0):
            logger.warning(
                "Input tokens ({}) exceed context window ({}), request will likely fail",
                input_tokens,
                self.context_window,
            )

        if effective_max < (self.max_tokens or 0):
            logger.info(
                "max_tokens capped: {} -> {} (context_window={}, input={})",
                self.max_tokens,
                effective_max,
                self.context_window,
                input_tokens,
            )

        kwargs: dict[str, Any] = {
            "model": model_str,
            "messages": messages,
            "max_tokens": effective_max,
            "max_completion_tokens": effective_max,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "drop_params": True,
            "stream": stream,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if tools:
            kwargs["tools"] = tools

        if "max_tokens" in extra and isinstance(extra["max_tokens"], int):
            extra["max_tokens"] = min(extra["max_tokens"], effective_max)
        kwargs.update(extra)

        # 最终清理在 _run_acompletion 中进行
        return kwargs

    def _parse_error(self, error: Exception, model: str) -> LLMException:
        """解析异常为统一的 LLMException（委托给 retry 模块）。"""
        return parse_llm_exception(error, model)

    async def _call_with_retry(
        self,
        func,
        *args,
        **kwargs,
    ) -> Any:
        """带重试机制的调用（委托给 retry 模块）。"""
        return await retry_with_backoff(
            func,
            *args,
            config=self.retry_config,
            model=self.model,
            **kwargs,
        )

    def _parse_tool_calls(self, raw_tool_calls: list[Any] | None) -> list[ToolCall]:
        """解析 tool calls（非流式），包含 JSON 修复兜底。"""
        if not raw_tool_calls:
            return []

        result: list[ToolCall] = []
        for tc in raw_tool_calls:
            try:
                func = (
                    tc.get("function")
                    if isinstance(tc, dict)
                    else getattr(tc, "function", None)
                )
                if not func:
                    continue

                args_raw = (
                    func.get("arguments")
                    if isinstance(func, dict)
                    else getattr(func, "arguments", "")
                )

                if isinstance(args_raw, dict):
                    arguments = args_raw
                elif isinstance(args_raw, str) and args_raw.strip():
                    arguments = self._parse_tool_args_with_repair(args_raw)
                else:
                    arguments = {}

                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                if not tc_id:
                    tc_id = f"tc_{uuid.uuid4().hex[:12]}"
                func_name = (
                    func.get("name")
                    if isinstance(func, dict)
                    else getattr(func, "name", "")
                )

                result.append(
                    ToolCall(id=tc_id, name=func_name or "", arguments=arguments)
                )
            except Exception as exc:
                logger.warning(f"Failed to parse tool_call: {exc}")

        return result

    @staticmethod
    def _parse_tool_args_with_repair(args_raw: str) -> dict[str, Any]:
        """Parse tool call arguments with a lightweight JSON repair fallback."""
        try:
            return json.loads(args_raw)
        except json.JSONDecodeError:
            repaired = args_raw.strip()

            if not repaired.startswith("{"):
                start = repaired.find("{")
                if start != -1:
                    repaired = repaired[start:]

            repaired = re.sub(r"\s+$", "", repaired)
            repaired = repaired.replace("\r", "")

            if not repaired.endswith("}"):
                repaired = repaired + "}"

            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                repaired = re.sub(
                    r"(?<=[\[{,:])\s*'([^']*?)'\s*(?=[,\]}\:])",
                    r'"\1"',
                    repaired,
                )
                repaired = re.sub(r",\s*\}", "}", repaired)
                return json.loads(repaired)

    def _build_response_from_raw(self, raw: Any) -> LLMResponse:
        """从 litellm 原始响应构建 LLMResponse。"""
        content: str | None = None
        tc: list[ToolCall] = []
        usage = {}
        fr = None
        if hasattr(raw, "choices") and raw.choices:
            message = raw.choices[0].message
            content = getattr(message, "content", None)

            # Handle list-type content (Anthropic-style content blocks not fully converted)
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts) if text_parts else None

            tc = self._parse_tool_calls(getattr(message, "tool_calls", None))
            fr = getattr(raw.choices[0], "finish_reason", None)

            # Fallback: try extracting tool calls from non-standard locations
            #   - finish_reason says tool_calls/tool_use but litellm parsed none
            #   - content contains raw tool call tokens (DeepSeek XML, Qwen, etc.)
            if not tc and (
                fr in ("tool_calls", "tool_use")
                or (content and looks_like_tool_call_text(content))
            ):
                fallback_tc = self._fallback_extract_tool_calls(message, raw)
                if fallback_tc:
                    tc = fallback_tc
                    logger.info(
                        "Recovered {} tool calls via fallback extraction", len(tc)
                    )
                elif fr in ("tool_calls", "tool_use"):
                    logger.warning(
                        "finish_reason={} but no tool_calls parsed. "
                        "message.content type={}, message.tool_calls type={}, "
                        "message attrs={}",
                        fr,
                        type(getattr(message, "content", None)).__name__,
                        type(getattr(message, "tool_calls", None)).__name__,
                        [a for a in dir(message) if not a.startswith("_")],
                    )

        if hasattr(raw, "usage"):
            usage = raw.usage
        return LLMResponse(
            content=content,
            tool_calls=tc,
            usage=usage,
            model=self.model,
            finish_reason=fr,
            raw_response=raw,
        )

    def _fallback_extract_tool_calls(self, message: Any, raw: Any) -> list[ToolCall]:
        """Try extracting tool calls from non-standard locations in the response.

        Covers: Anthropic content blocks not converted by litellm,
        raw content list with tool_use entries, hidden original response.
        """
        result: list[ToolCall] = []

        # Strategy 1: message.content is a list with tool_use blocks
        raw_content = getattr(message, "content", None)
        if isinstance(raw_content, list):
            for block in raw_content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tc_input = block.get("input", {})
                    if isinstance(tc_input, str):
                        try:
                            tc_input = json.loads(tc_input)
                        except (json.JSONDecodeError, TypeError):
                            tc_input = {}
                    result.append(
                        ToolCall(
                            id=block.get("id") or f"tc_{uuid.uuid4().hex[:12]}",
                            name=block.get("name", ""),
                            arguments=tc_input if isinstance(tc_input, dict) else {},
                        )
                    )
            if result:
                return result

        # Strategy 2: check _hidden_params for original Anthropic response
        hidden = getattr(raw, "_hidden_params", None)
        if isinstance(hidden, dict):
            original = hidden.get("original_response")
            if original:
                try:
                    data = (
                        json.loads(original) if isinstance(original, str) else original
                    )
                    if isinstance(data, dict):
                        for block in data.get("content") or []:
                            if (
                                isinstance(block, dict)
                                and block.get("type") == "tool_use"
                            ):
                                tc_input = block.get("input", {})
                                if isinstance(tc_input, str):
                                    try:
                                        tc_input = json.loads(tc_input)
                                    except (json.JSONDecodeError, TypeError):
                                        tc_input = {}
                                result.append(
                                    ToolCall(
                                        id=block.get("id")
                                        or f"tc_{uuid.uuid4().hex[:12]}",
                                        name=block.get("name", ""),
                                        arguments=tc_input
                                        if isinstance(tc_input, dict)
                                        else {},
                                    )
                                )
                except Exception as exc:
                    logger.debug(
                        "Fallback extraction from hidden_params failed: {}", exc
                    )

        # Strategy 3: parse raw tool call text from content string
        content_str = getattr(message, "content", None)
        if isinstance(content_str, str) and looks_like_tool_call_text(content_str):
            parsed = self._parse_raw_content_tool_calls(content_str)
            if parsed:
                logger.info(
                    "Recovered {} tool calls from raw content text (Strategy 3)",
                    len(parsed),
                )
                return parsed

        return result

    @staticmethod
    def _parse_raw_content_tool_calls(content: str) -> list[ToolCall]:
        """Parse tool calls from raw content text across various model formats.

        Covers:
        - <execute_skill><skill_name>...<parameters>...</execute_skill>  (DeepSeek)
        - <tool_call>{JSON}</tool_call>  (Qwen / ChatGLM)
        - <function_calls><invoke name="..."><parameter>...</invoke>  (MiniMax / Anthropic-style XML)
        - [TOOL_CALL]{JSON}[/TOOL_CALL]  (MiniMax bracket)
        - Standalone JSON: {"name":"...","arguments":...} / {"tool":"...","parameters":...}
        - KIMI: functions.xxx:N {JSON}
        """
        if not content:
            return []

        result: list[ToolCall] = []

        # ── Format 1: <execute_skill> XML (DeepSeek) ──
        for m in re.finditer(
            r"<execute_skill>(.*?)</execute_skill>", content, re.DOTALL | re.IGNORECASE
        ):
            block = m.group(1)
            name_m = re.search(
                r"<skill_name>(.*?)</skill_name>", block, re.DOTALL | re.IGNORECASE
            )
            if not name_m:
                continue
            name = name_m.group(1).strip()
            params_m = re.search(
                r"<parameters>(.*?)</parameters>", block, re.DOTALL | re.IGNORECASE
            )
            args: dict[str, Any] = {}
            if params_m:
                for p in re.finditer(
                    r"<(\w+)>(.*?)</\1>", params_m.group(1), re.DOTALL
                ):
                    val = p.group(2).strip()
                    try:
                        args[p.group(1)] = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        args[p.group(1)] = val
            result.append(
                ToolCall(
                    id=f"tc_{uuid.uuid4().hex[:12]}",
                    name=name,
                    arguments=args,
                )
            )
        if result:
            return result

        # ── Format 2: <tool_call>{JSON}</tool_call> (Qwen / ChatGLM) ──
        for m in re.finditer(
            r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL | re.IGNORECASE
        ):
            tc = _try_parse_json_tool_call(m.group(1))
            if tc:
                result.append(tc)
        if result:
            return result

        # ── Format 3: <invoke name="..."><parameter name="...">value</parameter></invoke> ──
        for m in re.finditer(
            r'<invoke\s+name=["\']?(\w+)["\']?\s*>(.*?)</invoke>',
            content,
            re.DOTALL | re.IGNORECASE,
        ):
            name = m.group(1)
            args = {}
            for p in re.finditer(
                r'<(?:parameter|arg)\s+name=["\']?(\w+)["\']?\s*>(.*?)</(?:parameter|arg)>',
                m.group(2),
                re.DOTALL | re.IGNORECASE,
            ):
                val = p.group(2).strip()
                try:
                    args[p.group(1)] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    args[p.group(1)] = val
            if name:
                result.append(
                    ToolCall(
                        id=f"tc_{uuid.uuid4().hex[:12]}",
                        name=name,
                        arguments=args,
                    )
                )
        if result:
            return result

        # ── Format 4: [TOOL_CALL]{JSON}[/TOOL_CALL] (MiniMax bracket) ──
        for m in re.finditer(
            r"\[TOOL_CALLS?\]\s*(.*?)\s*\[/TOOL_CALLS?\]",
            content,
            re.DOTALL | re.IGNORECASE,
        ):
            tc = _try_parse_json_tool_call(m.group(1))
            if tc:
                result.append(tc)
        if result:
            return result

        # ── Format 5: standalone JSON tool call objects ──
        for m in re.finditer(
            r'\{\s*"(?:name|tool|function|tool_name)"\s*:\s*"[^"]+?"',
            content,
        ):
            tc = _try_parse_json_tool_call_at(content, m.start())
            if tc:
                result.append(tc)
        if result:
            return result

        # ── Format 6: KIMI functions.xxx:N {JSON} ──
        for m in re.finditer(
            r"functions\.(\w+):\d+\s*(?:<\|tool_call_argument_begin\|>)?\s*"
            r"(\{.*?\})\s*(?:<\|tool_call_argument_end\|>)?",
            content,
            re.DOTALL,
        ):
            name = m.group(1)
            try:
                args = json.loads(m.group(2))
                if isinstance(args, dict) and name:
                    result.append(
                        ToolCall(
                            id=f"tc_{uuid.uuid4().hex[:12]}",
                            name=name,
                            arguments=args,
                        )
                    )
            except (json.JSONDecodeError, TypeError):
                pass

        return result

    async def async_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """异步非流式调用 LLM。"""
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        log_llm_request(full_messages, tools, self.model)

        completion_kwargs = self._build_completion_kwargs(
            full_messages, tools=tools, **kwargs
        )

        try:
            raw_response = await self.circuit_breaker.call(
                lambda: self._call_with_retry(_run_acompletion, completion_kwargs)
            )
        except LLMException:
            raise
        except Exception as e:
            raise self._parse_error(e, self.model)

        response_obj = self._build_response_from_raw(raw_response)
        sanitized = _sanitize_response(response_obj)

        raw_token_retries = 0
        while (
            self.raw_token_config.retry_on_raw_tokens
            and not sanitized.tool_calls
            and (
                looks_like_tool_call_text(response_obj.content or "")
                or (
                    response_obj.finish_reason in ("tool_calls", "tool_use")
                    and not response_obj.content
                )
            )
            and raw_token_retries < self.raw_token_config.max_raw_token_retries
        ):
            raw_token_retries += 1
            logger.warning(
                "Tool call recovery retry (attempt {}/{}): finish_reason={}, "
                "has_content={}, has_tool_calls={}",
                raw_token_retries,
                self.raw_token_config.max_raw_token_retries,
                response_obj.finish_reason,
                bool(response_obj.content),
                bool(response_obj.tool_calls),
            )
            try:
                raw_response = await self.circuit_breaker.call(
                    lambda: self._call_with_retry(_run_acompletion, completion_kwargs)
                )
                response_obj = self._build_response_from_raw(raw_response)
                sanitized = _sanitize_response(response_obj)
            except Exception as e:
                logger.warning("Raw token retry failed: {}", e)
                break

        log_llm_response(sanitized, model=self.model)
        return sanitized

    async def async_stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """异步流式调用 LLM。"""
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        completion_kwargs = self._build_completion_kwargs(
            full_messages, tools=tools, stream=True, **kwargs
        )

        try:
            logger.debug(f"[LLM Stream] Starting stream call with model: {self.model}")
            log_llm_request(full_messages, tools)

            raw_stream = await acompletion(**completion_kwargs)
            chunk_count = 0
            total_chars = 0
            accumulated_content = ""
            _last_usage = {}
            _last_finish_reason: str | None = None
            _raw_tokens_detected = False

            _tc_acc: dict[int, dict[str, str]] = {}

            async for chunk in raw_stream:
                usage = _last_usage
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = chunk.usage
                    if hasattr(usage, "model_dump"):
                        _last_usage = usage.model_dump()
                    elif hasattr(usage, "dict"):
                        _last_usage = usage.dict()
                    else:
                        _last_usage = dict(usage)

                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason
                if finish_reason:
                    _last_finish_reason = finish_reason

                tool_calls_delta = getattr(delta, "tool_calls", None)
                if tool_calls_delta:
                    for tc_delta in tool_calls_delta:
                        idx = getattr(tc_delta, "index", 0) or 0
                        if idx not in _tc_acc:
                            _tc_acc[idx] = {"id": "", "name": "", "args_str": ""}
                        entry = _tc_acc[idx]
                        tc_id = getattr(tc_delta, "id", None)
                        if tc_id:
                            entry["id"] = tc_id
                        func = getattr(tc_delta, "function", None)
                        if func:
                            fn_name = getattr(func, "name", None)
                            if fn_name:
                                entry["name"] = fn_name
                            fn_args = getattr(func, "arguments", None)
                            if fn_args is not None:
                                entry["args_str"] += fn_args

                content = getattr(delta, "content", None)
                if content:
                    chunk_count += 1
                    total_chars += len(content)
                    accumulated_content += content

                if not finish_reason:
                    if content and not _raw_tokens_detected:
                        if looks_like_tool_call_text(accumulated_content):
                            _raw_tokens_detected = True
                            logger.warning(
                                "Stream: raw tool tokens detected mid-stream at chunk {}, "
                                "suppressing further content output",
                                chunk_count,
                            )
                        else:
                            yield LLMStreamChunk(delta_content=content)
                    continue

                if finish_reason:
                    logger.info(
                        "LLM stream: finish_reason={}, tool_calls_acc={}",
                        finish_reason,
                        len(_tc_acc),
                    )

                if _tc_acc:
                    parsed_tool_calls: list[ToolCall] = []
                    parse_failed = False

                    for idx in sorted(_tc_acc):
                        entry = _tc_acc[idx]
                        logger.debug(
                            "[LLM Stream] Tool call assembled: name={}, args_len={}, args={}",
                            entry["name"],
                            len(entry["args_str"]),
                            entry["args_str"][:200] if entry["args_str"] else "(empty)",
                        )
                        try:
                            args = (
                                json.loads(entry["args_str"])
                                if entry["args_str"]
                                else {}
                            )
                        except json.JSONDecodeError:
                            try:
                                args = self._parse_tool_args_with_repair(
                                    entry["args_str"]
                                )
                            except (json.JSONDecodeError, Exception):
                                logger.warning(
                                    "[LLM Stream] Failed to parse tool call args after repair, "
                                    "retrying via non-stream"
                                )
                                parse_failed = True
                                break

                        parsed_tool_calls.append(
                            ToolCall(
                                id=entry["id"] or f"tc_{uuid.uuid4().hex[:12]}",
                                name=entry["name"],
                                arguments=args,
                            )
                        )

                    if parse_failed:
                        try:
                            fallback = await self.async_chat(
                                messages=messages,
                                tools=tools,
                                system=system,
                                **kwargs,
                            )
                            parsed_tool_calls = fallback.tool_calls
                        except Exception as exc:
                            logger.warning(
                                "[LLM Stream] Non-stream fallback failed: {}",
                                exc,
                            )
                            parsed_tool_calls = []

                    if parsed_tool_calls:
                        for idx, tc in enumerate(parsed_tool_calls):
                            yield LLMStreamChunk(
                                delta_content=content if idx == 0 else None,
                                delta_tool_call=tc,
                                finish_reason=finish_reason,
                                usage=usage if idx == 0 else None,
                            )
                            content = None
                    else:
                        yield LLMStreamChunk(
                            delta_content=content,
                            finish_reason=finish_reason,
                            usage=usage,
                        )
                elif accumulated_content and looks_like_tool_call_text(
                    accumulated_content
                ):
                    logger.warning(
                        "Stream detected raw tool tokens, falling back to non-stream retry"
                    )
                    try:
                        fallback = await self.async_chat(
                            messages=messages,
                            tools=tools,
                            system=system,
                            **kwargs,
                        )
                        if fallback.tool_calls:
                            for tc in fallback.tool_calls:
                                yield LLMStreamChunk(
                                    delta_tool_call=tc,
                                    finish_reason=FINISH_TOOL_CALLS,
                                )
                        else:
                            yield LLMStreamChunk(
                                finish_reason=finish_reason,
                                usage=usage,
                            )
                    except Exception as exc:
                        logger.warning("Stream fallback to async_chat failed: %s", exc)
                        yield LLMStreamChunk(
                            finish_reason=finish_reason,
                            usage=usage,
                        )
                else:
                    yield LLMStreamChunk(
                        delta_content=content,
                        finish_reason=finish_reason,
                        usage=usage,
                    )

            logger.info(
                "[LLM Stream] Completed: {} chunks, {} total chars, {} tool_calls",
                chunk_count,
                total_chars,
                len(_tc_acc),
            )

            if accumulated_content or _tc_acc:
                response_obj = LLMResponse(
                    content=accumulated_content or None,
                    tool_calls=[],
                    usage=_last_usage,
                    model=self.model,
                    finish_reason=_last_finish_reason,
                    raw_response=None,
                )
                log_llm_response(response_obj, model=self.model)

        except LLMException:
            raise
        except Exception as e:
            if (
                "building chunks" in str(e).lower()
                or "usage calculation" in str(e).lower()
            ):
                logger.warning(f"[LLM Stream] Ignored LiteLLM internal error: {e}")
                return
            raise self._parse_error(e, self.model)


# ── Module-level convenience wrappers ──────────────────────────────

_llm: LLMClient | None = None
_llm_lock = threading.Lock()


def _get_llm() -> LLMClient:
    """获取 LLM 单例（延迟初始化, 线程安全）。"""
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = LLMClient()
    return _llm


def reload_llm_config() -> None:
    """刷新全局 LLM 单例的配置。

    当模型配置发生变化（如删除模型、切换模型）时调用此方法，
    确保后续的 LLM 调用使用最新的配置。
    """
    global _llm
    if _llm is not None:
        _llm.reload_config()
        logger.info("Global LLM singleton configuration reloaded")
    else:
        logger.debug("LLM singleton not initialized, nothing to reload")


async def _chat_completions_impl(system: str, messages: list[dict[str, Any]]) -> str:
    resp = await _get_llm().async_chat(messages=messages, system=system)
    return resp.content or ""


def chat_completions(system: str, messages: list[dict[str, Any]]) -> str:
    """同步 LLM chat completion，在无 event loop 时使用 asyncio.run，否则用线程池。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _chat_completions_impl(system, messages))
            return future.result()
    return asyncio.run(_chat_completions_impl(system, messages))


async def chat_completions_async(
    system: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
) -> str:
    """异步 LLM chat completion。"""
    kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = await _get_llm().async_chat(messages=messages, system=system, **kwargs)
    return resp.content or ""
