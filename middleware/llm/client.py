"""
异步 LLM 客户端 — 统一调用接口。

特性：
- 基于 litellm 的多提供商支持
- 自动重试机制（指数退避）
- 超时控制
- 熔断保护（Circuit Breaker）
- 统一响应格式
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator

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
)
from .schema import LLMResponse, LLMStreamChunk, ToolCall, Message

logger = get_logger()


@dataclass
class RetryConfig:
    """重试配置。"""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple = (
        LLMTimeoutError,
        LLMRateLimitError,
        LLMConnectionError,
    )


@dataclass
class CircuitBreakerConfig:
    """熔断器配置。"""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3


class CircuitBreaker:
    """简单熔断器实现。"""

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self.failures = 0
        self.last_failure_time: float | None = None
        self.state = "closed"  # closed, open, half-open
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """在熔断器保护下执行函数。"""
        async with self._lock:
            if self.state == "open":
                if (
                    time.time() - (self.last_failure_time or 0)
                    > self.config.recovery_timeout
                ):
                    self.state = "half-open"
                    self.half_open_calls = 0
                    logger.info("Circuit breaker entering half-open state")
                else:
                    raise LLMConnectionError("Circuit breaker is open")

            if (
                self.state == "half-open"
                and self.half_open_calls >= self.config.half_open_max_calls
            ):
                raise LLMConnectionError("Circuit breaker half-open limit reached")

            if self.state == "half-open":
                self.half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    async def _on_success(self):
        async with self._lock:
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
                self.half_open_calls = 0
                logger.info("Circuit breaker closed")

    async def _on_failure(self):
        async with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.config.failure_threshold:
                if self.state != "open":
                    self.state = "open"
                    logger.warning(
                        f"Circuit breaker opened after {self.failures} failures"
                    )


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
    ):
        """
        Args:
            config_manager: 配置管理器，默认使用全局实例
            retry_config: 重试配置
            circuit_config: 熔断器配置
        """
        self.config_manager = config_manager or ConfigManager()
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker(circuit_config)

        # 加载配置
        self._load_config()

    def _load_config(self):
        """从 ConfigManager 加载 LLM 配置。

        始终从磁盘重新加载以获取最新配置，确保模型切换生效。
        context_window 优先从 litellm 自动检测，检测不到则用 profile 默认值。
        """
        try:
            config = self.config_manager.load()

            llm_config = config.llm
            profile = llm_config.current_profile

            self.model = profile.model
            self.api_key = profile.api_key
            self.base_url = profile.base_url
            self.context_window = self._detect_context_window(profile)
            self.max_tokens = profile.max_tokens
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

    @staticmethod
    def _detect_context_window(profile) -> int:
        """尝试通过 litellm 自动检测模型 context window，失败则用 profile 默认值。"""
        try:
            info = litellm.get_model_info(profile.model)
            detected = info.get("max_input_tokens") or info.get("max_tokens")
            if detected and isinstance(detected, int) and detected > 0:
                logger.info(
                    "Auto-detected context_window={} for model {}",
                    detected, profile.model,
                )
                return detected
        except Exception:
            pass
        return profile.context_window

    def _build_model_string(self) -> str:
        """构建 litellm 模型字符串。"""
        model = self.model

        if "/" in model:
            # 已有 provider 前缀，直接使用
            return model

        # 根据 base_url 推断 provider
        if self.base_url:
            if "anthropic" in self.base_url:
                return f"anthropic/{model}"
            elif "openrouter" in self.base_url:
                return f"openrouter/{model}"
            elif "openai" in self.base_url:
                return f"openai/{model}"

        # 默认使用 anthropic
        return f"anthropic/{model}"

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """确保 system message 只出现在第一位。

        某些模型（如 OpenAI o1/o3、部分 Anthropic 接口）要求 system message
        必须且只能出现在消息列表的开头。执行循环中会在中间追加 system 消息作为
        运行时控制指令，在发送前将非第一位的 system 消息转换为 user 消息以兼容
        所有模型。
        """
        if not messages:
            return messages

        normalized = []
        for i, msg in enumerate(messages):
            if i > 0 and msg.get("role") == "system":
                normalized.append({
                    "role": "user",
                    "content": f"[System]: {msg['content']}",
                })
            else:
                normalized.append(msg)
        return normalized

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

        input_tokens = count_tokens_messages(messages)
        effective_max = min(self.max_tokens, self.context_window - input_tokens)
        effective_max = max(effective_max, 256)
        if effective_max < self.max_tokens:
            logger.info(
                "max_tokens capped: {} -> {} (context_window={}, input={})",
                self.max_tokens, effective_max, self.context_window, input_tokens,
            )

        kwargs: dict[str, Any] = {
            "model": model_str,
            "messages": messages,
            "max_tokens": effective_max,
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

        kwargs.update(extra)
        return kwargs

    def _parse_error(self, error: Exception, model: str) -> LLMException:
        """解析异常为统一的 LLMException。"""
        error_str = str(error).lower()

        if "timeout" in error_str:
            return LLMTimeoutError(model=model)
        elif "rate limit" in error_str or "429" in error_str:
            return LLMRateLimitError(model=model)
        elif "connection" in error_str or "network" in error_str:
            return LLMConnectionError(model=model)
        elif "authentication" in error_str or "401" in error_str or "403" in error_str:
            return LLMAuthenticationError(model=model)
        elif "content filter" in error_str or "moderation" in error_str:
            return LLMContentFilterError(model=model)
        else:
            return LLMException(str(error), model=model, retryable=True)

    async def _call_with_retry(
        self,
        func,
        *args,
        **kwargs,
    ) -> Any:
        """带重试机制的调用。"""
        last_exception = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                llm_error = self._parse_error(e, self.model)
                last_exception = llm_error

                if not llm_error.retryable or attempt >= self.retry_config.max_retries:
                    raise llm_error

                # 计算退避延迟
                delay = min(
                    self.retry_config.base_delay
                    * (self.retry_config.exponential_base**attempt),
                    self.retry_config.max_delay,
                )

                if isinstance(llm_error, LLMRateLimitError) and llm_error.retry_after:
                    delay = max(delay, llm_error.retry_after)

                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}), "
                    f"retrying in {delay:.1f}s: {llm_error.message}"
                )
                await asyncio.sleep(delay)

        raise last_exception

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
                func_name = (
                    func.get("name")
                    if isinstance(func, dict)
                    else getattr(func, "name", "")
                )

                result.append(
                    ToolCall(id=tc_id or "", name=func_name or "", arguments=arguments)
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
                repaired = repaired.replace("'", '"')
                repaired = re.sub(r",\s*\}", "}", repaired)
                return json.loads(repaired)

    async def async_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        异步非流式调用 LLM。

        Args:
            messages: 对话历史
            tools: 工具定义
            system: 系统提示
            **kwargs: 额外参数

        Returns:
            LLMResponse 统一响应对象
        """
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # DEBUG: 记录 LLM 请求
        log_llm_request(full_messages, tools, self.model)

        completion_kwargs = self._build_completion_kwargs(
            full_messages, tools=tools, **kwargs
        )

        async def _do_completion():
            return await acompletion(**completion_kwargs)

        try:
            raw_response = await self.circuit_breaker.call(
                lambda: self._call_with_retry(_do_completion)
            )
        except LLMException:
            raise
        except Exception as e:
            raise self._parse_error(e, self.model)

        # 解析响应
        content: str | None = None
        tool_calls: list[ToolCall] = []
        usage = {}
        finish_reason = None

        if hasattr(raw_response, "choices") and raw_response.choices:
            message = raw_response.choices[0].message
            content = getattr(message, "content", None)
            tool_calls = self._parse_tool_calls(getattr(message, "tool_calls", None))
            finish_reason = getattr(raw_response.choices[0], "finish_reason", None)

        if hasattr(raw_response, "usage"):
            usage = raw_response.usage

        logger.info(
            "LLM async_chat: finish_reason={}, tool_calls={}, content_len={}",
            finish_reason,
            len(tool_calls),
            len(content or ""),
        )
        logger.info("LLM async_chat raw_response={}", raw_response)

        # DEBUG: 记录 LLM 响应
        response_obj = LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=self.model,
            finish_reason=finish_reason,
            raw_response=raw_response,
        )
        log_llm_response(response_obj)

        return response_obj

    async def async_stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """
        异步流式调用 LLM。

        Args:
            messages: 对话历史
            tools: 工具定义
            system: 系统提示
            **kwargs: 额外参数

        Yields:
            LLMStreamChunk 流式块
        """
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        completion_kwargs = self._build_completion_kwargs(
            full_messages, tools=tools, stream=True, **kwargs
        )

        try:
            # 流式调用不经过 _call_with_retry，直接调用以支持 generator
            logger.debug(f"[LLM Stream] Starting stream call with model: {self.model}")

            # DEBUG: 记录流式请求
            from utils.debug_logger import log_llm_request

            log_llm_request(full_messages, tools)

            raw_stream = await acompletion(**completion_kwargs)
            chunk_count = 0
            total_chars = 0
            accumulated_content = ""  # 累积完整内容
            _last_usage = {}
            _last_finish_reason = "stop"

            # 流式 tool call 累积器：index -> {id, name, args_str}
            _tc_acc: dict[int, dict[str, str]] = {}

            async for chunk in raw_stream:
                # 提取 usage（通常在最后一个 chunk 中）
                usage = _last_usage  # 默认使用上一次值
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

                # 累积 tool call delta（流式 delta 中 arguments 是 JSON 片段，不能直接 json.loads）
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
                    accumulated_content += content  # 累积内容
                    logger.debug(
                        f"[LLM Stream] Chunk #{chunk_count}: {len(content)} chars, total: {total_chars}"
                    )

                # 流未结束时：只 yield 文本 chunk
                if not finish_reason:
                    if content:
                        yield LLMStreamChunk(delta_content=content)
                    continue

                if finish_reason:
                    logger.info(
                        "LLM stream: finish_reason={}, tool_calls_acc={}",
                        finish_reason,
                        len(_tc_acc),
                    )

                # finish_reason 已设置：组装完整 ToolCall 并 yield
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
                            logger.warning(
                                "[LLM Stream] Failed to parse tool call args, retrying via non-stream completion"
                            )
                            parse_failed = True
                            break

                        parsed_tool_calls.append(
                            ToolCall(
                                id=entry["id"],
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
                                usage=usage
                                if idx == 0
                                else None,  # 只在第一个 chunk 带 usage
                            )
                            content = None  # 只在第一个 chunk 带文本
                    else:
                        yield LLMStreamChunk(
                            delta_content=content,
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
                f"[LLM Stream] Completed: {chunk_count} chunks, {total_chars} total chars"
            )

            # DEBUG: 记录流式响应完成
            if accumulated_content:
                from utils.debug_logger import log_llm_response
                from .schema import LLMResponse

                response_obj = LLMResponse(
                    content=accumulated_content,
                    tool_calls=[],  # 流式 tool calls 已单独处理
                    usage=_last_usage,
                    model=self.model,
                    finish_reason=_last_finish_reason,
                    raw_response=None,
                )
                log_llm_response(response_obj)

        except LLMException:
            raise
        except Exception as e:
            # LiteLLM internal bug: streaming usage calculation error is non-fatal
            # (content has already been yielded successfully)
            if (
                "building chunks" in str(e).lower()
                or "usage calculation" in str(e).lower()
            ):
                logger.warning(f"[LLM Stream] Ignored LiteLLM internal error: {e}")
                return
            raise self._parse_error(e, self.model)
