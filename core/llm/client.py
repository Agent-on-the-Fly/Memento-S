
from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from litellm import acompletion
from litellm.types.utils import ModelResponse

from core.config import g_settings
from core.config.logging import get_logger
from .schema import LLMResponse, LLMStreamChunk, ToolCall

logger = get_logger(__name__)



def _build_model_string(provider: str, model_name: str) -> str:
    provider = provider.lower().strip()

    if provider in ("anthropic", "claude"):
        return f"anthropic/{model_name}" if "/" not in model_name else model_name
    if provider == "openrouter":
        return (
            f"openrouter/{model_name}"
            if not model_name.startswith("openrouter/")
            else model_name
        )
    if provider == "openai":
        return f"openai/{model_name}" if "/" not in model_name else model_name
    if provider in ("ollama", "vllm", "sglang", "together_ai"):
        return f"{provider}/{model_name}"
    return model_name if "/" in model_name else f"{provider}/{model_name}"


def _resolve_api_key() -> str | None:
    return g_settings.resolve_llm_api_key()


def _resolve_base_url() -> str | None:
    return g_settings.resolve_llm_base_url()


def _build_completion_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    provider = (g_settings.llm_api or "").lower()

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": g_settings.llm_max_tokens,
        "temperature": g_settings.llm_temperature,
        "timeout": g_settings.llm_timeout,
        "drop_params": True,
        "stream": False,
    }

    api_key = _resolve_api_key()
    if api_key:
        kwargs["api_key"] = api_key
    base_url = _resolve_base_url()
    if base_url:
        kwargs["api_base"] = base_url

    if tools:
        kwargs["tools"] = tools

    extra_body: dict[str, Any] = {}
    if "sglang" in model.lower():
        extra_body["thinking"] = False
    if extra_body:
        kwargs["extra_body"] = extra_body

    if "openrouter" in model.lower():
        headers: dict[str, str] = {}
        if g_settings.openrouter_site_url:
            headers["HTTP-Referer"] = g_settings.openrouter_site_url
        if g_settings.openrouter_app_name:
            headers["X-Title"] = g_settings.openrouter_app_name
        if headers:
            kwargs["extra_headers"] = headers

    kwargs.update(extra)
    return kwargs


def _parse_tool_calls(raw_tool_calls: list[Any] | None) -> list[ToolCall]:
    if not raw_tool_calls:
        return []

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    result: list[ToolCall] = []
    for tc in raw_tool_calls:
        try:
            func = _get(tc, "function")
            if func is None:
                logger.warning("tool_call missing 'function' field: %s", tc)
                continue

            args_raw = _get(func, "arguments", "")

            if isinstance(args_raw, dict):
                arguments = args_raw
            elif isinstance(args_raw, str) and args_raw.strip():
                arguments = json.loads(args_raw)
            else:
                arguments = {}

            tc_id = _get(tc, "id") or ""
            func_name = _get(func, "name") or ""
            result.append(ToolCall(id=tc_id, name=func_name, arguments=arguments))
        except Exception as exc:
            logger.warning("Failed to parse tool_call: %s — %s", tc, exc)
    return result



class LLM:

    def __init__(self, model: str | None = None) -> None:
        if model:
            self._default_model = model
        else:
            provider = g_settings.llm_api or "anthropic"
            model_name = g_settings.llm_model
            self._default_model = _build_model_string(provider, model_name)

    @property
    def default_model(self) -> str:
        return self._default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        effective_model = model or self._default_model

        full_messages: list[dict[str, Any]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        completion_kwargs = _build_completion_kwargs(
            effective_model, full_messages, tools=tools, **kwargs
        )

        try:
            raw_response: ModelResponse = await acompletion(**completion_kwargs)  # type: ignore[assignment]
        except Exception as exc:
            error_msg = f"LLM Request Failed [{effective_model}]: {exc}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from exc

        content: str | None = None
        tool_calls: list[ToolCall] = []

        if hasattr(raw_response, "choices") and raw_response.choices:
            message = raw_response.choices[0].message  # type: ignore[union-attr]
            if hasattr(message, "content") and message.content:
                content = message.content
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = _parse_tool_calls(message.tool_calls)

        return LLMResponse(content=content, tool_calls=tool_calls, raw=raw_response)


    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        effective_model = model or self._default_model

        full_messages: list[dict[str, Any]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        completion_kwargs = _build_completion_kwargs(
            effective_model, full_messages, tools=tools, **kwargs
        )
        completion_kwargs["stream"] = True

        try:
            raw_stream = await acompletion(**completion_kwargs)
        except Exception as exc:
            error_msg = f"LLM Stream Request Failed [{effective_model}]: {exc}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from exc

        async for chunk in raw_stream:
            if not hasattr(chunk, "choices") or not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason
            yield LLMStreamChunk(
                delta_content=getattr(delta, "content", None),
                tool_calls_delta=getattr(delta, "tool_calls", None),
                finish_reason=finish_reason,
            )

    async def chat_stream_collect(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> tuple[LLMResponse, list[LLMStreamChunk]]:
        content_parts: list[str] = []
        tool_calls_raw: dict[int, dict[str, Any]] = {}
        chunks: list[LLMStreamChunk] = []

        async for chunk in self.chat_stream(
            messages, tools=tools, system=system, model=model, **kwargs
        ):
            chunks.append(chunk)
            if chunk.delta_content:
                content_parts.append(chunk.delta_content)
            if chunk.tool_calls_delta:
                for tc_delta in chunk.tool_calls_delta:
                    idx = getattr(tc_delta, "index", 0) if hasattr(tc_delta, "index") else 0
                    if idx not in tool_calls_raw:
                        tool_calls_raw[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    entry = tool_calls_raw[idx]
                    tc_id = getattr(tc_delta, "id", None)
                    if tc_id:
                        entry["id"] = tc_id
                    func = getattr(tc_delta, "function", None)
                    if func:
                        func_name = getattr(func, "name", None)
                        if func_name:
                            entry["function"]["name"] += str(func_name)
                        func_args = getattr(func, "arguments", None)
                        if func_args:
                            entry["function"]["arguments"] += str(func_args)

        content = "".join(content_parts) if content_parts else None
        parsed_tool_calls = (
            _parse_tool_calls(
                [tool_calls_raw[k] for k in sorted(tool_calls_raw)]
            )
            if tool_calls_raw
            else []
        )
        return LLMResponse(content=content, tool_calls=parsed_tool_calls), chunks
