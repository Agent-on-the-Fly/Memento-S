"""Build a LangChain ChatOpenAI model from OpenRouter environment variables.

Reads env vars at call time (not import time) so that ``/config set``
changes take effect without restarting the process.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def build_chat_model() -> ChatOpenAI:
    """Create a ``ChatOpenAI`` configured for OpenRouter.

    Environment variables used:
        OPENROUTER_API_KEY      – API key (required for real calls)
        OPENROUTER_BASE_URL     – Base URL (default: OpenRouter v1)
        OPENROUTER_MODEL        – Model identifier
        OPENROUTER_MAX_TOKENS   – Max output tokens
        OPENROUTER_TIMEOUT      – Request timeout in seconds
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
    max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", "100000"))
    timeout = int(os.getenv("OPENROUTER_TIMEOUT", "60"))

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
    )
