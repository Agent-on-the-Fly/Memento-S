"""Token counting utilities using tiktoken for accurate calculation.

Provides accurate token counting for various LLM models using tiktoken.
Falls back to estimation when tiktoken is not available or for unsupported models.
"""

from functools import lru_cache
from typing import Any


def get_tokenizer(model_name: str = "gpt-4"):
    """Get tiktoken tokenizer for the specified model.
    
    Args:
        model_name: Model name (e.g., "gpt-4", "gpt-3.5-turbo", "claude-3")
        
    Returns:
        Tiktoken encoding object or None if tiktoken not available
    """
    try:
        import tiktoken
        
        # Map common model names to tiktoken encodings
        model_mapping = {
            # OpenAI models
            "gpt-4": "cl100k_base",
            "gpt-4-turbo": "cl100k_base",
            "gpt-4o": "o200k_base",
            "gpt-4o-mini": "o200k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-3.5": "cl100k_base",
            "text-embedding-ada-002": "cl100k_base",
            "text-embedding-3-small": "cl100k_base",
            "text-embedding-3-large": "cl100k_base",
            # Default fallback
            "default": "cl100k_base",
        }
        
        # Try exact match first
        if model_name in model_mapping:
            return tiktoken.get_encoding(model_mapping[model_name])
        
        # Try to match model prefix
        for key, encoding in model_mapping.items():
            if key in model_name.lower():
                return tiktoken.get_encoding(encoding)
        
        # Default to cl100k_base for most modern models
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None
    except Exception:
        return None


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Count tokens accurately using tiktoken.

    Falls back to estimation if tiktoken is not available.
    Results are cached via LRU for repeated calls on the same text.

    Args:
        text: Input text to count tokens for
        model_name: Model name to determine encoding

    Returns:
        Token count
    """
    if not text:
        return 0
    return _count_tokens_cached(text, model_name)


@lru_cache(maxsize=1024)
def _count_tokens_cached(text: str, model_name: str = "gpt-4") -> int:
    """LRU-cached token counting implementation."""
    tokenizer = get_tokenizer(model_name)
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return _estimate_tokens_fallback(text)


def count_tokens_messages(messages: list[dict[str, Any]], model_name: str = "gpt-4") -> int:
    """Count tokens for a list of messages (OpenAI format).
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model_name: Model name to determine encoding
        
    Returns:
        Total token count including message formatting overhead
    """
    if not messages:
        return 0
    
    tokenizer = get_tokenizer(model_name)
    if not tokenizer:
        # Fallback: sum content lengths
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += _estimate_tokens_fallback(content)
            elif isinstance(content, list):
                # Multimodal content
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total += _estimate_tokens_fallback(part.get("text", ""))
        return total + len(messages) * 4  # Formatting overhead
    
    total_tokens = 0
    
    for message in messages:
        # Every message follows <|start|>{role/name}\n{content}<|end|>\n

        total_tokens += 3  # <|start|>
        
        # Add role tokens
        role = message.get("role", "")
        total_tokens += len(tokenizer.encode(role))
        
        # Add content tokens
        content = message.get("content", "")
        if isinstance(content, str):
            total_tokens += len(tokenizer.encode(content))
        elif isinstance(content, list):
            # Multimodal content - count text parts
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total_tokens += len(tokenizer.encode(part.get("text", "")))
        
        total_tokens += 3  # <|end|>
    
    # Every reply is primed with <|start|>assistant<|message|>
    total_tokens += 3
    
    return total_tokens


def _estimate_tokens_fallback(text: str) -> int:
    """Estimate token count when tiktoken is not available.
    
    Uses improved heuristics:
    - English/ASCII: ~3.5 chars per token (was 4)
    - Chinese/CJK: ~2.0 tokens per char (was 1.5, but actual is closer to 1.5-2)
    - Other chars: ~1.5 chars per token
    
    Args:
        text: Input text to estimate
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Count different character types
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    cjk_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3000" <= c <= "\u303f")
    other_chars = len(text) - ascii_chars - cjk_chars
    
    # Estimate tokens with more accurate ratios
    ascii_tokens = ascii_chars / 3.5  # ~3.5 chars per token for English (more accurate)
    cjk_tokens = cjk_chars * 1.8  # ~1.8 tokens per CJK char (closer to actual)
    other_tokens = other_chars / 1.5  # conservative estimate for other chars
    
    return int(ascii_tokens + cjk_tokens + other_tokens)


def estimate_tokens(text: str) -> int:
    """Estimate token count for mixed language text (legacy API).
    
    Now uses tiktoken when available for better accuracy.
    
    Args:
        text: Input text to estimate
        
    Returns:
        Estimated token count
    """
    return count_tokens(text)


def estimate_tokens_batch(texts: list[str]) -> int:
    """Estimate total tokens for a list of texts (legacy API).
    
    Args:
        texts: List of texts to estimate
        
    Returns:
        Total estimated token count
    """
    return sum(count_tokens(text) for text in texts)


def get_token_stats(text: str, model_name: str = "gpt-4") -> dict:
    """Get detailed token statistics.
    
    Args:
        text: Input text to analyze
        model_name: Model name to determine encoding
        
    Returns:
        Dictionary with token breakdown and method used
    """
    if not text:
        return {
            "total_chars": 0,
            "ascii_chars": 0,
            "cjk_chars": 0,
            "other_chars": 0,
            "token_count": 0,
            "method": "tiktoken",
        }
    
    tokenizer = get_tokenizer(model_name)
    method = "tiktoken" if tokenizer else "estimate"
    
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    cjk_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3000" <= c <= "\u303f")
    other_chars = len(text) - ascii_chars - cjk_chars
    
    return {
        "total_chars": len(text),
        "ascii_chars": ascii_chars,
        "cjk_chars": cjk_chars,
        "other_chars": other_chars,
        "token_count": count_tokens(text, model_name),
        "method": method,
    }


def calculate_context_tokens(
    system_prompt: str,
    history: list[dict[str, Any]],
    user_message: str,
    model_name: str = "gpt-4"
) -> dict[str, int]:
    """Calculate token distribution for a complete context.
    
    Args:
        system_prompt: System prompt text
        history: List of history messages
        user_message: Current user message
        model_name: Model name to determine encoding
        
    Returns:
        Dictionary with token breakdown by component
    """
    system_tokens = count_tokens(system_prompt, model_name)
    history_tokens = count_tokens_messages(history, model_name)
    user_tokens = count_tokens(user_message, model_name)
    
    total = system_tokens + history_tokens + user_tokens + 3  # +3 for assistant priming
    
    return {
        "system": system_tokens,
        "history": history_tokens,
        "user": user_tokens,
        "total": total,
        "remaining": 128000 - total,  # Assume 128k context window
    }


# Convenience function for GUI
def format_token_display(current: int, max_tokens: int = 128000) -> str:
    """Format token count for display in UI.
    
    Args:
        current: Current token count
        max_tokens: Maximum context window size
        
    Returns:
        Formatted string like "1.2k / 128k"
    """
    def fmt(n: int) -> str:
        if n >= 1000:
            return f"{n/1000:.1f}k"
        return str(n)
    
    return f"{fmt(current)} / {fmt(max_tokens)}"


def get_token_progress(current: int, max_tokens: int = 128000) -> float:
    """Get progress value (0.0-1.0) for token usage.
    
    Args:
        current: Current token count
        max_tokens: Maximum context window size
        
    Returns:
        Progress value between 0.0 and 1.0
    """
    return min(current / max_tokens, 1.0)


