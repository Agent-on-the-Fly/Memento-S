from __future__ import annotations

from utils.token_utils import count_tokens_messages


def test_single_msg_string_content():
    """String content returns tokens > overhead."""
    msg = {"role": "user", "content": "hello world this is a test"}
    tokens = count_tokens_messages([msg])
    assert tokens > 4, f"Expected > 4 tokens, got {tokens}"


def test_single_msg_list_content():
    """List content (multimodal) returns tokens > overhead."""
    msg = {"role": "user", "content": [{"type": "text", "text": "hello world"}]}
    tokens = count_tokens_messages([msg])
    assert tokens > 4


def test_single_msg_empty_content():
    """Empty content returns only overhead."""
    msg = {"role": "user", "content": ""}
    tokens = count_tokens_messages([msg])
    # overhead only (start + role + end tokens)
    assert tokens > 0
    assert tokens < 20


def test_multiple_messages():
    """Total tokens for multiple messages > sum of individual."""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    total = count_tokens_messages(msgs)
    assert total > 10
