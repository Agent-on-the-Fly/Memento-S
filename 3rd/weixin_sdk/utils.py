"""
Utility functions for Weixin SDK.

This module provides helper functions similar to TypeScript's util/ directory.
"""

import re


def markdown_to_plain_text(text: str) -> str:
    """
    Convert markdown-formatted text to plain text for Weixin delivery.

    Similar to TypeScript's markdownToPlainText() in send.ts.

    Args:
        text: Markdown formatted text

    Returns:
        Plain text with markdown syntax removed

    Example:
        >>> markdown_to_plain_text("**bold** and *italic*")
        'bold and italic'
        >>> markdown_to_plain_text("[link](http://example.com)")
        'link'
    """
    result = text

    # Code blocks: strip fences, keep content
    # ```language\ncode\n``` -> code
    result = re.sub(r"```[^\n]*\n?([\s\S]*?)```", lambda m: m.group(1).strip(), result)

    # Inline code: `code` -> code
    result = re.sub(r"`([^`]+)`", r"\1", result)

    # Images: ![alt](url) -> (removed entirely)
    result = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", result)

    # Links: [text](url) -> text
    result = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", result)

    # Tables: remove separator rows, convert to text
    # | a | b | -> a  b
    result = re.sub(r"^\|[\s:|-]+\|$", "", result, flags=re.MULTILINE)
    result = re.sub(
        r"^\|(.+)\|$",
        lambda m: "  ".join(cell.strip() for cell in m.group(1).split("|")),
        result,
        flags=re.MULTILINE,
    )

    # Bold: **text** or __text__ -> text
    result = re.sub(r"\*\*([^*]+)\*\*", r"\1", result)
    result = re.sub(r"__([^_]+)__", r"\1", result)

    # Italic: *text* or _text_ -> text
    result = re.sub(r"\*([^*]+)\*", r"\1", result)
    result = re.sub(r"_([^_]+)_", r"\1", result)

    # Strikethrough: ~~text~~ -> text
    result = re.sub(r"~~([^~]+)~~", r"\1", result)

    # Headers: # text -> text
    result = re.sub(r"^#+\s*", "", result, flags=re.MULTILINE)

    # Blockquotes: > text -> text
    result = re.sub(r"^>\s*", "", result, flags=re.MULTILINE)

    # Horizontal rules: --- or *** or ___ -> (remove)
    result = re.sub(r"^[\-*_]{3,}\s*$", "", result, flags=re.MULTILINE)

    # Clean up extra whitespace
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = result.strip()

    return result


def chunk_text(text: str, limit: int = 4000, respect_paragraphs: bool = True) -> list:
    """
    Split text into chunks at paragraph boundaries when possible.

    Similar to TypeScript's text chunking in send.ts.

    Args:
        text: Text to chunk
        limit: Maximum chunk size
        respect_paragraphs: Try to split at paragraph boundaries

    Returns:
        List of text chunks

    Example:
        >>> chunks = chunk_text("Paragraph 1\n\nParagraph 2\n\nParagraph 3", limit=20)
        >>> len(chunks)
        3
    """
    if len(text) <= limit:
        return [text]

    chunks = []

    if respect_paragraphs:
        # Try to split at paragraph boundaries first
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for paragraph in paragraphs:
            # If paragraph itself is too long, we'll need to split it
            if len(paragraph) > limit:
                # First, add current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Split long paragraph at sentence boundaries or just hard split
                while paragraph:
                    if len(paragraph) <= limit:
                        current_chunk = paragraph
                        paragraph = ""
                    else:
                        # Try to find a good split point
                        split_point = limit

                        # Look for sentence ending (period + space)
                        for i in range(limit - 1, limit // 2, -1):
                            if paragraph[i : i + 2] in [". ", "? ", "! "]:
                                split_point = i + 2
                                break

                        chunks.append(paragraph[:split_point].strip())
                        paragraph = paragraph[split_point:].strip()
            else:
                # Paragraph fits within limit
                if current_chunk:
                    # Check if adding this paragraph would exceed limit
                    potential_chunk = current_chunk + "\n\n" + paragraph
                    if len(potential_chunk) <= limit:
                        current_chunk = potential_chunk
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = paragraph
                else:
                    current_chunk = paragraph

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
    else:
        # Simple hard split
        for i in range(0, len(text), limit):
            chunks.append(text[i : i + limit])

    return chunks


def truncate_string(s: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate a string if it exceeds max_length.

    Args:
        s: String to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated

    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s

    return s[: max_length - len(suffix)] + suffix


def generate_short_id(prefix: str = "wx") -> str:
    """
    Generate a short unique ID.

    Args:
        prefix: ID prefix

    Returns:
        Short unique ID string
    """
    import uuid

    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def is_remote_url(path: str) -> bool:
    """
    Check if a path is a remote URL.

    Args:
        path: Path or URL to check

    Returns:
        True if remote URL, False if local path
    """
    return path.startswith("http://") or path.startswith("https://")


def is_local_file_path(path: str) -> bool:
    """
    Check if a path is a local file path (not a URL).

    Args:
        path: Path to check

    Returns:
        True if local file path
    """
    return not path.startswith("http://") and not path.startswith("https://")


def normalize_weixin_user_id(raw_id: str) -> str:
    """
    Normalize Weixin user ID.

    Weixin user IDs should end with @im.wechat.

    Args:
        raw_id: Raw user ID

    Returns:
        Normalized user ID
    """
    if "@" not in raw_id:
        return f"{raw_id}@im.wechat"
    return raw_id


# Export all utility functions
__all__ = [
    "markdown_to_plain_text",
    "chunk_text",
    "truncate_string",
    "generate_short_id",
    "is_remote_url",
    "is_local_file_path",
    "normalize_weixin_user_id",
]
