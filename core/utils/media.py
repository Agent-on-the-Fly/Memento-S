"""Media and content formatting utilities."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import Any


async def format_user_content(
    text: str,
    media: list[str] | list[Path] | None,
) -> str | list[dict[str, Any]]:
    """
    将用户文本与可选图片组装为单条 user 消息的 content（纯文本或 multimodal 列表）。
    图片读取在线程池执行，不阻塞事件循环。
    """
    if not media:
        return text
    images = []
    for path in media:
        p = Path(path)
        mime, _ = mimetypes.guess_type(str(p))
        if not p.is_file() or not mime or not mime.startswith("image/"):
            continue
        raw = await asyncio.to_thread(p.read_bytes)
        b64 = base64.b64encode(raw).decode()
        images.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        )
    if not images:
        return text
    return images + [{"type": "text", "text": text}]
