
from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from core.skills.schema import SkillCall



async def format_user_content(
    text: str,
    media: list[str] | list[Path] | None,
) -> str | list[dict[str, Any]]:
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
        images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    if not images:
        return text
    return images + [{"type": "text", "text": text}]


def skill_call_to_openai_payload(skill_call: SkillCall) -> dict[str, Any]:
    return {
        "id": skill_call.id,
        "type": "function",
        "function": {
            "name": skill_call.name,
            "arguments": json.dumps(skill_call.arguments, ensure_ascii=False),
        },
    }



