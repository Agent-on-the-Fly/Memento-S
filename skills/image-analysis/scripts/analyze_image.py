#!/usr/bin/env python3
"""Analyze a local image with OpenRouter multimodal chat completions."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _normalize_openrouter_base(url: str) -> str:
    base = str(url or "").strip().rstrip("/")
    if not base:
        return "https://openrouter.ai/api/v1"
    if base.endswith("/api"):
        return base + "/v1"
    if base.endswith("/api/v1"):
        return base
    if base.endswith("openrouter.ai"):
        return base + "/api/v1"
    return base


def _image_to_data_url(image_path: Path) -> str:
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/png"
    raw = image_path.read_bytes()
    if not raw:
        raise RuntimeError(f"Image file is empty: {image_path}")
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_content_text(raw: dict[str, Any]) -> str:
    choices = raw.get("choices") or []
    if not choices:
        return ""
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def analyze_image(
    *,
    image_path: Path,
    prompt: str,
    model: str | None = None,
    max_tokens: int = 4096,
    timeout: int = 60,
) -> str:
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    selected_model = (
        str(model or "").strip()
        or (os.getenv("OPENROUTER_VISION_MODEL") or "").strip()
        or (os.getenv("OPENROUTER_MODEL") or "").strip()
    )
    if not selected_model:
        raise RuntimeError("No model configured. Set OPENROUTER_MODEL or OPENROUTER_VISION_MODEL.")

    base = _normalize_openrouter_base(os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    url = f"{base}/chat/completions"
    image_url = _image_to_data_url(image_path)

    formatted_prompt = (prompt or "").strip()
    if not formatted_prompt:
        formatted_prompt = "Analyze the image and provide a concise answer."

    payload: dict[str, Any] = {
        "model": selected_model,
        "max_tokens": int(max_tokens),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatted_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    }

    provider_order_raw = (os.getenv("OPENROUTER_PROVIDER_ORDER") or "").strip()
    provider_raw = (os.getenv("OPENROUTER_PROVIDER") or "").strip()
    allow_fallbacks = (os.getenv("OPENROUTER_ALLOW_FALLBACKS") or "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    provider_order: list[str] = []
    if provider_order_raw:
        provider_order = [p.strip() for p in provider_order_raw.split(",") if p.strip()]
    elif provider_raw:
        provider_order = [provider_raw]
    if provider_order:
        payload["provider"] = {"order": provider_order, "allow_fallbacks": allow_fallbacks}

    headers: dict[str, str] = {
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }
    site_url = (os.getenv("OPENROUTER_SITE_URL") or "").strip()
    app_name = (os.getenv("OPENROUTER_APP_NAME") or "").strip()
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=int(timeout)) as resp:
            raw_text = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(f"OpenRouter error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    response_obj = json.loads(raw_text or "{}")
    content = _extract_content_text(response_obj).strip()
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a local image with OpenRouter.")
    parser.add_argument("--image", required=True, help="Path to local image")
    parser.add_argument("--prompt", required=True, help="Question or instruction for the image")
    parser.add_argument("--model", default="", help="Override model id")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.is_absolute():
        image_path = (Path.cwd() / image_path).resolve()
    else:
        image_path = image_path.resolve()

    output = analyze_image(
        image_path=image_path,
        prompt=str(args.prompt),
        model=str(args.model or "").strip() or None,
        max_tokens=int(args.max_tokens),
        timeout=int(args.timeout),
    )
    print(output)


if __name__ == "__main__":
    main()
