#!/usr/bin/env python3
"""Analyze a local image using the project's configured LLM (via litellm).

Falls back to basic Pillow metadata extraction when no LLM is available.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from pathlib import Path
from typing import Any


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


def _load_llm_config() -> dict[str, Any]:
    """Load model/API config from the project's configuration."""
    try:
        from middleware.config.config_manager import ConfigManager
        config = ConfigManager().load()
        profile = config.llm.current_profile
        return {
            "model": profile.model,
            "api_key": profile.api_key,
            "base_url": profile.base_url,
            "max_tokens": profile.max_tokens,
            "timeout": profile.timeout,
            "extra_headers": profile.extra_headers,
        }
    except Exception:
        return {}


def _build_model_string(model: str, base_url: str | None = None) -> str:
    """Build litellm model string with provider prefix."""
    if "/" in model:
        return model
    if base_url:
        if "anthropic" in base_url:
            return f"anthropic/{model}"
        elif "openrouter" in base_url:
            return f"openrouter/{model}"
        elif "openai" in base_url:
            return f"openai/{model}"
    return model


def analyze_image_basic(image_path: Path) -> str:
    """Extract basic image metadata using Pillow (no LLM needed)."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:
        return "ERROR: Pillow is not installed. Run: pip install Pillow"

    try:
        img = Image.open(image_path)
    except OSError as e:
        return f"ERROR: Could not open image file: {e}"

    info = {
        "file": str(image_path),
        "format": img.format,
        "size": f"{img.width}x{img.height}",
        "mode": img.mode,
    }

    # EXIF data
    exif_data = {}
    try:
        raw_exif = img._getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if isinstance(value, (str, int, float)):
                    exif_data[str(tag)] = value
    except Exception:
        print("WARNING: Could not extract EXIF data", file=sys.stderr)

    if exif_data:
        info["exif"] = exif_data

    # Color statistics
    try:
        if img.mode in ("RGB", "RGBA"):
            colors = img.getcolors(maxcolors=256 * 256)
            if colors:
                colors.sort(key=lambda x: x[0], reverse=True)
                top_colors = []
                for count, color in colors[:5]:
                    if isinstance(color, tuple):
                        hex_color = "#{:02x}{:02x}{:02x}".format(*color[:3])
                        top_colors.append({"color": hex_color, "count": count})
                info["dominant_colors"] = top_colors
    except Exception:
        print("WARNING: Could not extract color statistics", file=sys.stderr)

    return json.dumps(info, indent=2, ensure_ascii=False, default=str)


def analyze_image(
    *,
    image_path: Path,
    prompt: str = "Describe this image in detail.",
    model: str | None = None,
    max_tokens: int = 2048,
    timeout: int = 60,
) -> str:
    """Analyze an image using litellm with the project's LLM config."""
    try:
        import litellm
    except ImportError:
        return "ERROR: litellm is not installed. Run: pip install litellm"

    # Load project config
    config = _load_llm_config()
    if not config and not model:
        print("WARNING: No LLM config found. Falling back to basic mode.", file=sys.stderr)
        return analyze_image_basic(image_path)

    selected_model = model or config.get("model", "")
    if not selected_model:
        print("WARNING: No model configured. Falling back to basic mode.", file=sys.stderr)
        return analyze_image_basic(image_path)

    model_str = _build_model_string(selected_model, config.get("base_url"))
    image_url = _image_to_data_url(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.strip() or "Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    kwargs: dict[str, Any] = {
        "model": model_str,
        "messages": messages,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }

    api_key = config.get("api_key")
    base_url = config.get("base_url")
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["api_base"] = base_url
    extra_headers = config.get("extra_headers")
    if extra_headers:
        kwargs["extra_headers"] = extra_headers

    try:
        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        return f"ERROR: LLM call failed: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a local image.")
    parser.add_argument("--image", required=True, help="Path to local image")
    parser.add_argument("--prompt", default="Describe this image in detail.", help="Question or instruction")
    parser.add_argument("--model", default="", help="Override model id")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")
    parser.add_argument("--basic", action="store_true", help="Basic mode: metadata only, no LLM")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.is_absolute():
        image_path = (Path.cwd() / image_path).resolve()
    else:
        image_path = image_path.resolve()

    if args.basic:
        output = analyze_image_basic(image_path)
    else:
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
