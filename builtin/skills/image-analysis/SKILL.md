---
name: image-analysis
description: Analyze local images using vision-capable LLM. Use when the user question depends on visual content from a local image file — visual question answering, describing images, reading text in images, identifying objects, etc.
metadata:
  dependencies:
    - Pillow
    - litellm
---

# Image Analysis

Analyze local images using the project's configured LLM (via litellm). Works with any vision-capable model (GPT-4o, Claude 3, Gemini, etc.).

## Quick start

```bash
# Analyze an image with a question
python3 scripts/analyze_image.py --image "/path/to/image.png" --prompt "Describe what you see in the image"

# Use a specific model (overrides project config)
python3 scripts/analyze_image.py --image "/path/to/photo.jpg" --prompt "What text is visible?" --model "openai/gpt-4o"

# Basic mode: extract image metadata without LLM (no API key needed)
python3 scripts/analyze_image.py --image "/path/to/image.png" --basic

# Increase output length and timeout
python3 scripts/analyze_image.py --image "/path/to/diagram.png" --prompt "Explain this diagram" --max-tokens 4096 --timeout 120
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--image` | Path to local image file (required) | — |
| `--prompt` | Question or instruction for the image | "Describe this image in detail" |
| `--model` | Override model id (e.g. `openai/gpt-4o`) | project config |
| `--max-tokens` | Max output tokens | `2048` |
| `--timeout` | HTTP timeout in seconds | `60` |
| `--basic` | Extract image metadata only (no LLM needed) | off |

## Model configuration

The script reads model/API configuration from the project's config (`middleware/config`). Ensure your configured model supports vision (multimodal) input.

Override with `--model` to use a specific model for this call.

## Basic mode

When `--basic` is used (or when no LLM is configured), the script uses Pillow to extract:
- Image format, size, color mode
- EXIF metadata (camera, date, GPS if available)
- Color statistics (dominant colors, histogram)

## Supported formats

PNG, JPEG, GIF, WebP, BMP, TIFF, and other common image formats.
