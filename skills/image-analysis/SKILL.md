---
name: image-analysis
description: Analyze local images using OpenRouter vision models. Use when the user question depends on visual content from a local image file — visual question answering, describing images, reading text in images, identifying objects, etc.
metadata: {"requires":{"bins":["python3"],"env":["OPENROUTER_API_KEY"]}}
---

# Image Analysis

Analyze local images with OpenRouter multimodal chat completions.

## Quick start

```bash
# Analyze an image with a question
python3 {baseDir}/scripts/analyze_image.py --image "/path/to/image.png" --prompt "Describe what you see in the image"

# Use a specific model
python3 {baseDir}/scripts/analyze_image.py --image "/path/to/photo.jpg" --prompt "What text is visible?" --model "google/gemini-2.0-flash-001"

# Increase output length and timeout
python3 {baseDir}/scripts/analyze_image.py --image "/path/to/diagram.png" --prompt "Explain this diagram" --max-tokens 4096 --timeout 120
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--image` | Path to local image file (required) | — |
| `--prompt` | Question or instruction for the image (required) | — |
| `--model` | Override model id | env-configured |
| `--max-tokens` | Max output tokens | `2048` |
| `--timeout` | HTTP timeout in seconds | `60` |

## Model selection

Model is resolved in this order:
1. `--model` argument
2. `OPENROUTER_VISION_MODEL` env var
3. `OPENROUTER_MODEL` env var

## API key

Set `OPENROUTER_API_KEY` env var. Optionally configure `OPENROUTER_BASE_URL`, `OPENROUTER_PROVIDER_ORDER`, and `OPENROUTER_ALLOW_FALLBACKS`.

## Supported formats

PNG, JPEG, GIF, WebP, BMP, and other common image formats.
