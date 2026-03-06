---
name: image-analysis
description: Analyze local images using an OpenAI-compatible multimodal endpoint. Use when the user question depends on visual content from a local image file — visual question answering, describing images, reading text in images, identifying objects, etc.
metadata: {"requires":{"bins":["python3"],"env":["LLM_API_KEY","LLM_MODEL","LLM_BASE_URL"]}}
---

# Image Analysis

Analyze local images with OpenAI-compatible multimodal chat completions.

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
2. `LLM_MODEL` env var

## API key

Set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`.
Optionally configure `LLM_PROVIDER_ORDER`, `LLM_PROVIDER`, `LLM_ALLOW_FALLBACKS`, `LLM_SITE_URL`, and `LLM_APP_NAME`.

## Supported formats

PNG, JPEG, GIF, WebP, BMP, and other common image formats.
