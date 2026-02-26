---
name: uv-pip-install
description: Install and manage Python packages using uv pip. Use when a Python import fails with ModuleNotFoundError, user asks to install a package, or a script requires a missing dependency.
---

# uv-pip-install

Install and manage Python packages in the uv-managed virtual environment.

## Quick start

```bash
# Install a package
uv pip install requests

# Install multiple packages
uv pip install pandas numpy matplotlib

# Install with extras
uv pip install httpx[http2]

# Check if a package is installed
uv pip show python-docx

# List all installed packages
uv pip list
```

## When to use

- Python import fails with `ModuleNotFoundError`
- User asks to install a Python package
- A script requires dependencies that are not installed

## Common module-to-package mappings

| Import name | Package name |
|-------------|--------------|
| cv2 | opencv-python |
| PIL | pillow |
| sklearn | scikit-learn |
| yaml | pyyaml |
| docx | python-docx |
| bs4 | beautifulsoup4 |
| dotenv | python-dotenv |

## Workflow

1. If an import error occurs, extract the module name.
2. Map module name to package name if different (see table above).
3. Check if already installed: `uv pip show <package>`
4. If not installed: `uv pip install <package>`

## Notes

- Always run from the project directory so the correct `.venv` is used.
- Use `uv pip` (not plain `pip`) to ensure packages go into the uv-managed environment.
