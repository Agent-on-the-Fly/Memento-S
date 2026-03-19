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

# Upgrade a package
uv pip install --upgrade requests
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
| pptx | python-pptx |
| bs4 | beautifulsoup4 |
| dotenv | python-dotenv |
| attr | attrs |
| dateutil | python-dateutil |
| gi | pygobject |
| magic | python-magic |
| serial | pyserial |
| usb | pyusb |
| Crypto | pycryptodome |
| jwt | pyjwt |
| jose | python-jose |
| lxml | lxml |
| wx | wxPython |
| gi | PyGObject |
| skimage | scikit-image |
| fitz | PyMuPDF |
| cv | opencv-python |
| telegram | python-telegram-bot |
| discord | discord.py |
| flask_cors | flask-cors |
| sqlalchemy | SQLAlchemy |
| alembic | alembic |
| pymongo | pymongo |
| redis | redis |
| celery | celery |
| fastapi | fastapi |
| uvicorn | uvicorn |
| httpx | httpx |
| aiohttp | aiohttp |
| websockets | websockets |
| jinja2 | Jinja2 |
| markdownify | markdownify |

## Workflow

1. If an import error occurs, extract the module name.
2. Map module name to package name if different (see table above).
3. Check if already installed: `uv pip show <package>`
4. If not installed: `uv pip install <package>`

## Troubleshooting

### Common issues

| Problem | Solution |
|---------|----------|
| `uv: command not found` | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Package installs but import still fails | Check you're in the correct venv: `which python` |
| Version conflict | Pin version: `uv pip install package==1.2.3` |
| Build fails (C extension) | Install system deps: `brew install pkg-config` (macOS) |
| SSL certificate error | `uv pip install --trusted-host pypi.org package` |

## Notes

- Always run from the project directory so the correct `.venv` is used.
- Use `uv pip` (not plain `pip`) to ensure packages go into the uv-managed environment.
- For packages with C extensions, ensure build tools are installed (`xcode-select --install` on macOS).
