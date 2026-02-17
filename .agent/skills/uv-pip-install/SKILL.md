---
name: uv-pip-install
description: Install missing Python packages using uv pip. Use when (1) a Python import fails with ModuleNotFoundError, (2) user asks to install a Python package, (3) a script requires a missing dependency. This skill automatically detects missing libraries and installs them in the uv-managed .venv environment.
---

# uv-pip-install

Install Python packages in the uv-managed virtual environment.

## When to use

- User encounters `ModuleNotFoundError: No module named 'xxx'`
- User asks to install a Python package
- A script requires dependencies that are not installed
- Need to check if a package is installed

## Workflow

1. If user reports an import error, extract the module name
2. Map module name to package name if different (e.g., `cv2` -> `opencv-python`)
3. Check if package is already installed using `uv pip show`
4. If not installed, install using `uv pip install`

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

## Output contract (JSON only)

Return a single JSON object with `tool_calls` array:

```json
{
  "working_dir": "/path/to/project",
  "tool_calls": [
    { "id": "call_1", "type": "function", "function": { "name": "check", "arguments": "{\"package\":\"package-name\"}" } },
    { "id": "call_2", "type": "function", "function": { "name": "install", "arguments": "{\"package\":\"package-name\"}" } },
    { "id": "call_3", "type": "function", "function": { "name": "install", "arguments": "{\"package\":\"package-name\",\"extras\":\"[extra1,extra2]\"}" } },
    { "id": "call_4", "type": "function", "function": { "name": "list", "arguments": "{}" } }
  ]
}
```

### Supported function names

- `check`: Check if a package is installed
  - `package` (required): Package name to check
- `install`: Install a package
  - `package` (required): Package name to install
  - `extras` (optional): Extras to install, e.g., "[dev,test]"
- `list`: List all installed packages

## Examples

### Example 1: ModuleNotFoundError for docx

User reports: `ModuleNotFoundError: No module named 'docx'`

```json
{
  "working_dir": "/Users/zhou/Memento-S",
  "tool_calls": [
    { "id": "call_1", "type": "function", "function": { "name": "check", "arguments": "{\"package\":\"python-docx\"}" } },
    { "id": "call_2", "type": "function", "function": { "name": "install", "arguments": "{\"package\":\"python-docx\"}" } }
  ]
}
```

### Example 2: Install a package with extras

```json
{
  "working_dir": "/Users/zhou/Memento-S",
  "tool_calls": [
    { "id": "call_1", "type": "function", "function": { "name": "install", "arguments": "{\"package\":\"httpx\",\"extras\":\"[http2]\"}" } }
  ]
}
```

### Example 3: Check installed packages

```json
{
  "working_dir": "/Users/zhou/Memento-S",
  "tool_calls": [
    { "id": "call_1", "type": "function", "function": { "name": "list", "arguments": "{}" } }
  ]
}
```

## Notes

- Always use the project's working directory to ensure .venv is found
- The skill automatically finds and uses the .venv in the project directory
- All commands use `uv pip` to ensure packages are installed in the correct environment
