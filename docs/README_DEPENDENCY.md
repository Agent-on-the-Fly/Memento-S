# Automatic Dependency Installation

## Overview

Skills can declare Python dependencies in their `SKILL.md` metadata. The system automatically checks and installs missing dependencies before skill execution.

## How It Works

### 1. Dependency Declaration

In `SKILL.md`:

```yaml
---
name: xlsx
description: "Excel file processing skill"
metadata:
  dependencies:
    - openpyxl
    - pandas
    - pip:python-pptx  # Explicit pip package
    - py:cv2           # Import name (auto-resolved to opencv-python)
    - cli:ffmpeg       # CLI tool (checked but not auto-installed)
---
```

### 2. Automatic Installation Flow

```
Skill Execution Request
    ↓
SkillGateway.execute()
    ↓
run_pre_execute_gate() ✓
    ↓
[NEW] Dependency Check & Install
    ├─ Check missing dependencies
    ├─ Normalize package names (cv2 → opencv-python)
    ├─ Install via pip (if missing)
    └─ Warn about missing CLI tools
    ↓
SkillAgent.run()
```

### 3. Dependency Spec Formats

| Format | Example | Description |
|--------|---------|-------------|
| Plain | `pandas` | Auto-detected as Python package |
| `pip:` | `pip:python-pptx` | Explicit pip package |
| `py:` | `py:cv2` | Import name (resolved via aliases) |
| `cli:` | `cli:ffmpeg` | CLI tool (checked, not installed) |

### 4. Alias Resolution

Common import-to-package mappings:

```python
cv2 → opencv-python
pil → Pillow
docx → python-docx
pptx → python-pptx
sklearn → scikit-learn
bs4 → beautifulsoup4
```

See `shared/tools/dependency_aliases.py` for full list.

## Installation Behavior

### Python Packages

- **Auto-installed** using `pip install --quiet`
- Installed in current Python environment (`.venv` if exists)
- 5-minute timeout per installation
- Failures block skill execution with `DEPENDENCY_MISSING` error

### CLI Tools

- **Not auto-installed** (requires system-level permissions)
- Checked for availability in PATH
- Missing tools generate warnings but don't block execution

## Error Handling

If dependency installation fails:

```json
{
  "ok": false,
  "status": "blocked",
  "error_code": "DEPENDENCY_MISSING",
  "summary": "Failed to install dependencies: ...",
  "diagnostics": {
    "error_type": "dependency_error",
    "error_detail": {
      "missing_dependencies": ["package1", "package2"],
      "error_message": "...",
      "stage": "dependency_installation"
    }
  }
}
```

## Performance Considerations

- **First run**: Dependencies installed (may take 10-60s)
- **Subsequent runs**: Dependencies already satisfied (instant check)
- **Caching**: Installed packages persist in `.venv`

## Disabling Auto-Install

To disable for a specific skill, remove `dependencies` from metadata:

```yaml
---
name: my-skill
# No dependencies field = no auto-install
---
```

## Testing

```python
from middleware.sandbox.uv import UvLocalSandbox

# The normal application bootstrap loads g_config before constructing a sandbox.
sandbox = UvLocalSandbox()
success, error = sandbox.install_python_deps(['pandas', 'openpyxl'])

print(f"Success: {success}")
if not success:
    print(f"Error: {error}")
```

## Implementation Files

- `middleware/sandbox/uv.py` — Local sandbox environment and dependency installation
- `core/skill/gateway.py` — Integration point (before SkillAgent.run)
- `core/skill/execution/policy/pre_execute.py` — Dependency checking
- `shared/tools/dependency_aliases.py` — Import name resolution
