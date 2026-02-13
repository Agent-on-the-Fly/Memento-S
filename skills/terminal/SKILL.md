---
name: terminal
description: Safe terminal operations using CAMEL TerminalToolkit utils (check_command_safety, sanitize_command, env setup helpers). Use for executing shell commands with safety preflight.
---

# Terminal (CAMEL TerminalToolkit utils)

## When to use
- User asks to run shell commands or scripts.
- Need to validate/sanitize a command before execution.
- Need to set up Python environments (uv/venv), clone envs, or check Node.js availability.

## Required behavior
1. Preflight every command using CAMEL utils semantics:
   - `check_command_safety(command, allowed_commands)` for a quick safety check.
   - `sanitize_command(command, use_docker_backend, safe_mode, working_dir, allowed_commands)` to get `(is_safe, message_or_command)`.
2. If unsafe, do NOT execute; return a refusal message.
3. If safe, execute the sanitized command locally with the specified working directory.
4. Keep commands scoped to the intended working directory.

## Output contract (JSON only)
Return a single JSON object — either `{"ops":[...]}` with a **non-empty** array, or `{"final":"..."}` for a text-only answer.

Never use `tool_calls`, `mcp_call`, or `mcp_tool`.

**CRITICAL — every element in `ops` MUST be a JSON object with a top-level string key `"type"`.**
Missing, misspelled, or nested `type` causes an "unknown op" error and the op is silently skipped.

```
✅  {"ops": [{"type": "run_command", "command": "ls", "working_dir": "/tmp"}]}
❌  {"ops": [{"op": "run_command", "command": "ls"}]}        ← wrong key name
❌  {"ops": [{"command": "ls"}]}                              ← missing type
❌  {"ops": [{"type": "", "command": "ls"}]}                  ← empty type
```

**IMPORTANT:** Always set `allowed_commands` to `null` (or omit it). This disables command whitelisting
and avoids false refusals (e.g., `cd` in `cmd && ...`). Safety is still enforced by
`check_command_safety` and `sanitize_command`.

### Op schema (all fields required unless noted)

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | **REQUIRED.** Must be one of the supported op types below. |
| `command` | string | Required for `run_command`. The shell command to execute. |
| `working_dir` | string | Required for `run_command` and setup ops. Absolute path to working directory. |
| `safe_mode` | boolean | Optional for `run_command`. Default: `true`. |
| `use_docker_backend` | boolean | Optional for `run_command`. Default: `false`. |
| `allowed_commands` | null | Optional. Always use `null` to disable whitelisting. |
| `timeout` | number | Optional for `run_command`. Default: `60`. |

### Example output
```json
{
  "ops": [
    {
      "type": "run_command",
      "command": "uv pip install requests",
      "working_dir": "/Users/huichi/Memento-S",
      "safe_mode": true,
      "use_docker_backend": false,
      "allowed_commands": null,
      "timeout": 60
    }
  ]
}
```

### Supported op types
| Type | Required fields |
|------|-----------------|
| `run_command` | `type`, `command`, `working_dir` |
| `ensure_uv_available` | `type` |
| `setup_initial_env_with_uv` | `type`, `env_path`, `uv_path`, `working_dir` |
| `setup_initial_env_with_venv` | `type`, `env_path`, `working_dir` |
| `clone_current_environment` | `type`, `env_path`, `working_dir` |
| `is_uv_environment` | `type` |
| `check_nodejs_availability` | `type` |

## CAMEL utils reference (from docs)
- `check_command_safety(command, allowed_commands=None) -> (bool, str)`
- `sanitize_command(command, use_docker_backend=False, safe_mode=True, working_dir=None, allowed_commands=None) -> (bool, str)`
- `is_uv_environment() -> bool`
- `ensure_uv_available(update_callback=None) -> (bool, Optional[str])`
- `setup_initial_env_with_uv(env_path, uv_path, working_dir, update_callback=None) -> str`
- `setup_initial_env_with_venv(env_path, working_dir, update_callback=None) -> str`
- `clone_current_environment(env_path, working_dir, update_callback=None) -> str`
- `check_nodejs_availability(update_callback=None) -> bool`

## Notes
- Keep responses brief and deterministic.
- **This is a uv-managed environment.** Always use `uv pip` instead of `pip` for package operations:
  - `uv pip install <package>` instead of `pip install <package>`
  - `uv pip list` instead of `pip list`
  - `uv pip uninstall <package>` instead of `pip uninstall <package>`
  - `uv pip show <package>` instead of `pip show <package>`
- `use_docker_backend=true` is validated but not executed (local execution only).
