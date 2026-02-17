# Skill Types and Bridge Patterns

Skills are executed by the host bridge runtime:
1. LLM generates JSON using SKILL.md instructions.
2. JSON is either `{"final":"..."}` or `{"tool_calls":[...]}`.
3. The host executes tool calls by dispatching to built-in skills.

## Universal Output Contract

Use one of these forms:

```json
{"final":"Answer text"}
```

```json
{
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "call_skill",
        "arguments": "{\"skill\":\"filesystem\",\"plan\":{\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}}]}}"
      }
    }
  ]
}
```

Rules:
- Return JSON only.
- Use OpenAI tool-call shape (`type=function`, `function.name`, `function.arguments` JSON string).
- Prefer `call_skill` for delegation instead of direct subprocess commands.

## Portable Bundled Resource Paths (General Compatibility)

For command-execution skills, prefer skill-local relative paths when invoking bundled files:

- `python scripts/my_tool.py ...`
- `node scripts/build.js ...`
- `bash scripts/run.sh ...`

Runtime behavior:
- Relative paths under `scripts/`, `references/`, `assets/`, `templates/`, and `examples/` are resolved against the active skill directory.
- Other relative paths remain relative to `working_dir` (or project root when omitted).

Guideline:
- Use skill-local relative paths for bundled resources.
- Use explicit `working_dir` and explicit output paths for user/project artifacts.

## Type 1: File-Centric Skills

Delegate to `filesystem` via `call_skill` and place filesystem actions in `plan.tool_calls`.

Example call_skill arguments payload:

```json
{
  "skill": "filesystem",
  "plan": {
    "tool_calls": [
      {
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "directory_tree",
          "arguments": "{\"path\":\".\",\"depth\":2}"
        }
      }
    ]
  }
}
```

## Type 2: Command Execution Skills

Delegate to `terminal` via `call_skill`.

Example call_skill arguments payload:

```json
{
  "skill": "terminal",
  "plan": {
    "tool_calls": [
      {
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "run_command",
          "arguments": "{\"command\":\"git status\",\"working_dir\":\".\",\"safe_mode\":true}"
        }
      }
    ]
  }
}
```

## Type 3: Web + Synthesis Skills

Use web-search for retrieval, then produce final answer or write file in a later round.

Round 1 (gather):

```json
{
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "call_skill",
        "arguments": "{\"skill\":\"web-search\",\"plan\":{\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"latest rust release notes\\\",\\\"num_results\\\":5}\"}}]}}"
      }
    }
  ]
}
```

Round 2 (after previous output is fed back):

```json
{"final":"Summary based on retrieved results ..."}
```

## Type 4: Dependency Installation Skills

Delegate to `uv-pip-install` via `call_skill`.

Example call_skill arguments payload:

```json
{
  "skill": "uv-pip-install",
  "plan": {
    "tool_calls": [
      {
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "check",
          "arguments": "{\"package\":\"pandas\"}"
        }
      },
      {
        "id": "call_2",
        "type": "function",
        "function": {
          "name": "install",
          "arguments": "{\"package\":\"pandas\"}"
        }
      }
    ]
  }
}
```

## Type 5: Multi-Round Workflow Skills

When step B depends on step A output, split rounds.

- Round 1: info gathering tool calls only.
- Round 2: generate `final` or write tool calls using real round-1 output.

Avoid mixing gather + final write in one round when content depends on fetched data.

## Common Mistakes

1. Returning markdown instead of JSON.
2. Missing `function.name` in tool calls.
3. Using MCP wrappers (`mcp_call` / `mcp_tool`).
4. Trying to do everything in one round when later steps depend on earlier outputs.
5. Calling subprocesses for delegation instead of `call_skill`.
