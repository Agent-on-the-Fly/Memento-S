# Skill Types and Bridge Patterns

Skills are executed by the host bridge runtime:
1. LLM generates JSON using SKILL.md instructions.
2. JSON is either `{"final":"..."}` or `{"ops":[...]}`.
3. The host executes `ops` by dispatching to built-in skills.

## Universal Output Contract

Use one of these forms:

```json
{"final": "Answer text"}
```

```json
{
  "ops": [
    {"type": "call_skill", "skill": "filesystem", "ops": [{"type": "read_file", "path": "README.md"}]}
  ]
}
```

Rules:
- Return JSON only.
- Each op must have `type`.
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

Use filesystem delegation.

```json
{
  "ops": [
    {"type": "call_skill", "skill": "filesystem", "ops": [{"type": "directory_tree", "path": ".", "depth": 2}]},
    {"type": "call_skill", "skill": "filesystem", "ops": [{"type": "write_file", "path": "report.md", "content": "..."}]}
  ]
}
```

## Type 2: Command Execution Skills

Use terminal delegation.

```json
{
  "ops": [
    {
      "type": "call_skill",
      "skill": "terminal",
      "ops": [
        {"type": "run_command", "command": "git status", "working_dir": ".", "safe_mode": true}
      ]
    }
  ]
}
```

## Type 3: Web + Synthesis Skills

Use web-search for retrieval, then produce final answer or write file in a later round.

Round 1 (gather):

```json
{
  "ops": [
    {"type": "call_skill", "skill": "web-search", "ops": [{"type": "web_search", "query": "latest rust release notes", "num_results": 5}]}
  ]
}
```

Round 2 (after previous output is fed back):

```json
{"final": "Summary based on retrieved results ..."}
```

## Type 4: Dependency Installation Skills

Use uv-pip-install delegation.

```json
{
  "ops": [
    {"type": "call_skill", "skill": "uv-pip-install", "ops": [{"type": "check", "package": "pandas"}]},
    {"type": "call_skill", "skill": "uv-pip-install", "ops": [{"type": "install", "package": "pandas"}]}
  ]
}
```

## Type 5: Multi-Round Workflow Skills

When step B depends on step A output, split rounds.

- Round 1: info gathering ops only.
- Round 2: generate `final` or write ops using real round-1 output.

Avoid mixing gather + final write in one round when content depends on fetched data.

## Common Mistakes

1. Returning markdown instead of JSON.
2. Missing `type` on ops.
3. Using MCP/tool_calls formats instead of `ops`.
4. Trying to do everything in one round when later steps depend on earlier outputs.
5. Calling subprocesses for delegation instead of `call_skill`.
