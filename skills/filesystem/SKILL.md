---
name: filesystem
description: Direct filesystem operations (read, write, edit, list, search files and directories). Use for any file manipulation tasks including reading file contents, writing or overwriting files, editing/replacing text in files, copying, moving, deleting files, listing directories, building directory trees, and searching with glob patterns. Also use when the user asks to view, show, display, or inspect files or folder structures.
---

# Filesystem Skill

Direct filesystem operations. No external dependencies.

## RESPONSE FORMAT (MANDATORY)

Return ONLY a single valid JSON object — no markdown, no prose, no explanation.

Return one of:

`{"ops": [<op>, ...]}` — for filesystem actions  
`{"final": "text"}` — only when genuinely no filesystem action is needed

### ★ THE ONE RULE: every op MUST have `"type": "<string>"`

The runtime reads `op["type"]` to dispatch. **If `"type"` is missing or not a string, the op is silently dropped and returns an error.**

- The key MUST be the literal string `"type"` — not `"op"`, not `"action"`, not `"name"`, not `"tool"`.
- The value MUST be a non-empty string from the valid list below.
- Every op object in the `"ops"` array MUST independently contain its own `"type"` key.
- Do NOT use wrapper types like `"mcp_call"`, `"mcp_tool"`, or `"call_skill"` with `"skill": "filesystem"` — emit direct ops only.

**Self-check before returning:** scan every `{...}` in your `"ops"` array. If any object lacks a literal `"type"` key with a string value, add it.

**Template — copy this shape for every op:**
```json
{"type": "<operation>", ...params}
```

**Valid `"type"` values (exhaustive, case-sensitive):**  
`read_file` · `write_file` · `edit_file` · `append_file` · `list_directory` · `directory_tree` · `create_directory` · `move_file` · `copy_file` · `delete_file` · `file_info` · `search_files` · `file_exists`

### JSON schema

```
{
  "ops": [
    { "type": "<operation>", ...params },
    { "type": "<operation>", ...params }
  ]
}
```

**Top-level rules:**
- Allowed top-level keys: **only** `"ops"` or `"final"`. Nothing else.
- `"ops"` is a flat JSON array of op objects. No nesting, no wrappers.
- Every element in `"ops"` MUST be an object with a `"type"` string key.

### Forbidden patterns (all cause silent failures)

**Missing or wrong key name — op is skipped:**
- `{"op": "read_file", ...}` — wrong key (must be `"type"`)
- `{"action": "read_file", ...}` — wrong key
- `{"name": "read_file", ...}` — wrong key
- `{"tool": "read_file", ...}` — wrong key
- `{"path": "/file.txt"}` — `"type"` missing entirely

**Invalid type values:**
- `{"type": "mcp_call", ...}` — wrapper, not a direct op
- `{"type": "mcp_tool", ...}` — wrapper, not a direct op
- `{"type": "call_skill", "skill": "filesystem", ...}` — no self-delegation

**Wrong top-level structure:**
- `{"tool_calls": [...]}` — must use `"ops"`, not `"tool_calls"`
- Extra keys like `thoughts`, `code`, `reasoning` — only `"ops"` or `"final"` allowed

### Correct examples

Read a file:
```json
{"ops": [{"type": "read_file", "path": "/home/user/file.txt"}]}
```

Tree view:
```json
{"ops": [{"type": "directory_tree", "path": "/project", "depth": 2}]}
```

Read then edit:
```json
{"ops": [
  {"type": "read_file", "path": "/app/main.py"},
  {"type": "edit_file", "path": "/app/main.py", "old_text": "foo", "new_text": "bar"}
]}
```

Write a new file:
```json
{"ops": [{"type": "write_file", "path": "/tmp/out.txt", "content": "hello world"}]}
```

Multiple reads:
```json
{"ops": [
  {"type": "read_file", "path": "/a.txt"},
  {"type": "read_file", "path": "/b.txt"},
  {"type": "read_file", "path": "/c.txt"}
]}
```

List then read:
```json
{"ops": [
  {"type": "list_directory", "path": "/data/conversations"},
  {"type": "read_file", "path": "/data/conversations/abc.json"}
]}
```

### Wrong (will fail)

```json
{"ops": [{"op": "read_file", "path": "..."}]}
```
↑ `"op"` instead of `"type"` — op will be silently skipped

```json
{"ops": [{"name": "read_file", "path": "..."}]}
```
↑ `"name"` instead of `"type"` — op will be silently skipped

```json
{"ops": [{"path": "/file.txt"}]}
```
↑ missing `"type"` entirely — op will be silently skipped

```json
{"ops": [{"type": "mcp_call", "tool": "read_file"}]}
```
↑ wrapper type, not a direct op

```json
{"tool_calls": [...]}
```
↑ wrong top-level key

## Operations

| type | required params | optional | description |
|------|----------------|----------|-------------|
| `read_file` | `path` | `head`, `tail` | Read file content |
| `write_file` | `path`, `content` | | Write/overwrite file |
| `edit_file` | `path`, `old_text`, `new_text` | `dry_run` | Replace first occurrence of old_text |
| `append_file` | `path`, `content` | | Append to file |
| `list_directory` | `path` | | List directory entries |
| `directory_tree` | `path` | `depth` (default 3) | Recursive tree view |
| `create_directory` | `path` | | Create directory (with parents) |
| `move_file` | `src`, `dst` | | Move/rename |
| `copy_file` | `src`, `dst` | | Copy file or directory |
| `delete_file` | `path` | | Delete file or directory |
| `file_info` | `path` | | Get file metadata |
| `search_files` | `path`, `pattern` | | Glob search |
| `file_exists` | `path` | | Check existence |

## Key Behavior Notes

- When asked to read/show/display a file, emit `{"ops":[{"type":"read_file","path":"..."}]}`. The runtime executes the op and returns the file contents.
- Paths: absolute or relative to working_dir. Parent dirs auto-created for writes.
- Aliases accepted by runtime: `old`→`old_text`, `new`→`new_text`, `replace_text`→`edit_file`, `mkdir`→`create_directory`, `rm`→`delete_file`, `mv`→`move_file`, `cp`→`copy_file`.
- For multi-step tasks, emit all ops in a single `"ops"` array. The runtime executes them sequentially.
- When you need to read a file to decide what to do next, return ONLY the read op. After seeing the content in the next round, return follow-up ops.
- For complete file replacement, use `write_file` — do NOT use `edit_file` for full rewrites.
- For `edit_file`, `old_text` must match exactly (including whitespace/newlines). If unsure, `read_file` first.
- If you are delegated to from another skill via `call_skill`, the ops format is identical — every op still needs `"type"`.

## Final Checklist (run mentally before returning)

1. Is the response a single JSON object with only `"ops"` or `"final"` at top level?
2. Is `"ops"` a flat array (not nested)?
3. Does **every** object in `"ops"` have a key literally named `"type"` (not `"op"` or `"action"`) with a non-empty string value?
4. Are all `"type"` values from the valid operations list above?
5. Are there zero extra top-level keys (`thoughts`, `reasoning`, `code`, `tool_calls`, etc.)?
6. If editing a file, does `old_text` match the file content exactly (read first if unsure)?
