---
name: filesystem
description: Direct filesystem operations (read, write, edit, list, search files and directories). Use for any file manipulation tasks including reading file contents, writing or overwriting files, editing/replacing text in files, copying, moving, deleting files, listing directories, building directory trees, and searching with glob patterns. Also use when the user asks to view, show, display, or inspect files or folder structures.
---

# Filesystem Skill

Direct filesystem operations. No external dependencies.

## RESPONSE FORMAT (MANDATORY)

Return ONLY one JSON object (no markdown, no prose):

- `{"tool_calls":[...]}` for filesystem actions
- `{"final":"..."}` only if no filesystem action is needed

## TOOL_CALLS CONTRACT

Each element in `tool_calls` MUST use OpenAI function-call shape:

```json
{
  "id": "call_1",
  "type": "function",
  "function": {
    "name": "read_file",
    "arguments": "{\"path\":\"/tmp/a.txt\"}"
  }
}
```

Rules:
- `type` MUST be `"function"`.
- `function.name` MUST be one supported operation name.
- `function.arguments` MUST be a JSON string encoding an object.
- Do not emit `mcp_call` / `mcp_tool` wrappers.
- Do not emit `call_skill` with `skill=filesystem` from inside this skill.

## Supported Function Names

| function.name | required arguments | optional arguments | description |
|---|---|---|---|
| `read_file` | `path` | `head`, `tail` | Read file content |
| `write_file` | `path`, `content` | | Write/overwrite file |
| `edit_file` | `path`, `old_text`, `new_text` | `dry_run` | Replace first occurrence |
| `append_file` | `path`, `content` | | Append to file |
| `list_directory` | `path` | | List directory entries |
| `directory_tree` | `path` | `depth` | Tree view |
| `create_directory` | `path` | | Create directory (with parents) |
| `move_file` | `src`, `dst` | | Move/rename |
| `copy_file` | `src`, `dst` | | Copy file or directory |
| `delete_file` | `path` | | Delete file or directory |
| `file_info` | `path` | | File metadata |
| `search_files` | `path`, `pattern` | | Glob search |
| `file_exists` | `path` | | Check existence |

Runtime aliases still accepted:
- `replace_text` -> `edit_file`
- `mkdir` -> `create_directory`
- `rm` -> `delete_file`
- `mv` -> `move_file`
- `cp` -> `copy_file`
- `old` -> `old_text`
- `new` -> `new_text`

## Correct Examples

Read a file:

```json
{
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"path\":\"/home/user/file.txt\"}"
      }
    }
  ]
}
```

Tree view:

```json
{
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "directory_tree",
        "arguments": "{\"path\":\"/project\",\"depth\":2}"
      }
    }
  ]
}
```

Read then edit:

```json
{
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"path\":\"/app/main.py\"}"
      }
    },
    {
      "id": "call_2",
      "type": "function",
      "function": {
        "name": "edit_file",
        "arguments": "{\"path\":\"/app/main.py\",\"old_text\":\"foo\",\"new_text\":\"bar\"}"
      }
    }
  ]
}
```

## Wrong Examples (will fail)

```json
{"ops":[{"type":"read_file","path":"..."}]}
```

```json
{"tool_calls":[{"type":"read_file","path":"..."}]}
```

```json
{"tool_calls":[{"type":"function","function":{"name":"read_file","arguments":{"path":"/a.txt"}}}]}
```

## Behavior Notes

- When asked to read/show/display a file, call `read_file`.
- Paths may be absolute or relative to `working_dir`.
- Parent directories are auto-created for writes.
- For full rewrites, use `write_file` (not `edit_file`).
- For `edit_file`, `old_text` must match exactly; read first if unsure.
- When step B depends on step A output, do multi-round execution:
  - Round 1: gather data (`read_file`, `list_directory`, etc.)
  - Round 2: return follow-up `tool_calls` or `final`
