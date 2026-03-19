"""
Tool registry and OpenAI function-calling schemas for the built-in tools.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .bash import bash_tool
from .file_ops import (
    list_dir_tool,
    read_file_tool,
    file_create_tool,
    edit_file_by_lines_tool,
)
from .grep import grep_tool
from .web import fetch_webpage_tool, tavily_search_tool

logger = logging.getLogger(__name__)


def get_tool_schema(tool_name: str) -> dict | None:
    """根据 tool name 获取其参数 schema。"""
    for schema in BUILTIN_TOOL_SCHEMAS:
        if schema["function"]["name"] == tool_name:
            return schema["function"]["parameters"]
    return None

# --- Tool Registry (name -> handler) ---

BUILTIN_TOOL_REGISTRY = {
    "bash": bash_tool,
    "edit_file_by_lines": edit_file_by_lines_tool,
    "fetch_webpage": fetch_webpage_tool,
    "file_create": file_create_tool,
    "grep": grep_tool,
    "list_dir": list_dir_tool,
    "read_file": read_file_tool,
    "search_web": tavily_search_tool,
}

# --- OpenAI Function-Calling Schemas (name + params) ---
BUILTIN_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the contents of a directory as a tree. Provide a valid directory path; do NOT pass content in the path field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list (default is current workspace). Must be a directory path, not content.",
                        "default": ".",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum recursion depth (default 2).",
                        "default": 2,
                    },
                    "base_dir": {
                        "type": "string",
                        "description": "Optional base directory for path resolution.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file with line numbers. Provide a valid file path; do NOT pass file content in the path field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read. Must be a file path, not file content.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed, default 1).",
                        "default": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Line number to stop reading (inclusive). Use -1 to read to the end.",
                        "default": -1,
                    },
                    "base_dir": {
                        "type": "string",
                        "description": "Optional base directory for path resolution.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a regex pattern in files OR in a text string. If 'text' is provided, search within that text. Otherwise, search across files in 'dir_path'. Use this to find function definitions, variables, or search within loaded content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": 'The Python regex pattern to search for (e.g., "def process_data", "error|Error").',
                    },
                    "dir_path": {
                        "type": "string",
                        "description": "The directory to search in (used when text is not provided).",
                        "default": ".",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": 'Glob pattern to filter files (e.g., "*.py", "*.ts", default is "*").',
                        "default": "*",
                    },
                    "text": {
                        "type": "string",
                        "description": "If provided, search within this text string instead of files. Useful for searching in fetched content, command output, or loaded file content.",
                    },
                    "show_line_numbers": {
                        "type": "boolean",
                        "description": "Whether to show line numbers in results (default True). Only applies when searching text.",
                        "default": True,
                    },
                    "base_dir": {
                        "type": "string",
                        "description": "Optional base directory for path resolution.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetch a webpage and convert its main content to clean Markdown. Use this to read documentation, news, or API references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The HTTP/HTTPS URL to fetch.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web using Tavily API and return LLM-friendly markdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keep under 400 chars).",
                    },
                    "search_depth": {
                        "type": "string",
                        "description": "Search depth: ultra-fast | fast | basic | advanced.",
                        "default": "basic",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results (0-20).",
                        "default": 3,
                    },
                    "include_raw_content": {
                        "type": "boolean",
                        "description": "Include raw page content when available.",
                        "default": False,
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_create",
            "description": "Create a new file. Provide a valid file path; do NOT pass file content in the path field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the new file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The initial content to write into the file.",
                        "default": "",
                    },
                    "base_dir": {
                        "type": "string",
                        "description": "Optional base directory for path resolution.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file_by_lines",
            "description": "Replace specific lines in a file with new content. This is extremely robust. To INSERT code, replace a line with itself + new code. To DELETE lines, pass an empty string to new_content. IMPORTANT: You must ensure the indentation of new_content matches the original file!",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file. Must be a file path, not file content.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "The first line number to replace (1-indexed).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "The last line number to replace (inclusive).",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The exact new text to put in place of the replaced lines.",
                    },
                    "base_dir": {
                        "type": "string",
                        "description": "Optional base directory for path resolution.",
                    },
                },
                "required": ["path", "start_line", "end_line", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command in the run_dir. IMPORTANT: This is a STATELESS environment. Environment variables or `cd` will not persist across calls. Use `&&` to chain commands (e.g., `cd src && ls`). Interactive commands (like vim, nano, top) are strictly prohibited.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    },
                    "stdin": {
                        "type": "string",
                        "description": "Optional standard input passed to the command.",
                    },
                    "base_dir": {
                        "type": "string",
                        "description": "Optional base directory for path resolution (security boundary).",
                    },
                    "work_dir": {
                        "type": "string",
                        "description": "Optional working directory for command execution.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

def is_builtin_tool(name: str) -> bool:
    """Return True if name is a built-in tool."""
    return name in BUILTIN_TOOL_REGISTRY


def get_tools_summary() -> str:
    """从 BUILTIN_TOOL_SCHEMAS 生成工具摘要，供 prompt 使用。"""
    lines = []
    for schema in BUILTIN_TOOL_SCHEMAS:
        func = schema["function"]
        name = func["name"]
        desc = func["description"].split(".")[0]
        lines.append(f"- `{name}` — {desc}")
    return "\n".join(lines)


async def execute_builtin_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> str:
    """Execute a built-in tool by name with the given arguments."""
    arguments.pop("description", None)
    if base_dir is not None and "base_dir" not in arguments:
        arguments["base_dir"] = str(base_dir)

    fn = BUILTIN_TOOL_REGISTRY.get(name)
    if fn is None:
        return f"ERR: unknown builtin tool '{name}'"
    return await fn(**arguments)
