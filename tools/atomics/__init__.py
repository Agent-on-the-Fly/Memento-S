# SPDX-License-Identifier: Apache-2.0
"""Atomic tools layer — stateless, pure async callables.

Each tool is a simple async function with an attached _schema attribute
for OpenAI function-calling compatibility.
"""

from __future__ import annotations

from tools.atomics.list_dir import list_dir
from tools.atomics.read_file import read_file
from tools.atomics.file_create import file_create
from tools.atomics.edit_file_by_lines import edit_file_by_lines
from tools.atomics.grep import grep
from tools.atomics.bash import bash
from tools.atomics.python_repl import python_repl
from tools.atomics.js_repl import js_repl
from tools.atomics.web import search_web, fetch_webpage
from tools.atomics.glob import glob
from tools.atomics.mcp_tools import mcp_list_resources, mcp_read_resource

__all__ = [
    "list_dir",
    "read_file",
    "file_create",
    "edit_file_by_lines",
    "grep",
    "bash",
    "python_repl",
    "js_repl",
    "search_web",
    "fetch_webpage",
    "glob",
    "mcp_list_resources",
    "mcp_read_resource",
]
