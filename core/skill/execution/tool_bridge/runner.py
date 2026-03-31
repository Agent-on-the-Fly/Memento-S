"""Tool execution runner (execution only, no policy/recovery)."""

from __future__ import annotations

import os
from typing import Any

from core.shared.tools_facade import execute_tool


class ToolRunner:
    async def run(
        self, tool_name: str, args: dict, env_vars: dict[str, str] | None = None
    ) -> Any:
        """Execute tool with optional environment variables.

        ENV VAR JAIL: Injects environment variables into the execution context.
        For python_repl tool, env_vars are made available via os.environ.
        """
        # ENV VAR JAIL: Temporarily inject environment variables
        if env_vars:
            old_values = {}
            try:
                # Set new environment variables
                for key, value in env_vars.items():
                    old_values[key] = os.environ.get(key)
                    os.environ[key] = value

                result = await execute_tool(tool_name, args)

            finally:
                # Restore original environment variables
                for key, old_value in old_values.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value
        else:
            result = await execute_tool(tool_name, args)

        return result
