# SPDX-License-Identifier: Apache-2.0
"""PathPolicyHook — 分层路径策略 hook（BEFORE_TOOL_EXEC）。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from shared.hooks.executor import HookDefinition
from shared.hooks.types import HookPayload, HookResult
from utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class PathPolicyHook(HookDefinition):
    """
    分层路径策略 hook — BEFORE_TOOL_EXEC

    策略：
    - LOOSE 工具（bash, python_repl）：不做路径重写，只做黑名单警告
    - STRICT 工具（file_create, read_file 等）：检查路径是否落在系统目录
    """

    STRICT_TOOLS: frozenset[str] = frozenset({
        "file_create", "write_file", "read_file", "edit_file",
        "grep_file", "glob", "copy_file", "move_file",
        "find_files", "find_replace",
    })

    LOOSE_TOOLS: frozenset[str] = frozenset({
        "bash", "python_repl",
    })

    SYSTEM_PATH_PREFIXES: tuple[str, ...] = (
        "/etc/", "/usr/", "/bin/", "/sbin/", "/sys/", "/proc/",
        "/root/", "/.ssh/", "/.config/",
    )

    async def execute(self, payload: HookPayload) -> HookResult:
        tool_name = payload.tool_name
        args = payload.args

        if tool_name in self.LOOSE_TOOLS:
            return self._handle_loose_tool(tool_name, args)
        elif tool_name in self.STRICT_TOOLS:
            return self._handle_strict_tool(tool_name, args)

        return HookResult(allowed=True)

    def _handle_loose_tool(self, tool_name: str, args: dict[str, Any]) -> HookResult:
        """宽松模式：bash/python_repl 保持原始路径，只做黑名单警告。"""
        for key, value in args.items():
            if key == "command" and isinstance(value, str):
                # 检测 bash 命令中是否有访问系统目录的路径
                violations = self._check_command_system_paths(value)
                for path in violations:
                    logger.warning(
                        f"[PathPolicy] LOOSE tool '{tool_name}' accessing system path: {path}"
                    )
        return HookResult(allowed=True)

    def _handle_strict_tool(self, tool_name: str, args: dict[str, Any]) -> HookResult:
        """严格模式：检查结构化路径参数是否落在系统目录。"""
        for key, value in args.items():
            if isinstance(value, str):
                # 检查字符串值是否是以系统目录开头的路径
                for prefix in self.SYSTEM_PATH_PREFIXES:
                    if value.startswith(prefix) or value.startswith(f"/private{prefix}"):
                        logger.warning(
                            f"[PathPolicy] STRICT tool '{tool_name}' accessing system path: {value}"
                        )
                        return HookResult(
                            allowed=False,
                            reason=f"Path '{value}' is a system directory and not allowed for tool '{tool_name}'"
                        )
        return HookResult(allowed=True)

    def _check_command_system_paths(self, command: str) -> list[str]:
        """检测命令中访问系统目录的路径。"""
        violations = []
        for prefix in self.SYSTEM_PATH_PREFIXES:
            # 匹配命令行中的绝对路径
            pattern = rf'({re.escape(prefix)}[^\s"\']*)'
            matches = re.findall(pattern, command)
            violations.extend(matches)
        return violations
