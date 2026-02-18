"""Bridge submodules split from skill_executor.py."""

from .dispatcher import TOOL_PROTOCOL_VERSION, ToolCall, ToolCallResult, coerce_call_stack, dispatch_bridge_op
from .registry import ToolSchema, ToolSpec, build_tool_registry

__all__ = [
    "TOOL_PROTOCOL_VERSION",
    "ToolSchema",
    "ToolSpec",
    "ToolCall",
    "ToolCallResult",
    "build_tool_registry",
    "coerce_call_stack",
    "dispatch_bridge_op",
]

