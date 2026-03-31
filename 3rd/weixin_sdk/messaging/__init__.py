"""
Weixin SDK messaging module.

Provides slash command handling for bot debugging and control,
plus debug mode functionality for performance monitoring.

Example:
    from weixin_sdk.messaging import SlashCommandRegistry, DebugModeManager, TimingContext

    # Use slash commands
    registry = SlashCommandRegistry()
    if registry.is_command(message_text):
        response = await registry.process(message_text, context)

    # Use debug mode
    debug_manager = DebugModeManager()
    debug_manager.enable("account_123")

    with TimingContext(debug_manager, "account_123", "ai_generation"):
        ai_response = await generate_response(message)
"""

from .commands import (
    SlashCommandRegistry,
    CommandHandler,
    default_registry,
    get_default_registry,
)
from .debug_mode import (
    DebugMode,
    DebugModeManager,
    TimingContext,
    MessagePipelineTracer,
    timing_context,
    TimingRecord,
    TimingTrace,
)

__all__ = [
    "SlashCommandRegistry",
    "CommandHandler",
    "default_registry",
    "get_default_registry",
    "DebugMode",
    "DebugModeManager",
    "TimingContext",
    "MessagePipelineTracer",
    "timing_context",
    "TimingRecord",
    "TimingTrace",
]
