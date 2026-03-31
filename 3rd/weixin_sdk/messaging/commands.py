"""
Slash command handling for Weixin SDK bot debugging and control.

Provides command registry and built-in commands like /echo, /toggle-debug, /status, /help.

Example usage:
    from weixin_sdk.messaging.commands import SlashCommandRegistry

    registry = SlashCommandRegistry()

    # Register custom command
    @registry.register("/custom")
    async def handle_custom(args, context):
        return f"Custom command received: {args}"

    # Process message
    if registry.is_command("/echo hello"):
        response = await registry.process("/echo hello", context)
        await client.send_text(user_id, response)
"""

import asyncio
import inspect
import logging
import time
from typing import Callable, Dict, List, Optional, Any, Union
from functools import wraps

logger = logging.getLogger(__name__)

# Type alias for command handler
CommandHandler = Callable[[List[str], Dict[str, Any]], Union[str, None, Any]]


class SlashCommandRegistry:
    """
    Registry for slash commands in the Weixin bot.

    Supports both sync and async handlers. Built-in commands are automatically
    registered on instantiation.

    Attributes:
        _handlers: Dictionary mapping command names to handler functions
        _descriptions: Dictionary mapping command names to descriptions
        _command_usage: Dictionary tracking command usage statistics
    """

    COMMAND_PREFIX = "/"

    def __init__(self):
        """Initialize the command registry with built-in commands."""
        self._handlers: Dict[str, CommandHandler] = {}
        self._descriptions: Dict[str, str] = {}
        self._command_usage: Dict[str, int] = {}
        self._start_time = time.time()

        # Register built-in commands
        self._register_builtin_commands()

    def _register_builtin_commands(self) -> None:
        """Register all built-in commands."""
        # Register with manual decorator pattern to set descriptions
        self._register_with_description(
            "/echo", self._handle_echo, "Echo message back with timing stats"
        )
        self._register_with_description(
            "/toggle-debug", self._handle_toggle_debug, "Toggle debug mode on/off"
        )
        self._register_with_description(
            "/status", self._handle_status, "Show bot status information"
        )
        self._register_with_description(
            "/help", self._handle_help, "Show available commands"
        )

    def _register_with_description(
        self, command: str, handler: CommandHandler, description: str
    ) -> None:
        """Register a command with its description."""
        self._handlers[command] = handler
        self._descriptions[command] = description
        self._command_usage[command] = 0

    def register(
        self,
        command: str,
        handler: Optional[CommandHandler] = None,
        description: str = "",
    ) -> Union[Callable, CommandHandler]:
        """
        Register a command handler.

        Can be used as a decorator or as a regular method:
            # As decorator
            @registry.register("/custom")
            async def handle_custom(args, context):
                return "Custom response"

            # As method
            registry.register("/custom", handle_custom, "Description")

        Args:
            command: Command name (e.g., "/custom")
            handler: Handler function (optional when used as decorator)
            description: Optional description for help text

        Returns:
            The handler function (when used as decorator) or None

        Raises:
            ValueError: If command doesn't start with /
        """
        if not command.startswith(self.COMMAND_PREFIX):
            raise ValueError(
                f"Command must start with '{self.COMMAND_PREFIX}': {command}"
            )

        def decorator(func: CommandHandler) -> CommandHandler:
            self._handlers[command] = func
            self._descriptions[command] = description or f"Custom command: {command}"
            self._command_usage[command] = 0
            logger.debug(f"Registered command: {command}")
            return func

        if handler is not None:
            # Direct registration
            self._handlers[command] = handler
            self._descriptions[command] = description or f"Custom command: {command}"
            self._command_usage[command] = 0
            return handler

        # Return decorator for @syntax
        return decorator

    def unregister(self, command: str) -> bool:
        """
        Unregister a command.

        Args:
            command: Command name to unregister

        Returns:
            True if command was found and removed, False otherwise
        """
        if command in self._handlers:
            del self._handlers[command]
            del self._descriptions[command]
            del self._command_usage[command]
            logger.debug(f"Unregistered command: {command}")
            return True
        return False

    def is_command(self, text: str) -> bool:
        """
        Check if text is a slash command.

        Args:
            text: Text to check

        Returns:
            True if text starts with command prefix
        """
        if not text or not isinstance(text, str):
            return False
        return text.strip().startswith(self.COMMAND_PREFIX)

    def parse_command(self, text: str) -> tuple[str, List[str]]:
        """
        Parse command and arguments from text.

        Args:
            text: Full command text

        Returns:
            Tuple of (command_name, args_list)

        Example:
            "/echo hello world" -> ("/echo", ["hello", "world"])
        """
        if not self.is_command(text):
            return "", []

        parts = text.strip().split()
        if not parts:
            return "", []

        command = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        return command, args

    async def process(self, text: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Process a command and return response.

        Args:
            text: Full command text (e.g., "/echo hello")
            context: Context dictionary with keys like:
                - account_id: Bot account ID
                - user_id: User who sent the command
                - timestamp: Message timestamp
                - client: WeixinClient instance
                - Any other custom context

        Returns:
            Response string or None if command not found

        Raises:
            Exception: Any exception from handler is caught and returned as error message
        """
        if not self.is_command(text):
            return None

        command, args = self.parse_command(text)

        if command not in self._handlers:
            return f"Unknown command: {command}. Type /help for available commands."

        handler = self._handlers[command]
        self._command_usage[command] += 1

        try:
            # Track execution time
            start_time = time.time()

            # Check if handler is async
            if inspect.iscoroutinefunction(handler):
                result = await handler(args, context)
            else:
                result = handler(args, context)

            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"Command {command} executed in {elapsed_ms:.2f}ms")

            # Convert result to string if needed
            if result is None:
                return None
            if isinstance(result, str):
                return result
            return str(result)

        except Exception as e:
            logger.exception(f"Error executing command {command}: {e}")
            return f"Error executing {command}: {str(e)}"

    def get_commands(self) -> List[str]:
        """
        Get list of registered commands.

        Returns:
            List of command names
        """
        return list(self._handlers.keys())

    def get_command_info(self) -> Dict[str, str]:
        """
        Get command information with descriptions.

        Returns:
            Dictionary mapping command names to descriptions
        """
        return self._descriptions.copy()

    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get command usage statistics.

        Returns:
            Dictionary mapping command names to usage count
        """
        return self._command_usage.copy()

    # Built-in command handlers

    async def _handle_echo(self, args: List[str], context: Dict[str, Any]) -> str:
        """
        Echo command handler.

        Usage: /echo <message>
        Returns the message back with timing stats.
        """
        if not args:
            return "Usage: /echo <message>"

        message = " ".join(args)
        return f"Echo: {message}"

    async def _handle_toggle_debug(
        self, args: List[str], context: Dict[str, Any]
    ) -> str:
        """
        Toggle debug mode command handler.

        Usage: /toggle-debug [account_id]
        Toggles debug mode on/off for the current account or globally.

        If account_id is provided in context or args, toggles per-account debug mode.
        Otherwise toggles global debug mode.
        """
        from .debug_mode import DebugMode, DebugModeManager

        # Get account_id from context or args
        account_id = None
        if args:
            account_id = args[0]
        elif context and "account_id" in context:
            account_id = context["account_id"]

        # Get or create debug manager from context
        debug_manager = None
        if context and "debug_manager" in context:
            debug_manager = context["debug_manager"]
        elif context and "client" in context:
            client = context["client"]
            if hasattr(client, "debug_manager"):
                debug_manager = client.debug_manager

        if account_id and debug_manager:
            # Toggle per-account debug mode
            new_state = debug_manager.toggle(account_id)
            state_str = "ON" if new_state else "OFF"

            # Also toggle global debug mode for logging
            if new_state:
                DebugMode.enable()
            else:
                # Only disable global if no other accounts have debug enabled
                if not debug_manager.get_all_enabled():
                    DebugMode.disable()

            return f"Debug mode for account {account_id}: {state_str}"
        else:
            # Toggle global debug mode
            new_state = DebugMode.toggle()
            state_str = "ON" if new_state else "OFF"
            return f"Global debug mode: {state_str}\nLogging level set to {state_str}"

    async def _handle_status(self, args: List[str], context: Dict[str, Any]) -> str:
        """
        Status command handler.

        Usage: /status
        Shows bot status information.
        """
        from .debug_mode import DebugMode

        uptime = time.time() - self._start_time
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)

        lines = [
            "Bot Status:",
            f"  Uptime: {hours}h {minutes}m {seconds}s",
            f"  Global debug mode: {'ON' if DebugMode.is_enabled() else 'OFF'}",
            f"  Registered commands: {len(self._handlers)}",
        ]

        # Add context info if available
        account_id = None
        if context:
            if "account_id" in context:
                account_id = context["account_id"]
                lines.append(f"  Account ID: {account_id}")
            if "user_id" in context:
                lines.append(f"  User ID: {context['user_id']}")

        # Show per-account debug status if available
        if context and "debug_manager" in context and account_id:
            debug_manager = context["debug_manager"]
            if debug_manager.is_enabled(account_id):
                lines.append(f"  Account debug mode: ON")
                # Show timing trace if available
                trace = debug_manager.get_timing_trace(account_id)
                if trace:
                    lines.append(f"  Timing records: {len(trace)}")

        # Add command usage stats
        if any(self._command_usage.values()):
            lines.append("  Command usage:")
            for cmd, count in sorted(self._command_usage.items(), key=lambda x: -x[1]):
                if count > 0:
                    lines.append(f"    {cmd}: {count}")

        return "\n".join(lines)

    async def _handle_help(self, args: List[str], context: Dict[str, Any]) -> str:
        """
        Help command handler.

        Usage: /help [command]
        Shows help for all commands or specific command.
        """
        if args:
            # Show help for specific command
            command = args[0]
            if not command.startswith(self.COMMAND_PREFIX):
                command = self.COMMAND_PREFIX + command

            if command in self._descriptions:
                return f"{command}: {self._descriptions[command]}"
            else:
                return f"Unknown command: {command}"

        # Show all commands
        lines = ["Available commands:"]
        for cmd in sorted(self._handlers.keys()):
            desc = self._descriptions.get(cmd, "No description")
            lines.append(f"  {cmd} - {desc}")

        lines.append("\nUse /help <command> for more details.")

        return "\n".join(lines)


# Global registry instance for convenience
default_registry = SlashCommandRegistry()


def get_default_registry() -> SlashCommandRegistry:
    """
    Get the default global registry instance.

    Returns:
        Default SlashCommandRegistry instance
    """
    return default_registry
