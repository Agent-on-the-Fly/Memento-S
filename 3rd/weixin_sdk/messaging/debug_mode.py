"""
Debug mode functionality for Weixin SDK performance monitoring and troubleshooting.

This module provides:
- DebugMode: Global debug mode state (for slash commands)
- DebugModeManager: Manage debug mode state for accounts
- TimingContext: Context manager for measuring execution time
- Debug info injection into AI replies

Example:
    from weixin_sdk.messaging.debug_mode import DebugModeManager, TimingContext

    debug_manager = DebugModeManager()
    debug_manager.enable("account_123")

    with TimingContext(debug_manager, "account_123", "ai_generation"):
        ai_response = await generate_response(message)

    if debug_manager.is_enabled("account_123"):
        trace = debug_manager.format_timing_trace("account_123")
        response += f"\n\n[Debug] {trace}"
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TimingRecord:
    """Record of a single timing measurement."""

    operation: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class TimingTrace:
    """Complete timing trace for a message processing pipeline."""

    platform_ms: float = 0.0  # Time spent on platform operations
    processing_ms: float = 0.0  # Time spent processing the message
    ai_ms: float = 0.0  # Time spent on AI generation
    records: List[TimingRecord] = field(default_factory=list)

    def add_record(self, operation: str, duration_ms: float) -> None:
        """Add a timing record and categorize it."""
        record = TimingRecord(operation=operation, duration_ms=duration_ms)
        self.records.append(record)

        # Categorize by operation type
        if operation.startswith("platform_") or operation in ("receive", "send"):
            self.platform_ms += duration_ms
        elif operation.startswith("ai_") or operation in ("ai_generation", "llm_call"):
            self.ai_ms += duration_ms
        else:
            self.processing_ms += duration_ms

    def clear(self) -> None:
        """Clear all timing records."""
        self.platform_ms = 0.0
        self.processing_ms = 0.0
        self.ai_ms = 0.0
        self.records.clear()

    def format(self) -> str:
        """Format timing trace as readable string."""
        parts = []
        if self.platform_ms > 0:
            parts.append(f"platform: {self.platform_ms:.1f}ms")
        if self.processing_ms > 0:
            parts.append(f"processing: {self.processing_ms:.1f}ms")
        if self.ai_ms > 0:
            parts.append(f"ai: {self.ai_ms:.1f}ms")
        return ", ".join(parts) if parts else "no timing data"


class DebugModeManager:
    """
    Manager for debug mode state and timing traces.

    Tracks which accounts have debug mode enabled and maintains
    timing traces for performance monitoring.

    Attributes:
        _enabled_accounts: Set of account IDs with debug mode enabled
        _timing_traces: Dict mapping account_id to TimingTrace
    """

    def __init__(self):
        """Initialize with empty state."""
        self._enabled_accounts: set = set()
        self._timing_traces: Dict[str, TimingTrace] = defaultdict(TimingTrace)
        logger.debug("DebugModeManager initialized")

    def enable(self, account_id: str) -> None:
        """
        Enable debug mode for an account.

        Args:
            account_id: The account ID to enable debug mode for
        """
        self._enabled_accounts.add(account_id)
        logger.info(f"Debug mode enabled for account: {account_id}")

    def disable(self, account_id: str) -> None:
        """
        Disable debug mode for an account.

        Args:
            account_id: The account ID to disable debug mode for
        """
        self._enabled_accounts.discard(account_id)
        # Clean up timing trace when disabling
        if account_id in self._timing_traces:
            del self._timing_traces[account_id]
        logger.info(f"Debug mode disabled for account: {account_id}")

    def is_enabled(self, account_id: str) -> bool:
        """
        Check if debug mode is enabled for an account.

        Args:
            account_id: The account ID to check

        Returns:
            True if debug mode is enabled, False otherwise
        """
        return account_id in self._enabled_accounts

    def toggle(self, account_id: str) -> bool:
        """
        Toggle debug mode for an account.

        Args:
            account_id: The account ID to toggle

        Returns:
            The new debug mode state (True = enabled, False = disabled)
        """
        if account_id in self._enabled_accounts:
            self.disable(account_id)
            return False
        else:
            self.enable(account_id)
            return True

    def get_all_enabled(self) -> List[str]:
        """
        Get all accounts with debug mode enabled.

        Returns:
            List of account IDs with debug mode enabled
        """
        return list(self._enabled_accounts)

    def record_timing(
        self, account_id: str, operation: str, duration_ms: float
    ) -> None:
        """
        Record a timing measurement for an account.

        Args:
            account_id: The account ID
            operation: Name of the operation being timed
            duration_ms: Duration in milliseconds
        """
        if not self.is_enabled(account_id):
            return

        self._timing_traces[account_id].add_record(operation, duration_ms)
        logger.debug(
            f"Timing recorded for {account_id}/{operation}: {duration_ms:.2f}ms"
        )

    def get_timing_trace(self, account_id: str) -> List[Dict[str, Any]]:
        """
        Get timing history for an account.

        Args:
            account_id: The account ID

        Returns:
            List of timing records as dictionaries
        """
        trace = self._timing_traces.get(account_id)
        if not trace:
            return []

        return [
            {
                "operation": record.operation,
                "duration_ms": record.duration_ms,
                "timestamp": record.timestamp,
            }
            for record in trace.records
        ]

    def clear_timing_trace(self, account_id: str) -> None:
        """
        Clear timing history for an account.

        Args:
            account_id: The account ID
        """
        if account_id in self._timing_traces:
            self._timing_traces[account_id].clear()
            logger.debug(f"Timing trace cleared for account: {account_id}")

    def format_timing_trace(self, account_id: str) -> str:
        """
        Format timing trace as readable string.

        Args:
            account_id: The account ID

        Returns:
            Formatted timing string (e.g., "platform: Xms, processing: Yms, ai: Zms")
        """
        trace = self._timing_traces.get(account_id)
        if not trace or not trace.records:
            return "no timing data"

        return trace.format()

    def get_debug_footer(self, account_id: str) -> str:
        """
        Get debug footer for appending to AI replies.

        Args:
            account_id: The account ID

        Returns:
            Formatted debug footer string
        """
        if not self.is_enabled(account_id):
            return ""

        trace_str = self.format_timing_trace(account_id)
        return f"\n\n[Debug] {trace_str}"

    def inject_debug_info(self, account_id: str, response: str) -> str:
        """
        Inject debug info into an AI response.

        Args:
            account_id: The account ID
            response: The original AI response

        Returns:
            Response with debug footer appended if debug mode is enabled
        """
        if not self.is_enabled(account_id):
            return response

        footer = self.get_debug_footer(account_id)
        return response + footer


class TimingContext:
    """
    Context manager for measuring execution time.

    Automatically records timing when exiting the context.

    Example:
        with TimingContext(debug_manager, "account_123", "ai_generation"):
            ai_response = await generate_response(message)
    """

    def __init__(
        self,
        debug_manager: DebugModeManager,
        account_id: str,
        operation: str,
    ):
        """
        Initialize timing context.

        Args:
            debug_manager: The DebugModeManager instance
            account_id: The account ID being tracked
            operation: Name of the operation being timed
        """
        self.debug_manager = debug_manager
        self.account_id = account_id
        self.operation = operation
        self.start_time: Optional[float] = None
        self.duration_ms: float = 0.0

    def __enter__(self) -> "TimingContext":
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record the duration."""
        if self.start_time is not None:
            end_time = time.perf_counter()
            self.duration_ms = (end_time - self.start_time) * 1000
            self.debug_manager.record_timing(
                self.account_id, self.operation, self.duration_ms
            )

    async def __aenter__(self) -> "TimingContext":
        """Async start timing."""
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async stop timing and record the duration."""
        if self.start_time is not None:
            end_time = time.perf_counter()
            self.duration_ms = (end_time - self.start_time) * 1000
            self.debug_manager.record_timing(
                self.account_id, self.operation, self.duration_ms
            )


@contextmanager
def timing_context(
    debug_manager: DebugModeManager,
    account_id: str,
    operation: str,
):
    """
    Functional context manager for timing (alternative to class-based).

    Example:
        with timing_context(debug_manager, "account_123", "processing"):
            process_message(message)
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        debug_manager.record_timing(account_id, operation, duration_ms)


class MessagePipelineTracer:
    """
    Tracer for message processing pipeline stages.

    Tracks timestamps at each stage of message processing for detailed
    performance analysis.

    Example:
        tracer = MessagePipelineTracer(debug_manager, "account_123", "msg_456")
        tracer.stage_begin("receive")
        # ... receive message ...
        tracer.stage_end("receive")

        tracer.stage_begin("processing")
        # ... process message ...
        tracer.stage_end("processing")

        tracer.stage_begin("ai_generation")
        # ... generate AI response ...
        tracer.stage_end("ai_generation")
    """

    def __init__(
        self,
        debug_manager: DebugModeManager,
        account_id: str,
        message_id: str,
    ):
        """
        Initialize pipeline tracer.

        Args:
            debug_manager: The DebugModeManager instance
            account_id: The account ID
            message_id: Unique message identifier
        """
        self.debug_manager = debug_manager
        self.account_id = account_id
        self.message_id = message_id
        self._stage_starts: Dict[str, float] = {}
        self._stage_durations: Dict[str, float] = {}

    def stage_begin(self, stage: str) -> None:
        """Mark the beginning of a pipeline stage."""
        if not self.debug_manager.is_enabled(self.account_id):
            return
        self._stage_starts[stage] = time.perf_counter()
        logger.debug(f"Pipeline stage '{stage}' began for message {self.message_id}")

    def stage_end(self, stage: str) -> None:
        """Mark the end of a pipeline stage and record duration."""
        if not self.debug_manager.is_enabled(self.account_id):
            return

        if stage not in self._stage_starts:
            logger.warning(
                f"Stage '{stage}' ended without begin for message {self.message_id}"
            )
            return

        start_time = self._stage_starts[stage]
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        self._stage_durations[stage] = duration_ms

        self.debug_manager.record_timing(self.account_id, stage, duration_ms)
        logger.debug(
            f"Pipeline stage '{stage}' ended for message {self.message_id}: "
            f"{duration_ms:.2f}ms"
        )

    def get_stage_duration(self, stage: str) -> Optional[float]:
        """Get the duration of a completed stage."""
        return self._stage_durations.get(stage)

    def get_all_durations(self) -> Dict[str, float]:
        """Get all stage durations."""
        return self._stage_durations.copy()

    def clear(self) -> None:
        """Clear all stage data."""
        self._stage_starts.clear()
        self._stage_durations.clear()


# =============================================================================
# Global Debug Mode (for slash commands)
# =============================================================================


class DebugMode:
    """
    Global debug mode state manager.

    Thread-safe singleton pattern for managing debug state across the SDK.
    Can be toggled via the /toggle-debug slash command.

    This is separate from DebugModeManager which handles per-account debug state.

    Example:
        from weixin_sdk.messaging.debug_mode import DebugMode

        # Check if debug is enabled
        if DebugMode.is_enabled():
            print("Debug logging...")

        # Toggle debug mode
        new_state = DebugMode.toggle()
    """

    _enabled: bool = False
    _lock: threading.Lock = threading.Lock()
    _instance: Optional["DebugMode"] = None

    def __new__(cls) -> "DebugMode":
        """Singleton pattern - only one DebugMode instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def is_enabled(cls) -> bool:
        """
        Check if debug mode is currently enabled.

        Returns:
            True if debug mode is on, False otherwise
        """
        return cls._enabled

    @classmethod
    def enable(cls) -> bool:
        """
        Enable debug mode.

        Returns:
            True (new state)
        """
        with cls._lock:
            cls._enabled = True
            # Update root logger level
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("Debug mode enabled")
        return True

    @classmethod
    def disable(cls) -> bool:
        """
        Disable debug mode.

        Returns:
            False (new state)
        """
        with cls._lock:
            cls._enabled = False
            # Update root logger level
            logging.getLogger().setLevel(logging.INFO)
            logger.info("Debug mode disabled")
        return False

    @classmethod
    def toggle(cls) -> bool:
        """
        Toggle debug mode on/off.

        Returns:
            New state (True if enabled, False if disabled)
        """
        with cls._lock:
            cls._enabled = not cls._enabled
            # Update root logger level
            if cls._enabled:
                logging.getLogger().setLevel(logging.DEBUG)
                logger.info("Debug mode enabled (via toggle)")
            else:
                logging.getLogger().setLevel(logging.INFO)
                logger.info("Debug mode disabled (via toggle)")
            return cls._enabled

    @classmethod
    def set(cls, enabled: bool) -> bool:
        """
        Set debug mode to specific state.

        Args:
            enabled: True to enable, False to disable

        Returns:
            New state
        """
        if enabled:
            return cls.enable()
        else:
            return cls.disable()


# Convenience function for quick checks
def is_debug() -> bool:
    """
    Quick check if debug mode is enabled.

    Returns:
        True if debug mode is on
    """
    return DebugMode.is_enabled()
