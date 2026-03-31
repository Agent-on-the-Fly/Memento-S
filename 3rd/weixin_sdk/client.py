"""
Weixin API client (mirrors src/api/api.ts)
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import secrets
from typing import Optional, Callable, AsyncIterator, Dict, Any, List, Type, cast
from urllib.parse import urljoin

import aiohttp

from .types import (
    BaseInfo,
    GetUpdatesReq,
    GetUpdatesResp,
    SendMessageReq,
    SendMessageResp,
    SendTypingReq,
    SendTypingResp,
    GetConfigResp,
    GetUploadUrlReq,
    GetUploadUrlResp,
    WeixinMessage,
    MessageItem,
    MessageItemType,
    TextItem,
    MessageType,
    MessageState,
    dict_to_get_updates_resp,
)
from .exceptions import (
    WeixinError,
    WeixinAPIError,
    WeixinAuthError,
    WeixinTimeoutError,
    WeixinSessionExpiredError,
    WeixinSessionPausedError,
)
from .retry import RetryConfig, CircuitBreaker, retry_async
from .api.session_guard import SessionGuard
from .api.config_cache import ConfigCache
from .storage import SyncBufferManager
from .messaging.commands import SlashCommandRegistry
from .messaging.debug_mode import DebugModeManager, TimingContext, MessagePipelineTracer
from .media_upload import MediaSender

logger = logging.getLogger(__name__)

# Default timeouts
DEFAULT_LONG_POLL_TIMEOUT_MS = 35_000
DEFAULT_API_TIMEOUT_MS = 15_000
DEFAULT_CONFIG_TIMEOUT_MS = 10_000

# Default base URLs (mirrors TypeScript src/auth/accounts.ts and src/api/api.ts)
DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"
CDN_BASE_URL = "https://novac2c.cdn.weixin.qq.com/c2c"


class WeixinClient:
    """
    Weixin HTTP API client.

    Provides methods for:
    - getUpdates: Long-polling for incoming messages
    - sendMessage: Send messages (text/media)
    - sendTyping: Send typing indicators
    - getConfig: Get bot configuration
    - getUploadUrl: Get CDN upload URLs

    Example:
        async with WeixinClient(base_url="https://api.weixin.com", token="xxx") as client:
            # Receive messages
            async for message in client.poll_messages():
                print(f"Received: {message}")

            # Send message
            await client.send_text(to_user_id="user_123", text="Hello!")
    """

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        timeout_ms: int = DEFAULT_API_TIMEOUT_MS,
        long_poll_timeout_ms: int = DEFAULT_LONG_POLL_TIMEOUT_MS,
        route_tag: Optional[str] = None,
        retry_config: Optional[Dict[str, Any]] = None,
        session_guard: Optional["SessionGuard"] = None,
        sync_buffer_manager: Optional[SyncBufferManager] = None,
        config_cache: Optional["ConfigCache"] = None,
        debug_manager: Optional[DebugModeManager] = None,
    ):
        """
        Initialize Weixin client.

        Args:
            base_url: API base URL (e.g., "https://api.weixin.com")
            token: Authorization token (Bearer token)
            timeout_ms: Default timeout for regular API calls
            long_poll_timeout_ms: Timeout for long-poll getUpdates calls
            route_tag: Optional route tag for X-SKRouteTag header
            retry_config: Optional retry configuration dict with keys:
                - max_attempts: Max retry attempts (default: 3)
                - base_delay: Base delay in seconds (default: 1.0)
                - max_delay: Max delay cap in seconds (default: 30.0)
                - retryable_exceptions: List of exception classes to retry
                - exponential_base: Base for exponential backoff (default: 2.0)
                - jitter: Add random jitter (default: True)
                - use_circuit_breaker: Enable circuit breaker (default: False)
                - circuit_failure_threshold: Failures to open circuit (default: 5)
                - circuit_recovery_timeout: Seconds before half-open (default: 60)
            session_guard: Optional SessionGuard for handling session expiration
            sync_buffer_manager: Optional SyncBufferManager for cursor persistence
            config_cache: Optional ConfigCache for caching bot configuration.
                         If not provided, a default cache is created.
            debug_manager: Optional DebugModeManager for performance monitoring.
                          If not provided, a default instance is created.
        """
        self.base_url = base_url.rstrip("/") + "/"
        self.token = token
        self.timeout_ms = timeout_ms
        self.long_poll_timeout_ms = long_poll_timeout_ms
        self.route_tag = route_tag
        self._session: Optional[aiohttp.ClientSession] = None
        self._channel_version = self._read_channel_version()

        # Context token store (account_id:user_id -> token)
        self._context_tokens: Dict[str, str] = {}

        # Session guard for handling session expiration
        self._session_guard = session_guard
        self._account_id: Optional[str] = None

        # Slash command registry for bot commands
        self._slash_command_registry: Optional[SlashCommandRegistry] = None

        # Sync buffer manager for cursor persistence
        self._sync_buffer_manager = sync_buffer_manager
        self._current_cursor: Optional[str] = None
        self._messages_since_save: int = 0

        # Retry configuration
        self._retry_config = self._parse_retry_config(retry_config)
        self._circuit_breaker: Optional[CircuitBreaker] = None
        if (
            self._retry_config
            and retry_config
            and retry_config.get("use_circuit_breaker")
        ):
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=retry_config.get("circuit_failure_threshold", 5),
                recovery_timeout=retry_config.get("circuit_recovery_timeout", 60.0),
            )

        # Config cache for get_config() with retry logic
        self._config_cache = config_cache or ConfigCache()

        # Debug manager for performance monitoring
        self.debug_manager = debug_manager or DebugModeManager()

    def _parse_retry_config(
        self, config: Optional[Dict[str, Any]]
    ) -> Optional[RetryConfig]:
        """Parse retry configuration dict into RetryConfig."""
        if not config:
            return None

        return RetryConfig(
            max_attempts=config.get("max_attempts", 3),
            base_delay=config.get("base_delay", 1.0),
            max_delay=config.get("max_delay", 30.0),
            retryable_exceptions=tuple(
                config.get("retryable_exceptions", [WeixinTimeoutError, WeixinAPIError])
            ),
            exponential_base=config.get("exponential_base", 2.0),
            jitter=config.get("jitter", True),
        )

    def _read_channel_version(self) -> str:
        """Read version from package."""
        try:
            import importlib.metadata

            return importlib.metadata.version("openclaw-weixin-python")
        except Exception:
            return "unknown"

    def _build_base_info(self) -> BaseInfo:
        """Build base info for requests."""
        return BaseInfo(channel_version=self._channel_version)

    def _random_wechat_uin(self) -> str:
        """Generate random X-WECHAT-UIN header."""
        uint32 = secrets.randbits(32)
        return base64.b64encode(str(uint32).encode()).decode()

    def _build_headers(self, body: str) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "Content-Length": str(len(body.encode("utf-8"))),
            "X-WECHAT-UIN": self._random_wechat_uin(),
        }

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        if self.route_tag:
            headers["X-SKRouteTag"] = self.route_tag

        logger.debug(f"Request headers: {self._redact_headers(headers)}")
        return headers

    def _redact_body(self, body: str, max_len: int = 200) -> str:
        """
        Redact sensitive fields from JSON body.

        Args:
            body: JSON body string
            max_len: Maximum length before truncation

        Returns:
            Redacted and optionally truncated body
        """
        # Sensitive field patterns to redact
        sensitive_patterns = [
            r'("context_token"\s*:\s*")[^"]*"',
            r'("bot_token"\s*:\s*")[^"]*"',
            r'("token"\s*:\s*")[^"]*"',
            r'("authorization"\s*:\s*")[^"]*"',
            r'("Authorization"\s*:\s*")[^"]*"',
        ]

        redacted = body
        for pattern in sensitive_patterns:
            redacted = re.sub(pattern, r"\1<redacted>", redacted)

        # Truncate if too long
        if len(redacted) > max_len:
            return redacted[:max_len] + f"...(truncated, totalLen={len(redacted)})"

        return redacted

    def _redact_token(self, token: str, prefix_len: int = 6) -> str:
        """
        Show only prefix of token for logging.

        Args:
            token: Token string
            prefix_len: Number of characters to show

        Returns:
            Redacted token with only prefix visible
        """
        if not token:
            return "<redacted>"
        if len(token) <= prefix_len:
            return "<redacted>"
        return token[:prefix_len] + "*" * (len(token) - prefix_len)

    def _redact_url(self, url: str) -> str:
        """
        Remove query string from URL for logging.

        Args:
            url: URL string

        Returns:
            URL with query string removed
        """
        # Remove query parameters
        if "?" in url:
            return url.split("?")[0]
        return url

    def _redact_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Redact sensitive values in headers for logging."""
        redacted = headers.copy()
        sensitive_keys = {"Authorization", "authorization"}
        for key in sensitive_keys:
            if key in redacted:
                val = redacted[key]
                if val and val.startswith("Bearer "):
                    redacted[key] = f"Bearer {self._redact_token(val[7:])}"
                else:
                    redacted[key] = self._redact_token(val)
        return redacted

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def _api_fetch(
        self,
        endpoint: str,
        body: str,
        timeout_ms: int,
        label: str,
    ) -> str:
        """
        Common fetch wrapper: POST JSON to a Weixin API endpoint.

        Supports retry with exponential backoff and optional circuit breaker.

        Args:
            endpoint: API endpoint path (e.g., "getupdates")
            body: JSON body string
            timeout_ms: Request timeout in milliseconds
            label: Label for logging

        Returns:
            Response body as string

        Raises:
            WeixinTimeoutError: If request times out
            WeixinAPIError: If HTTP error occurs
            WeixinAuthError: If authentication fails
        """
        # Use retry if configured
        if self._retry_config:
            return await self._api_fetch_with_retry(endpoint, body, timeout_ms, label)

        # Original behavior without retry
        return await self._api_fetch_single(endpoint, body, timeout_ms, label)

    async def _api_fetch_single(
        self,
        endpoint: str,
        body: str,
        timeout_ms: int,
        label: str,
    ) -> str:
        """Single API fetch without retry."""
        await self._ensure_session()

        url = urljoin(self.base_url, endpoint)
        headers = self._build_headers(body)
        timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)

        logger.debug(f"{label}: POST {self._redact_url(url)}")
        logger.debug(f"{label}: body={self._redact_body(body)}")

        try:
            if self._session is None:
                raise WeixinAPIError("HTTP session not initialized")
            async with self._session.post(
                url,
                headers=headers,
                data=body,
                timeout=timeout,
            ) as response:
                text = await response.text()

                if response.status == 401:
                    raise WeixinAuthError(f"Authentication failed: {text}")

                if response.status != 200:
                    raise WeixinAPIError(
                        f"HTTP {response.status}: {text}",
                        code=response.status,
                    )

                logger.debug(f"{label}: response={self._redact_body(text)}")
                return text

        except asyncio.TimeoutError:
            raise WeixinTimeoutError(f"Request timeout after {timeout_ms}ms")
        except aiohttp.ClientError as e:
            raise WeixinAPIError(f"Request failed: {e}")

    async def _api_fetch_with_retry(
        self,
        endpoint: str,
        body: str,
        timeout_ms: int,
        label: str,
    ) -> str:
        """API fetch with retry logic."""
        import random

        config = self._retry_config
        max_attempts = config.max_attempts  # type: ignore
        last_exception: Optional[Exception] = None

        for attempt in range(max_attempts):
            # Check circuit breaker before attempt
            if self._circuit_breaker and not self._circuit_breaker.can_execute():
                raise WeixinAPIError(
                    f"Circuit breaker is OPEN for {label}, request rejected"
                )

            try:
                result = await self._api_fetch_single(endpoint, body, timeout_ms, label)
                # Record success
                if self._circuit_breaker:
                    self._circuit_breaker.record_success()
                return result

            except Exception as e:
                last_exception = e

                # Check if exception is retryable
                if not config.is_retryable(e):  # type: ignore
                    # Record failure for non-retryable errors too
                    if self._circuit_breaker:
                        self._circuit_breaker.record_failure()
                    raise

                # Calculate delay with exponential backoff
                if attempt < max_attempts - 1:
                    delay = config.calculate_delay(attempt)  # type: ignore
                    logger.warning(
                        f"Retryable error in {label}: {e}. "
                        f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_attempts})"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {max_attempts} attempts failed for {label}: {e}")

                # Record failure
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()

        if last_exception:
            raise last_exception
        raise RuntimeError(f"Retry logic error for {label}: no exception but no result")

    async def get_updates(
        self,
        get_updates_buf: str = "",
    ) -> GetUpdatesResp:
        """
        Long-polling getUpdates: retrieve new messages from the server.

        This method blocks until:
        - New messages arrive
        - Timeout occurs (server returns empty response)
        - Error occurs

        Args:
            get_updates_buf: Sync cursor from previous response (empty for first call)

        Returns:
            GetUpdatesResp containing messages and new cursor.
            On timeout, returns empty GetUpdatesResp with ret=0 and msgs=[].

        Raises:
            WeixinSessionExpiredError: If session expired (errcode -14)
            WeixinAPIError: For other API errors
        """
        # Check session guard before making API call
        await self._check_session_guard()

        req = GetUpdatesReq(get_updates_buf=get_updates_buf)
        body = json.dumps(
            {
                **req.__dict__,
                "base_info": self._build_base_info().__dict__,
            }
        )

        try:
            # Track timing for platform operation if debug mode is enabled
            if self._account_id and self.debug_manager.is_enabled(self._account_id):
                with TimingContext(
                    self.debug_manager, self._account_id, "platform_receive"
                ):
                    text = await self._api_fetch(
                        endpoint="ilink/bot/getupdates",
                        body=body,
                        timeout_ms=self.long_poll_timeout_ms,
                        label="getUpdates",
                    )
            else:
                text = await self._api_fetch(
                    endpoint="ilink/bot/getupdates",
                    body=body,
                    timeout_ms=self.long_poll_timeout_ms,
                    label="getUpdates",
                )

            data = json.loads(text)
            resp = dict_to_get_updates_resp(data)

            # Handle session expired - check session guard first
            if resp.errcode == -14:
                if self._session_guard and self._account_id:
                    await self._session_guard.pause(self._account_id)
                    remaining = await self._session_guard.get_remaining_pause_time(
                        self._account_id
                    )
                    raise WeixinSessionPausedError(
                        f"Session expired for account {self._account_id}. "
                        f"API calls paused for {remaining} seconds.",
                        code=resp.errcode,
                        response=data,
                        remaining_seconds=remaining,
                    )
                else:
                    # No session guard, raise original error
                    raise WeixinSessionExpiredError("Session expired, please re-login")

            # Handle other errors
            # ret can be None if API response doesn't include it (treat as success)
            if resp.ret is not None and resp.ret != 0:
                raise WeixinAPIError(
                    f"API error: ret={resp.ret}, errcode={resp.errcode}, errmsg={resp.errmsg}",
                    code=resp.ret,
                    response=data,
                )

            return resp

        except WeixinTimeoutError:
            # Long-poll timeout is normal; return empty response so caller can retry
            logger.debug("Long-poll timeout, returning empty response")
            return GetUpdatesResp(
                ret=0,
                msgs=[],
                get_updates_buf=get_updates_buf,
            )

    async def poll_messages(
        self,
        initial_cursor: str = "",
        resume_from_cursor: Optional[str] = None,
        auto_save_interval: int = 10,
    ) -> AsyncIterator[WeixinMessage]:
        """
        Continuously poll for messages using long-polling.

        This is an async generator that yields messages as they arrive.
        Handles reconnection and cursor management automatically.
        Integrates with SyncBufferManager for cursor persistence.

        Args:
            initial_cursor: Initial sync cursor (empty for fresh start)
            resume_from_cursor: Optional cursor to resume from (overrides initial_cursor)
            auto_save_interval: Save cursor after every N messages (0 to disable auto-save)

        Yields:
            WeixinMessage objects as they arrive

        Example:
            async for message in client.poll_messages():
                print(f"From {message.from_user_id}: {message}")

            # Resume from saved cursor
            cursor = manager.load_cursor("account_123")
            async for message in client.poll_messages(resume_from_cursor=cursor):
                # Process message
                pass
        """
        # Determine starting cursor
        if resume_from_cursor is not None:
            cursor = resume_from_cursor
            logger.info(f"Resuming poll from provided cursor")
        elif initial_cursor:
            cursor = initial_cursor
        else:
            cursor = ""

        self._current_cursor = cursor
        self._messages_since_save = 0

        while True:
            try:
                resp = await self.get_updates(cursor)

                # Yield messages
                if resp.msgs:
                    for msg in resp.msgs:
                        yield msg
                        self._messages_since_save += 1

                        # Auto-save cursor periodically
                        if (
                            auto_save_interval > 0
                            and self._messages_since_save >= auto_save_interval
                        ):
                            if (
                                self._account_id
                                and self._sync_buffer_manager
                                and self._current_cursor
                            ):
                                self.save_sync_state()
                                self._messages_since_save = 0

                # Update cursor for next request
                cursor = resp.get_updates_buf or ""
                self._current_cursor = cursor

                # Save cursor after successful poll if we have messages
                if resp.msgs and self._account_id and self._sync_buffer_manager:
                    self.save_sync_state()
                    self._messages_since_save = 0

                # Use server-suggested timeout if provided
                if resp.longpolling_timeout_ms:
                    self.long_poll_timeout_ms = resp.longpolling_timeout_ms

            except WeixinTimeoutError:
                # Timeout is normal for long-polling, continue
                logger.debug("Long-poll timeout, retrying...")
                continue

            except WeixinSessionExpiredError:
                logger.error("Session expired, stopping poll")
                raise

            except WeixinSessionPausedError:
                logger.error("Session paused due to expiration, stopping poll")
                raise

            except Exception as e:
                logger.error(f"Error in poll_messages: {e}")
                # Brief delay before retry
                await asyncio.sleep(1)

    def save_sync_state(self) -> bool:
        """
        Manually save the current sync cursor to disk.

        Returns:
            True if cursor was saved successfully, False otherwise
        """
        if not self._account_id:
            logger.warning("Cannot save sync state: no account_id set")
            return False

        if not self._sync_buffer_manager:
            logger.warning("Cannot save sync state: no SyncBufferManager configured")
            return False

        if not self._current_cursor:
            logger.debug("No cursor to save")
            return False

        try:
            self._sync_buffer_manager.save_cursor(
                self._account_id, self._current_cursor
            )
            logger.debug(f"Saved sync state for account {self._account_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save sync state: {e}")
            return False

    def clear_sync_state(self) -> bool:
        """
        Clear the saved sync cursor for the current account.

        Call this after successfully processing all messages to start
        fresh on next restart.

        Returns:
            True if cursor was cleared successfully, False otherwise
        """
        if not self._account_id:
            logger.warning("Cannot clear sync state: no account_id set")
            return False

        if not self._sync_buffer_manager:
            logger.warning("Cannot clear sync state: no SyncBufferManager configured")
            return False

        try:
            result = self._sync_buffer_manager.delete_cursor(self._account_id)
            if result:
                self._current_cursor = None
                logger.info(f"Cleared sync state for account {self._account_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to clear sync state: {e}")
            return False

    def load_sync_state(self) -> Optional[str]:
        """
        Load the saved sync cursor for the current account.

        Returns:
            Saved cursor string if found, None otherwise
        """
        if not self._account_id:
            logger.warning("Cannot load sync state: no account_id set")
            return None

        if not self._sync_buffer_manager:
            logger.warning("Cannot load sync state: no SyncBufferManager configured")
            return None

        try:
            cursor = self._sync_buffer_manager.load_cursor(self._account_id)
            if cursor:
                self._current_cursor = cursor
                logger.info(f"Loaded sync state for account {self._account_id}")
            return cursor
        except Exception as e:
            logger.error(f"Failed to load sync state: {e}")
            return None

    async def send_message(self, req: SendMessageReq) -> SendMessageResp:
        """
        Send a message (text or media).

        Args:
            req: SendMessageReq containing the message to send

        Returns:
            SendMessageResp (empty on success)
        """
        # Check session guard before making API call
        await self._check_session_guard()

        # Convert SendMessageReq to dict, handling nested WeixinMessage and MessageItem
        def convert_to_dict(obj):
            """Recursively convert dataclass objects to dict."""
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, (list, tuple)):
                return [convert_to_dict(item) for item in obj]
            if isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            # Check if it's a dataclass-like object
            if hasattr(obj, "model_dump"):
                return convert_to_dict(obj.model_dump())
            if hasattr(obj, "to_dict"):
                return convert_to_dict(obj.to_dict())
            if hasattr(obj, "__dict__"):
                return convert_to_dict(obj.__dict__)
            return str(obj)

        req_dict = convert_to_dict(req)

        body = json.dumps(
            {
                **req_dict,
                "base_info": self._build_base_info().__dict__,
            }
        )

        try:
            # Track timing for platform operation if debug mode is enabled
            if self._account_id and self.debug_manager.is_enabled(self._account_id):
                with TimingContext(
                    self.debug_manager, self._account_id, "platform_send"
                ):
                    await self._api_fetch(
                        endpoint="ilink/bot/sendmessage",
                        body=body,
                        timeout_ms=self.timeout_ms,
                        label="sendMessage",
                    )
            else:
                await self._api_fetch(
                    endpoint="ilink/bot/sendmessage",
                    body=body,
                    timeout_ms=self.timeout_ms,
                    label="sendMessage",
                )
        except WeixinAPIError as e:
            # Check if this is a session expired error
            if e.code == -14 or (e.response and e.response.get("errcode") == -14):
                if self._session_guard and self._account_id:
                    await self._session_guard.pause(self._account_id)
                    remaining = await self._session_guard.get_remaining_pause_time(
                        self._account_id
                    )
                    raise WeixinSessionPausedError(
                        f"Session expired for account {self._account_id}. "
                        f"API calls paused for {remaining} seconds.",
                        code=-14,
                        response=e.response,
                        remaining_seconds=remaining,
                    )
            raise

        return SendMessageResp()

    async def send_text(
        self,
        to_user_id: str,
        text: str,
        context_token: Optional[str] = None,
    ) -> str:
        """
        Send a text message (convenience method).

        Args:
            to_user_id: Target user ID
            text: Message text
            context_token: Optional context token for session continuity

        Returns:
            Client ID of the sent message
        """
        import uuid

        client_id = str(uuid.uuid4())

        msg = WeixinMessage(
            to_user_id=to_user_id,
            client_id=client_id,
            message_type=MessageType.BOT,
            message_state=MessageState.FINISH,
            item_list=[
                MessageItem(
                    type=MessageItemType.TEXT,
                    text_item=TextItem(text=text),
                )
            ],
            context_token=context_token,
        )

        req = SendMessageReq(msg=msg)
        await self.send_message(req)

        return client_id

    async def send_text_with_debug(
        self,
        to_user_id: str,
        text: str,
        context_token: Optional[str] = None,
        inject_debug: bool = True,
    ) -> str:
        """
        Send a text message with optional debug info injection.

        When debug mode is enabled for the account and inject_debug is True,
        appends timing trace information to the message.

        Args:
            to_user_id: Target user ID
            text: Message text
            context_token: Optional context token for session continuity
            inject_debug: Whether to inject debug info if debug mode is enabled

        Returns:
            Client ID of the sent message
        """
        # Inject debug info if enabled
        if inject_debug and self._account_id:
            text = self.debug_manager.inject_debug_info(self._account_id, text)

        return await self.send_text(to_user_id, text, context_token)

    def inject_debug_info(self, text: str) -> str:
        """
        Inject debug timing info into text if debug mode is enabled.

        Args:
            text: Original text

        Returns:
            Text with debug footer appended if debug mode is enabled
        """
        if not self._account_id:
            return text
        return self.debug_manager.inject_debug_info(self._account_id, text)

    def get_debug_timing_trace(self) -> str:
        """
        Get formatted timing trace for the current account.

        Returns:
            Formatted timing string or "no timing data"
        """
        if not self._account_id:
            return "no timing data"
        return self.debug_manager.format_timing_trace(self._account_id)

    def clear_debug_timing(self) -> None:
        """Clear timing trace for the current account."""
        if self._account_id:
            self.debug_manager.clear_timing_trace(self._account_id)

    async def send_typing(
        self,
        ilink_user_id: str,
        typing_ticket: str,
        status: int = 1,  # 1=typing, 2=cancel
    ) -> SendTypingResp:
        """
        Send typing indicator.

        Args:
            ilink_user_id: Target user ID
            typing_ticket: Typing ticket from getConfig
            status: 1=typing, 2=cancel typing

        Returns:
            SendTypingResp
        """
        # Check session guard before making API call
        await self._check_session_guard()

        req = SendTypingReq(
            ilink_user_id=ilink_user_id,
            typing_ticket=typing_ticket,
            status=status,
        )

        body = json.dumps(
            {
                **req.__dict__,
                "base_info": self._build_base_info().__dict__,
            }
        )

        # Debug logging
        logger.info(
            f"[WeixinClient] Sending typing indicator to {ilink_user_id}, status={status}"
        )
        logger.debug(f"[WeixinClient] Typing request body: {body}")

        try:
            text = await self._api_fetch(
                endpoint="ilink/bot/sendtyping",
                body=body,
                timeout_ms=DEFAULT_CONFIG_TIMEOUT_MS,
                label="sendTyping",
            )
            logger.info(f"[WeixinClient] Typing indicator sent successfully: {text}")
        except WeixinAPIError as e:
            # Check if this is a session expired error
            if e.code == -14 or (e.response and e.response.get("errcode") == -14):
                if self._session_guard and self._account_id:
                    await self._session_guard.pause(self._account_id)
                    remaining = await self._session_guard.get_remaining_pause_time(
                        self._account_id
                    )
                    raise WeixinSessionPausedError(
                        f"Session expired for account {self._account_id}. "
                        f"API calls paused for {remaining} seconds.",
                        code=-14,
                        response=e.response,
                        remaining_seconds=remaining,
                    )
            raise

        data = json.loads(text)
        return SendTypingResp(
            ret=data.get("ret"),
            errmsg=data.get("errmsg"),
        )

    async def _fetch_config_raw(self, user_id: Optional[str] = None) -> GetConfigResp:
        """
        Internal method to fetch config from API without caching.

        Args:
            user_id: Target user ID (ilink_user_id) - required for typing_ticket

        Returns:
            GetConfigResp containing typing_ticket

        Raises:
            WeixinSessionPausedError: If session expired (-14)
            WeixinAPIError: For other API errors
        """
        # Check session guard before making API call
        await self._check_session_guard()

        # Build request body with user_id if provided (required for typing_ticket)
        body_dict = {
            "base_info": self._build_base_info().__dict__,
        }
        if user_id:
            body_dict["ilink_user_id"] = user_id
            # Try to get context token for this user if available
            context_token = self._context_tokens.get(user_id)
            if context_token:
                body_dict["context_token"] = context_token

        body = json.dumps(body_dict)

        try:
            text = await self._api_fetch(
                endpoint="ilink/bot/getconfig",
                body=body,
                timeout_ms=DEFAULT_CONFIG_TIMEOUT_MS,
                label="getConfig",
            )
        except WeixinAPIError as e:
            # Check if this is a session expired error
            if e.code == -14 or (e.response and e.response.get("errcode") == -14):
                # Invalidate cache on session expiration
                self._config_cache.invalidate()
                if self._session_guard and self._account_id:
                    await self._session_guard.pause(self._account_id)
                    remaining = await self._session_guard.get_remaining_pause_time(
                        self._account_id
                    )
                    raise WeixinSessionPausedError(
                        f"Session expired for account {self._account_id}. "
                        f"API calls paused for {remaining} seconds.",
                        code=-14,
                        response=e.response,
                        remaining_seconds=remaining,
                    )
            raise

        data = json.loads(text)
        return GetConfigResp(
            ret=data.get("ret"),
            errmsg=data.get("errmsg"),
            typing_ticket=data.get("typing_ticket"),
        )

    async def get_config(self, user_id: Optional[str] = None) -> GetConfigResp:
        """
        Get bot configuration including typing_ticket.

        Uses ConfigCache for intelligent caching with automatic refresh
        and exponential backoff retry on failures.

        Args:
            user_id: Target user ID (ilink_user_id) - required for typing_ticket

        Returns:
            GetConfigResp containing typing_ticket
        """
        if user_id:
            # If user_id provided, fetch directly without caching
            # (config is per-user, cache design needs update for multi-user)
            return await self._fetch_config_raw(user_id)

        # Use cache for global config (no user_id)
        result = await self._config_cache.get_config(lambda: self._fetch_config_raw())
        return cast(GetConfigResp, result)

    async def refresh_config(self, user_id: Optional[str] = None) -> GetConfigResp:
        """
        Force refresh bot configuration, bypassing cache.

        Args:
            user_id: Target user ID (ilink_user_id) - required for typing_ticket

        Returns:
            GetConfigResp containing typing_ticket
        """
        if user_id:
            return await self._fetch_config_raw(user_id)

        result = await self._config_cache.refresh_config(
            lambda: self._fetch_config_raw()
        )
        return cast(GetConfigResp, result)

    def invalidate_config_cache(self) -> None:
        """Invalidate the config cache. Call this when session expires."""
        self._config_cache.invalidate()

    async def get_upload_url(self, req: GetUploadUrlReq) -> GetUploadUrlResp:
        """
        Get CDN upload URL for media.

        Args:
            req: GetUploadUrlReq with file metadata

        Returns:
            GetUploadUrlResp with upload parameters
        """
        # Check session guard before making API call
        await self._check_session_guard()

        body = json.dumps(
            {
                **req.__dict__,
                "base_info": self._build_base_info().__dict__,
            }
        )

        try:
            text = await self._api_fetch(
                endpoint="ilink/bot/getuploadurl",
                body=body,
                timeout_ms=self.timeout_ms,
                label="getUploadUrl",
            )
        except WeixinAPIError as e:
            # Check if this is a session expired error
            if e.code == -14 or (e.response and e.response.get("errcode") == -14):
                if self._session_guard and self._account_id:
                    await self._session_guard.pause(self._account_id)
                    remaining = await self._session_guard.get_remaining_pause_time(
                        self._account_id
                    )
                    raise WeixinSessionPausedError(
                        f"Session expired for account {self._account_id}. "
                        f"API calls paused for {remaining} seconds.",
                        code=-14,
                        response=e.response,
                        remaining_seconds=remaining,
                    )
            raise

        data = json.loads(text)
        return GetUploadUrlResp(
            upload_param=data.get("upload_param"),
            thumb_upload_param=data.get("thumb_upload_param"),
        )

    # Context token management

    def set_context_token(self, account_id: str, user_id: str, token: str):
        """Store context token for account+user pair."""
        key = f"{account_id}:{user_id}"
        self._context_tokens[key] = token
        logger.debug(f"Set context token: {key} = {self._redact_token(token)}")

    def get_context_token(self, account_id: str, user_id: str) -> Optional[str]:
        """Get context token for account+user pair."""
        key = f"{account_id}:{user_id}"
        return self._context_tokens.get(key)

    def clear_context_tokens(self, account_id: str):
        """Clear all context tokens for an account."""
        prefix = f"{account_id}:"
        keys_to_remove = [
            k for k in self._context_tokens.keys() if k.startswith(prefix)
        ]
        for key in keys_to_remove:
            del self._context_tokens[key]
        logger.info(
            f"Cleared {len(keys_to_remove)} context tokens for account={account_id}"
        )

    def set_account_id(self, account_id: str):
        """Set the account ID for session guard."""
        self._account_id = account_id

    async def _check_session_guard(self) -> None:
        """
        Check if session guard has paused this account.

        Raises:
            WeixinSessionPausedError: If account is paused
        """
        if not self._session_guard or not self._account_id:
            return

        if await self._session_guard.is_paused(self._account_id):
            remaining = await self._session_guard.get_remaining_pause_time(
                self._account_id
            )
            raise WeixinSessionPausedError(
                f"Session is paused for account {self._account_id}. "
                f"Retry after {remaining} seconds.",
                remaining_seconds=remaining,
            )

    async def _handle_session_error(self, error_code: int) -> bool:
        """
        Handle session error (-14) by pausing the account.

        Args:
            error_code: The error code from API response

        Returns:
            True if error was handled (session expired and account paused)
        """
        if not self._session_guard or not self._account_id:
            return False

        return await self._session_guard.check_session_error(
            error_code, self._account_id
        )

    # -------------------------------------------------------------------------
    # Slash Command Processing
    # -------------------------------------------------------------------------

    def set_slash_command_registry(
        self, registry: SlashCommandRegistry
    ) -> "WeixinClient":
        """
        Set the slash command registry for processing bot commands.

        Args:
            registry: SlashCommandRegistry instance

        Returns:
            Self for method chaining
        """
        self._slash_command_registry = registry
        logger.debug("Slash command registry set")
        return self

    def get_slash_command_registry(self) -> Optional[SlashCommandRegistry]:
        """
        Get the current slash command registry.

        Returns:
            SlashCommandRegistry instance or None if not set
        """
        return self._slash_command_registry

    def is_slash_command(self, text: str) -> bool:
        """
        Check if text is a slash command.

        Args:
            text: Message text to check

        Returns:
            True if text starts with command prefix "/"
        """
        if self._slash_command_registry:
            return self._slash_command_registry.is_command(text)
        # Fallback: simple check
        return bool(text and text.strip().startswith("/"))

    async def process_slash_command(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Process a slash command and return response.

        Checks if the message is a command and processes it through the
        slash command registry.

        Args:
            text: Message text to process
            context: Optional context dictionary with keys like:
                - account_id: Bot account ID
                - user_id: User who sent the command
                - timestamp: Message timestamp
                - Any other custom context

        Returns:
            Command response string if processed, None if not a command

        Example:
            # In message handler
            response = await client.process_slash_command(
                message_text,
                context={"user_id": user_id, "account_id": account_id}
            )
            if response:
                await client.send_text(user_id, response)
            else:
                # Pass to normal AI processing
                ai_response = await process_with_ai(message_text)
        """
        if not self._slash_command_registry:
            logger.warning("No slash command registry set, command not processed")
            return None

        # Build context if not provided
        if context is None:
            context = {}

        # Add client reference to context for commands that need it
        context["client"] = self

        # Ensure account_id is in context
        if "account_id" not in context and self._account_id:
            context["account_id"] = self._account_id

        try:
            return await self._slash_command_registry.process(text, context)
        except Exception as e:
            logger.exception(f"Error processing slash command: {e}")
            return f"Error processing command: {str(e)}"

    # Context manager support

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None

    async def close(self):
        """Close the client session."""
        if self._session:
            await self._session.close()
            self._session = None

    # -------------------------------------------------------------------------
    # Media Sending (High-level API)
    # -------------------------------------------------------------------------

    async def send_image(
        self,
        to_user_id: str,
        image_path: str,
        caption: str = "",
        context_token: Optional[str] = None,
    ) -> str:
        """
        Send an image message.

        Args:
            to_user_id: Target user ID
            image_path: Path to image file
            caption: Optional text caption
            context_token: Optional context token

        Returns:
            Message ID
        """
        sender = MediaSender(self)
        return await sender.send_image(to_user_id, image_path, caption, context_token)

    async def send_video(
        self,
        to_user_id: str,
        video_path: str,
        caption: str = "",
        context_token: Optional[str] = None,
    ) -> str:
        """
        Send a video message.

        Args:
            to_user_id: Target user ID
            video_path: Path to video file
            caption: Optional text caption
            context_token: Optional context token

        Returns:
            Message ID
        """
        sender = MediaSender(self)
        return await sender.send_video(to_user_id, video_path, caption, context_token)

    async def send_file(
        self,
        to_user_id: str,
        file_path: str,
        caption: str = "",
        context_token: Optional[str] = None,
    ) -> str:
        """
        Send a file attachment.

        Args:
            to_user_id: Target user ID
            file_path: Path to file
            caption: Optional text caption
            context_token: Optional context token

        Returns:
            Message ID
        """
        sender = MediaSender(self)
        return await sender.send_file(to_user_id, file_path, caption, context_token)

    async def send_voice(
        self,
        to_user_id: str,
        voice_path: str,
        duration_ms: int = 0,
        context_token: Optional[str] = None,
    ) -> str:
        """
        Send a voice message.

        Args:
            to_user_id: Target user ID
            voice_path: Path to voice file (SILK format recommended)
            duration_ms: Voice duration in milliseconds
            context_token: Optional context token

        Returns:
            Message ID
        """
        sender = MediaSender(self)
        return await sender.send_voice(
            to_user_id, voice_path, duration_ms, context_token
        )
