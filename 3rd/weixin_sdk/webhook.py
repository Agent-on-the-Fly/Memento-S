"""
Weixin webhook server for receiving callbacks and message push notifications.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Awaitable
from enum import IntEnum

import aiohttp
from aiohttp import web

from .types import (
    WeixinMessage,
    MessageItem,
    TextItem,
    ImageItem,
    VoiceItem,
    VideoItem,
    FileItem,
    MessageItemType,
    dict_to_weixin_message,
    dict_to_message_item,
)
from .exceptions import WeixinError

logger = logging.getLogger(__name__)


class WebhookEventType(IntEnum):
    """Webhook event types."""

    SUBSCRIBE = 1
    UNSUBSCRIBE = 2
    SCAN = 3
    LOCATION = 4
    CLICK = 5
    VIEW = 6


@dataclass
class WebhookMessage:
    """Parsed webhook message."""

    msg_type: str
    from_user_name: Optional[str] = None
    from_user_id: Optional[str] = None
    to_user_name: Optional[str] = None
    to_user_id: Optional[str] = None
    create_time: Optional[int] = None
    msg_id: Optional[int] = None
    content: Optional[str] = None  # For text messages
    media_id: Optional[str] = None  # For image/voice/video/file
    pic_url: Optional[str] = None  # For image messages
    format: Optional[str] = None  # For voice messages
    thumb_media_id: Optional[str] = None  # For video messages
    event: Optional[str] = None  # For event notifications
    event_key: Optional[str] = None  # For event with key
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WebhookEvent:
    """Parsed webhook event notification."""

    event_type: str
    from_user_name: Optional[str] = None
    from_user_id: Optional[str] = None
    to_user_name: Optional[str] = None
    to_user_id: Optional[str] = None
    create_time: Optional[int] = None
    event_key: Optional[str] = None
    ticket: Optional[str] = None  # For scan events
    latitude: Optional[float] = None  # For location events
    longitude: Optional[float] = None  # For location events
    precision: Optional[float] = None  # For location events
    raw_data: Dict[str, Any] = field(default_factory=dict)


# Type alias for message/event handlers
MessageHandler = Callable[[WebhookMessage], Awaitable[Any]]
EventHandler = Callable[[WebhookEvent], Awaitable[Any]]


class SignatureVerificationError(WeixinError):
    """Raised when signature verification fails."""

    pass


class WebhookHandler:
    """
    Handler for processing Weixin webhooks.

    Verifies signatures, parses incoming JSON, and routes to appropriate handlers.
    """

    def __init__(self, secret: Optional[str] = None):
        """
        Initialize webhook handler.

        Args:
            secret: Optional secret for HMAC-SHA256 signature verification.
        """
        self.secret = secret

    def verify_signature(
        self,
        signature: str,
        timestamp: str,
        nonce: str,
        token: Optional[str] = None,
    ) -> bool:
        """
        Verify the signature of a webhook request.

        Args:
            signature: The signature to verify (from Weixin request).
            timestamp: Timestamp of the request.
            nonce: Nonce string.
            token: Token to use (defaults to self.secret).

        Returns:
            True if signature is valid, False otherwise.
        """
        if not self.secret and not token:
            logger.warning("No secret configured, skipping signature verification")
            return True

        token = token or self.secret
        # Sort and concatenate: token, timestamp, nonce
        tmp_list = [token, timestamp, nonce]
        tmp_list.sort()
        tmp_str = "".join(tmp_list)
        # Calculate SHA1 (Weixin uses SHA1 for verification)
        computed_signature = hashlib.sha1(tmp_str.encode("utf-8")).hexdigest()

        return hmac.compare_digest(computed_signature, signature)

    def verify_hmac_signature(
        self,
        signature: str,
        timestamp: Optional[str] = None,
        nonce: Optional[str] = None,
        data: Optional[str] = None,
    ) -> bool:
        """
        Verify HMAC-SHA256 signature.

        Args:
            signature: The signature to verify.
            timestamp: Optional timestamp (can be used in signature).
            nonce: Optional nonce (can be used in signature).
            data: Optional raw data to sign.

        Returns:
            True if signature is valid.
        """
        if not self.secret:
            logger.warning("No secret configured, skipping HMAC signature verification")
            return True

        # Build the string to sign
        if data:
            sign_str = f"{timestamp or ''}{nonce or ''}{data}"
        else:
            sign_str = f"{self.secret}"

        computed = hmac.new(
            self.secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(computed, signature)

    def parse_message(self, data: Dict[str, Any]) -> WebhookMessage:
        """
        Parse incoming JSON into a WebhookMessage.

        Args:
            data: Raw JSON data from webhook request.

        Returns:
            Parsed WebhookMessage.
        """
        msg_type = data.get("msg_type", "").lower()

        message = WebhookMessage(
            msg_type=msg_type,
            from_user_name=data.get("from_user_name"),
            from_user_id=data.get("from_user_id"),
            to_user_name=data.get("to_user_name"),
            to_user_id=data.get("to_user_id"),
            create_time=data.get("create_time"),
            msg_id=data.get("msg_id"),
            content=data.get("content"),
            media_id=data.get("media_id"),
            pic_url=data.get("pic_url"),
            format=data.get("format"),
            thumb_media_id=data.get("thumb_media_id"),
            raw_data=data,
        )

        # Also try to parse as WeixinMessage for compatibility
        if msg_type in ("text", "image", "voice", "video", "file"):
            try:
                weixin_msg = dict_to_weixin_message(data)
                message.raw_data["weixin_message"] = weixin_msg
            except Exception as e:
                logger.warning(f"Failed to parse as WeixinMessage: {e}")

        return message

    def parse_event(self, data: Dict[str, Any]) -> WebhookEvent:
        """
        Parse incoming JSON into a WebhookEvent.

        Args:
            data: Raw JSON data from webhook request.

        Returns:
            Parsed WebhookEvent.
        """
        event_type = data.get("event", "").lower()

        return WebhookEvent(
            event_type=event_type,
            from_user_name=data.get("from_user_name"),
            from_user_id=data.get("from_user_id"),
            to_user_name=data.get("to_user_name"),
            to_user_id=data.get("to_user_id"),
            create_time=data.get("create_time"),
            event_key=data.get("event_key"),
            ticket=data.get("ticket"),
            latitude=data.get("latitude"),
            longitude=data.get("longitude"),
            precision=data.get("precision"),
            raw_data=data,
        )


class WebhookServer:
    """
    Async webhook server using aiohttp.

    Listens for Weixin callbacks and message push notifications.

    Example:
        server = WebhookServer(port=8080, secret="my-secret")

        @server.on_message
        async def handle_message(message):
            print(f"Received: {message}")

        @server.on_event("subscribe")
        async def handle_subscribe(event):
            print(f"User subscribed: {event}")

        await server.start()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        secret: Optional[str] = None,
    ):
        """
        Initialize webhook server.

        Args:
            host: Host to bind to.
            port: Port to listen on.
            secret: Optional secret for signature verification.
        """
        self.host = host
        self.port = port
        self.secret = secret

        self._handler = WebhookHandler(secret=secret)
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._message_handlers: list[MessageHandler] = []
        self._event_handlers: Dict[str, list[EventHandler]] = {}
        self._running = False

    def on_message(self, callback: MessageHandler) -> MessageHandler:
        """
        Register a message handler.

        Args:
            callback: Async function to handle incoming messages.

        Returns:
            The callback function (for decorator use).
        """
        self._message_handlers.append(callback)
        return callback

    def on_event(self, event_type: str) -> Callable[[EventHandler], EventHandler]:
        """
        Register an event handler for a specific event type.

        Args:
            event_type: The event type to handle (e.g., "subscribe", "unsubscribe").

        Returns:
            Decorator function.
        """

        def decorator(callback: EventHandler) -> EventHandler:
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            self._event_handlers[event_type].append(callback)
            return callback

        return decorator

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """
        Handle incoming webhook requests.

        Args:
            request: aiohttp request object.

        Returns:
            Web response with appropriate status code.
        """
        try:
            # Get query parameters for signature verification
            signature = request.query.get("signature", "")
            timestamp = request.query.get("timestamp", "")
            nonce = request.query.get("nonce", "")
            msg_signature = request.query.get(
                "msg_signature", ""
            )  # For encrypted messages

            # Verify signature if configured
            if self.secret and (signature or msg_signature):
                # Try Weixin-style verification first
                verify_sig = signature or msg_signature
                if not self._handler.verify_signature(verify_sig, timestamp, nonce):
                    logger.warning("Signature verification failed")
                    return web.Response(
                        status=403, text="Signature verification failed"
                    )

            # Parse request body
            try:
                data = await request.json()
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in request: {e}")
                return web.Response(status=400, text="Invalid JSON")

            # Route to appropriate handler
            msg_type = data.get("msg_type", "").lower()
            event_type = data.get("event", "").lower()

            # Handle event notifications
            if event_type:
                event = self._handler.parse_event(data)
                handlers = self._event_handlers.get(event_type, [])

                if handlers:
                    for handler in handlers:
                        try:
                            await handler(event)
                        except Exception as e:
                            logger.error(f"Event handler error: {e}", exc_info=True)
                else:
                    logger.debug(f"No handler for event type: {event_type}")

            # Handle messages
            if msg_type in ("text", "image", "voice", "video", "file"):
                message = self._handler.parse_message(data)
                handlers = self._message_handlers

                if handlers:
                    for handler in handlers:
                        try:
                            await handler(message)
                        except Exception as e:
                            logger.error(f"Message handler error: {e}", exc_info=True)
                else:
                    logger.debug("No message handlers registered")

            return web.Response(status=200, text="OK")

        except Exception as e:
            logger.error(f"Webhook handling error: {e}", exc_info=True)
            return web.Response(status=500, text="Internal server error")

    async def _handle_verify(self, request: web.Request) -> web.Response:
        """
        Handle verification request (Weixin URL verification).

        Args:
            request: aiohttp request object.

        Returns:
            Echo response with challenge string.
        """
        try:
            data = await request.json()
            echo_str = data.get("echostr", "")
            signature = request.query.get("signature", "")
            timestamp = request.query.get("timestamp", "")
            nonce = request.query.get("nonce", "")

            # Verify signature
            if self.secret:
                if not self._handler.verify_signature(signature, timestamp, nonce):
                    return web.Response(
                        status=403, text="Signature verification failed"
                    )

            return web.Response(text=echo_str, content_type="text/plain")

        except Exception as e:
            logger.error(f"Verify handling error: {e}", exc_info=True)
            return web.Response(status=500, text="Internal server error")

    async def start(self) -> None:
        """Start the webhook server."""
        if self._running:
            logger.warning("Server already running")
            return

        self._app = web.Application()

        # Register routes
        self._app.router.add_post("/webhook", self._handle_webhook)
        self._app.router.add_post("/verify", self._handle_verify)
        # Also support GET for verification
        self._app.router.add_get("/webhook", self._handle_webhook)
        self._app.router.add_get("/verify", self._handle_verify)

        # Create and start runner
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        self._running = True
        logger.info(f"Webhook server started on {self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the webhook server."""
        if not self._running:
            return

        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

        self._running = False
        self._app = None
        self._runner = None
        self._site = None
        logger.info("Webhook server stopped")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running


async def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    secret: Optional[str] = None,
    message_handler: Optional[MessageHandler] = None,
    event_handlers: Optional[Dict[str, EventHandler]] = None,
) -> WebhookServer:
    """
    Convenience function to run webhook server.

    Args:
        host: Host to bind to.
        port: Port to listen on.
        secret: Optional secret for signature verification.
        message_handler: Optional message handler.
        event_handlers: Optional dict of event type -> handler.

    Returns:
        Running WebhookServer instance.
    """
    server = WebhookServer(host=host, port=port, secret=secret)

    if message_handler:
        server.on_message(message_handler)

    if event_handlers:
        for event_type, handler in event_handlers.items():
            server.on_event(event_type)(handler)

    await server.start()
    return server


__all__ = [
    "WebhookServer",
    "WebhookHandler",
    "WebhookMessage",
    "WebhookEvent",
    "WebhookEventType",
    "SignatureVerificationError",
    "run_server",
]
