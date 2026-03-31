"""
OpenClaw Weixin Python SDK

A Python implementation of the OpenClaw Weixin channel plugin.
Provides integration with Weixin (WeChat) via HTTP API.
"""

from .client import WeixinClient, DEFAULT_BASE_URL, CDN_BASE_URL
from .types import (
    WeixinMessage,
    MessageItem,
    TextItem,
    ImageItem,
    VoiceItem,
    FileItem,
    VideoItem,
    CDNMedia,
    SendMessageReq,
    GetUpdatesReq,
    GetUpdatesResp,
    MessageType,
    MessageItemType,
    MessageState,
    UploadMediaType,
)
from .exceptions import (
    WeixinError,
    WeixinAPIError,
    WeixinAuthError,
    WeixinTimeoutError,
    WeixinSessionExpiredError,
)
from .account import (
    WeixinAccount,
    WeixinAccountManager,
    get_default_account_manager,
)
from .media import (
    MediaUploader,
    MediaSender,
    send_image,
    send_video,
    send_file,
    send_voice,
)
from .utils import (
    markdown_to_plain_text,
    chunk_text,
)
from .retry import (
    RetryConfig,
    CircuitBreaker,
    CircuitState,
    retry,
    retry_async,
)
from .rate_limiter import (
    TokenBucket,
    SlidingWindow,
    RateLimiter,
    rate_limit,
)
from .webhook import (
    WebhookServer,
    WebhookHandler,
    WebhookMessage,
    WebhookEvent,
    WebhookEventType,
    SignatureVerificationError,
    run_server,
)
from .log_upload import (
    LogUploader,
    LogUploadError,
    get_upload_url,
    format_size,
)

__version__ = "1.0.0"
__all__ = [
    # Core client
    "WeixinClient",
    "DEFAULT_BASE_URL",
    "CDN_BASE_URL",
    # Types
    "WeixinMessage",
    "MessageItem",
    "TextItem",
    "ImageItem",
    "VoiceItem",
    "FileItem",
    "VideoItem",
    "CDNMedia",
    "SendMessageReq",
    "GetUpdatesReq",
    "GetUpdatesResp",
    "MessageType",
    "MessageItemType",
    "MessageState",
    "UploadMediaType",
    # Exceptions
    "WeixinError",
    "WeixinAPIError",
    "WeixinAuthError",
    "WeixinTimeoutError",
    "WeixinSessionExpiredError",
    # Account management
    "WeixinAccount",
    "WeixinAccountManager",
    "get_default_account_manager",
    # Media
    "MediaUploader",
    "MediaSender",
    "send_image",
    "send_video",
    "send_file",
    "send_voice",
    # Utils
    "markdown_to_plain_text",
    "chunk_text",
    # Retry
    "RetryConfig",
    "CircuitBreaker",
    "CircuitState",
    "retry",
    "retry_async",
    # Rate limiting
    "TokenBucket",
    "SlidingWindow",
    "RateLimiter",
    "rate_limit",
    # Webhook
    "WebhookServer",
    "WebhookHandler",
    "WebhookMessage",
    "WebhookEvent",
    "WebhookEventType",
    "SignatureVerificationError",
    "run_server",
    # Log upload
    "LogUploader",
    "LogUploadError",
    "get_upload_url",
    "format_size",
]
