"""
Weixin SDK exceptions.
"""

from typing import Optional


class WeixinError(Exception):
    """Base exception for Weixin SDK."""

    def __init__(
        self, message: str, code: Optional[int] = None, response: Optional[dict] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.response = response


class WeixinAPIError(WeixinError):
    """API request failed."""

    pass


class WeixinAuthError(WeixinError):
    """Authentication failed (invalid token, session expired)."""

    pass


class WeixinTimeoutError(WeixinError):
    """Request timeout (long-poll or regular)."""

    pass


class WeixinConfigError(WeixinError):
    """Configuration error."""

    pass


class WeixinSessionExpiredError(WeixinAuthError):
    """Session expired error (errcode -14)."""

    pass


class WeixinSessionPausedError(WeixinAuthError):
    """Session is paused due to expiration, API calls are temporarily blocked."""

    def __init__(
        self,
        message: str,
        code: Optional[int] = None,
        response: Optional[dict] = None,
        remaining_seconds: Optional[int] = None,
    ):
        super().__init__(message, code, response)
        self.remaining_seconds = remaining_seconds


class WeixinTranscodeError(WeixinError):
    """Audio/video transcoding failed."""

    pass
