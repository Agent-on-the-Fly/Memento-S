"""
Authentication module for Weixin SDK.
Provides QR code login flow functionality.
"""

from weixin_sdk.auth.qr_login import (
    LoginResult,
    LoginStatus,
    QRCodeResponse,
    QRLoginManager,
)

__all__ = [
    "QRLoginManager",
    "LoginResult",
    "LoginStatus",
    "QRCodeResponse",
]
