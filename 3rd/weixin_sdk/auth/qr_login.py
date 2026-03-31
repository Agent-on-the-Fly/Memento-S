"""
QR Code login flow for Weixin SDK.
Mirrors TypeScript implementation from src/auth/login-qr.ts
"""

import asyncio
import sys
from pathlib import Path
import io
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import aiohttp
import qrcode

from weixin_sdk.exceptions import WeixinAPIError, WeixinAuthError, WeixinTimeoutError

# Setup logger - try to use project utils.logger, fallback to standard logging
try:
    # Add project root to path for utils import
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class LoginStatus(Enum):
    """Login status enumeration."""

    WAIT = "wait"
    SCANNED = "scaned"
    CONFIRMED = "confirmed"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


@dataclass
class LoginResult:
    """Result of successful QR login."""

    bot_token: str
    ilink_bot_id: str
    user_id: str


@dataclass
class QRCodeResponse:
    """QR code response from API."""

    qr_url: str
    ticket: str


class QRLoginManager:
    """Manager for QR code login flow."""

    # Default bot_type matching TypeScript implementation
    DEFAULT_BOT_TYPE = "3"

    # Client version header matching TypeScript
    CLIENT_VERSION = "1"

    # Long poll timeout for status check (35 seconds)
    QR_LONG_POLL_TIMEOUT_MS = 35_000

    def __init__(self, base_url: str):
        """
        Initialize QR login manager.

        Args:
            base_url: API base URL (e.g., https://ilinkai.weixin.qq.com)
        """
        self.base_url = base_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            # Create session with proper headers
            headers = {
                "iLink-App-ClientVersion": self.CLIENT_VERSION,
                "Accept": "application/json",
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    async def _close_session(self) -> None:
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
        timeout_ms: Optional[int] = None,
        retry_count: int = 3,
    ) -> dict:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (with or without leading /)
            data: Request body data (for POST/PUT)
            params: URL query parameters (for GET)
            timeout_ms: Request timeout in milliseconds
            retry_count: Number of retries for network errors

        Returns:
            Response JSON as dict

        Raises:
            WeixinAPIError: On API errors
            WeixinTimeoutError: On timeout
        """
        session = await self._get_session()

        # Ensure endpoint starts with /
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        url = f"{self.base_url}{endpoint}"

        # Set timeout
        if timeout_ms:
            timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
        else:
            timeout = aiohttp.ClientTimeout(total=30)

        last_error = None
        for attempt in range(retry_count):
            try:
                # Make request with appropriate parameters
                request_kwargs = {"timeout": timeout}

                if method.upper() == "GET":
                    # GET requests use params
                    if params:
                        request_kwargs["params"] = params
                else:
                    # POST/PUT requests use json body
                    if data:
                        request_kwargs["json"] = data

                async with session.request(method, url, **request_kwargs) as response:
                    # Read raw text first
                    raw_text = await response.text()

                    if response.status != 200:
                        raise WeixinAPIError(
                            f"HTTP {response.status}: {response.reason}",
                            code=response.status,
                        )

                    # Parse JSON from raw text
                    import json

                    try:
                        result = json.loads(raw_text)
                    except json.JSONDecodeError:
                        # If not JSON, wrap in dict
                        result = {"text": raw_text}

                    # Check for API-level errors (if response has 'ret' field)
                    if isinstance(result, dict) and result.get("ret") not in (0, None):
                        raise WeixinAPIError(
                            result.get("errmsg", f"API error: {result.get('ret')}"),
                            code=result.get("ret"),
                            response=result,
                        )

                    return result

            except asyncio.TimeoutError as e:
                last_error = WeixinTimeoutError(f"Request timeout: {url}")
                if attempt < retry_count - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    await asyncio.sleep(wait_time)
                else:
                    raise last_error

            except aiohttp.ClientError as e:
                last_error = WeixinAPIError(f"Network error: {e}")
                if attempt < retry_count - 1:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    raise last_error

        raise last_error or WeixinAPIError("Request failed")

    async def fetch_qr_code(self, bot_type: Optional[str] = None) -> QRCodeResponse:
        """
        Fetch QR code from API.

        Uses GET /ilink/bot/get_bot_qrcode?bot_type=X
        Matching TypeScript implementation.

        Args:
            bot_type: Bot type (default: "3")

        Returns:
            QRCodeResponse with qr_url and ticket

        Raises:
            WeixinAPIError: On API errors
        """
        if bot_type is None:
            bot_type = self.DEFAULT_BOT_TYPE

        # Build endpoint with query params (matching TypeScript)
        params = {"bot_type": bot_type}
        endpoint = "/ilink/bot/get_bot_qrcode"

        # Make GET request with query params
        result = await self._make_request("GET", endpoint, params=params)

        # Log full response to see all available fields
        logger.info(f"[QRLoginManager] fetch_qr_code response: {result}")

        # Extract QR code URL and ticket from response
        # Response format: { "qrcode": "...", "qrcode_img_content": "..." }
        qr_code = result.get("qrcode") or result.get("qr_code")
        qr_img_content = result.get("qrcode_img_content") or result.get(
            "qr_code_img_content"
        )

        # Check for expire time in response
        expire_time = (
            result.get("expire_time")
            or result.get("expire")
            or result.get("ttl")
            or result.get("timeout")
        )
        if expire_time:
            logger.info(f"[QRLoginManager] QR code expire_time: {expire_time}")

        if not qr_code:
            raise WeixinAPIError(
                f"Invalid QR code response: missing qrcode field. Response: {result}",
                response=result,
            )

        # Use qrcode_img_content from API response (matching TypeScript)
        # This is already a complete URL like:
        # https://liteapp.weixin.qq.com/q/7GiQu1?qrcode=xxx&bot_type=3
        qr_url = qr_img_content
        if not qr_url:
            raise WeixinAPIError(
                f"Invalid QR code response: missing qrcode_img_content field. Response: {result}",
                response=result,
            )

        return QRCodeResponse(qr_url=qr_url, ticket=qr_code)

    def display_qr(self, qr_url: str) -> None:
        """
        Display QR code in terminal using qrcode library.

        Args:
            qr_url: URL to encode in QR code
        """
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=2,
        )
        qr.add_data(qr_url)
        qr.make(fit=True)

        # Generate terminal-friendly output
        print("\n" + "=" * 50)
        print("       SCAN QR CODE TO LOGIN")
        print("=" * 50)
        qr.print_ascii(invert=True)
        print("=" * 50)
        print(f"QR URL: {qr_url}")
        print("=" * 50 + "\n")

    async def poll_status(self, qrcode: str) -> tuple[LoginStatus, Optional[dict]]:
        """
        Poll login status.

        Uses GET /ilink/bot/get_qrcode_status?qrcode=XXX
        Matching TypeScript implementation with 35s long poll.

        Args:
            qrcode: QR code from fetch_qr_code (used as ticket)

        Returns:
            Tuple of (LoginStatus, data dict with login info if confirmed)

        Raises:
            WeixinAPIError: On API errors
        """
        # Build endpoint with query params (matching TypeScript)
        params = {"qrcode": qrcode}
        endpoint = "/ilink/bot/get_qrcode_status"

        # Make GET request with long poll timeout (35s)
        result = await self._make_request(
            "GET", endpoint, params=params, timeout_ms=self.QR_LONG_POLL_TIMEOUT_MS
        )

        # Parse status from response
        # Response format: { "status": "wait|scaned|confirmed|expired", ... }
        status_str = result.get("status", "").lower()

        if status_str == "wait":
            return LoginStatus.WAIT, None
        elif status_str == "scaned":
            return LoginStatus.SCANNED, None
        elif status_str == "confirmed":
            # Extract login data
            login_data = {
                "bot_token": result.get("bot_token"),
                "ilink_bot_id": result.get("ilink_bot_id"),
                "user_id": result.get("ilink_user_id") or result.get("user_id"),
                "baseurl": result.get("baseurl"),
            }
            return LoginStatus.CONFIRMED, login_data
        elif status_str == "expired":
            return LoginStatus.EXPIRED, None
        else:
            return LoginStatus.UNKNOWN, result

    async def start_login(
        self,
        timeout_seconds: int = 300,
        max_qr_refreshes: int = 3,
        poll_interval: float = 1.0,
    ) -> LoginResult:
        """
        Start QR login process with polling.

        Args:
            timeout_seconds: Maximum time to wait for login (default 5 minutes)
            max_qr_refreshes: Maximum QR refresh attempts on expiration
            poll_interval: Seconds between status polls

        Returns:
            LoginResult with bot_token, ilink_bot_id, and user_id

        Raises:
            WeixinTimeoutError: On login timeout
            WeixinAuthError: On authentication failure
            WeixinAPIError: On API errors
        """
        start_time = asyncio.get_event_loop().time()
        qr_refresh_count = 0

        while True:
            # Check overall timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_seconds:
                await self._close_session()
                raise WeixinTimeoutError(
                    f"Login timeout after {timeout_seconds} seconds"
                )

            # Fetch new QR code
            try:
                qr_response = await self.fetch_qr_code()
            except WeixinAPIError as e:
                await self._close_session()
                raise WeixinAuthError(f"Failed to fetch QR code: {e}")

            self.display_qr(qr_response.qr_url)

            # Poll for status
            while True:
                await asyncio.sleep(poll_interval)

                # Check timeout during polling
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout_seconds:
                    await self._close_session()
                    raise WeixinTimeoutError(
                        f"Login timeout after {timeout_seconds} seconds"
                    )

                try:
                    status, data = await self.poll_status(qr_response.ticket)
                except WeixinAPIError as e:
                    # Network error during polling - continue trying
                    print(f"Poll error: {e}, retrying...")
                    continue

                if status == LoginStatus.WAIT:
                    print("Waiting for QR scan...")
                    continue

                elif status == LoginStatus.SCANNED:
                    print("QR scanned, waiting for confirmation...")
                    continue

                elif status == LoginStatus.CONFIRMED:
                    # Login successful
                    if not data:
                        await self._close_session()
                        raise WeixinAuthError("Login confirmed but no data returned")

                    bot_token = data.get("bot_token") or data.get("botToken")
                    ilink_bot_id = data.get("ilink_bot_id") or data.get("ilinkBotId")
                    user_id = data.get("user_id") or data.get("userId")

                    if not bot_token or not ilink_bot_id:
                        await self._close_session()
                        raise WeixinAuthError(
                            "Login confirmed but missing bot_token or ilink_bot_id",
                            response=data,
                        )

                    await self._close_session()
                    return LoginResult(
                        bot_token=bot_token,
                        ilink_bot_id=ilink_bot_id,
                        user_id=user_id or "",
                    )

                elif status == LoginStatus.EXPIRED:
                    qr_refresh_count += 1
                    if qr_refresh_count > max_qr_refreshes:
                        await self._close_session()
                        raise WeixinAuthError(
                            f"QR code expired after {max_qr_refreshes} refresh attempts"
                        )

                    print(
                        f"QR expired, refreshing... ({qr_refresh_count}/{max_qr_refreshes})"
                    )
                    break  # Break inner poll loop to fetch new QR

                else:
                    print(f"Unknown status: {status}, continuing...")
                    continue

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_session()
