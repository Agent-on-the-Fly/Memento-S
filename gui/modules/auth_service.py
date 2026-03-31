"""
Authentication service for Memento-S GUI.

Handles login via verification code, token management, and user state persistence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from middleware.config import g_config
from utils.http_client import HttpClient

logger = logging.getLogger(__name__)

_TOKEN_FILE = "auth_token.json"


class AuthService:
    """Manages authentication state, API calls, and token persistence."""

    def __init__(self):
        self._token: str | None = None
        self._token_type: str = "Bearer"
        self._user_info: dict[str, Any] = {}
        self._http_client: HttpClient | None = None
        self._load_persisted_token()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        try:
            return g_config.auth.base_url.rstrip("/")
        except Exception:
            return ""

    @property
    def is_logged_in(self) -> bool:
        return self._token is not None

    @property
    def token(self) -> str | None:
        return self._token

    @property
    def user_info(self) -> dict[str, Any]:
        return self._user_info

    @property
    def display_name(self) -> str:
        """Best-effort display name from user_info."""
        if not self._user_info:
            return ""
        for key in ("nickname", "email", "phone", "uid"):
            val = self._user_info.get(key)
            if val:
                return str(val)
        return ""

    @property
    def _http(self) -> HttpClient:
        if self._http_client is None or self._http_client.base_url != self.base_url:
            self._http_client = HttpClient(base_url=self.base_url)
        return self._http_client

    # ------------------------------------------------------------------
    # Token persistence
    # ------------------------------------------------------------------

    def _token_path(self) -> Path:
        return g_config.get_data_dir() / _TOKEN_FILE

    def _load_persisted_token(self):
        path = self._token_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._token = data.get("token")
            self._token_type = data.get("token_type", "Bearer")
            self._user_info = data.get("user_info", {})
            logger.info("[AuthService] Loaded persisted token")
        except Exception as e:
            logger.warning("[AuthService] Failed to load token file: %s", e)

    def _persist_token(self):
        path = self._token_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "token": self._token,
                "token_type": self._token_type,
                "user_info": self._user_info,
            }
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning("[AuthService] Failed to persist token: %s", e)

    def _clear_persisted_token(self):
        path = self._token_path()
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.warning("[AuthService] Failed to remove token file: %s", e)

    # ------------------------------------------------------------------
    # Auth API
    # ------------------------------------------------------------------

    async def send_verification_code(self, account: str) -> tuple[bool, str]:
        """Send verification code. Returns (success, message)."""
        result = await self._http.post(
            "/api/v1/auth/send_verification_code",
            json={"account": account},
        )
        return result.success, result.msg

    async def login(self, account: str, code: str) -> tuple[bool, str]:
        """Login with account + verification code. Returns (success, message)."""
        result = await self._http.post(
            "/api/v1/auth/login",
            json={"account": account, "code": code},
        )
        if result.success:
            data = result.data or {}
            self._token = data.get("token")
            self._token_type = data.get("token_type", "Bearer")
            self._user_info = data.get("user_info", {})
            self._persist_token()
        return result.success, result.msg

    def clear_token(self) -> None:
        """清除本地 token 状态（用于 401 强制登出，不调用 logout API）。"""
        self._token = None
        self._token_type = "Bearer"
        self._user_info = {}
        self._clear_persisted_token()
        logger.info("[AuthService] Token cleared due to 401")

    async def logout(self) -> tuple[bool, str]:
        """Call logout API then clear local state. Returns (success, message)."""
        if self._token:
            headers = {"authorization": f"{self._token_type} {self._token}"}
            result = await self._http.post("/api/v1/auth/logout", headers=headers)
            logger.info(
                "[AuthService] logout: success=%s msg=%s", result.success, result.msg
            )
            if not result.success:
                return False, result.msg
        self._token = None
        self._token_type = "Bearer"
        self._user_info = {}
        self._clear_persisted_token()
        return True, ""
