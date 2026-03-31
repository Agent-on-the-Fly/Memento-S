"""
Session Guard for handling session expiration (errcode -14) in Weixin SDK.

This module provides automatic pause/resume functionality when sessions expire,
preventing spam on expired tokens.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SessionGuard:
    """
    Guards API calls by pausing when session expires (errcode -14).

    Features:
    - Thread-safe using asyncio.Lock
    - Configurable pause duration (default 1 hour)
    - Auto-resume after duration expires
    - Support for multiple accounts
    - In-memory pause state (can be extended for persistence)

    Example:
        guard = SessionGuard(pause_duration_seconds=3600)

        # Check if account is paused before making API calls
        if guard.is_paused("account_123"):
            remaining = guard.get_remaining_pause_time("account_123")
            raise WeixinSessionPausedError(f"Session paused, retry in {remaining}s")

        # When session expires (errcode -14), pause the account
        guard.pause("account_123")

        # Manually resume if needed
        guard.resume("account_123")
    """

    def __init__(self, pause_duration_seconds: int = 3600):
        """
        Initialize SessionGuard.

        Args:
            pause_duration_seconds: Duration to pause accounts for (default 3600 = 1 hour)
        """
        self.pause_duration_seconds = pause_duration_seconds
        # account_id -> datetime when pause expires
        self._paused_accounts: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def is_paused(self, account_id: str) -> bool:
        """
        Check if an account is currently paused.

        Args:
            account_id: The account identifier

        Returns:
            True if account is paused and pause hasn't expired, False otherwise
        """
        async with self._lock:
            if account_id not in self._paused_accounts:
                return False

            expires_at = self._paused_accounts[account_id]
            now = datetime.utcnow()

            if now >= expires_at:
                # Pause has expired, auto-resume
                del self._paused_accounts[account_id]
                logger.info(f"Auto-resumed account {account_id} after pause duration")
                return False

            return True

    async def pause(self, account_id: str) -> None:
        """
        Pause an account for the configured duration.

        Args:
            account_id: The account identifier to pause
        """
        async with self._lock:
            expires_at = datetime.utcnow() + timedelta(
                seconds=self.pause_duration_seconds
            )
            self._paused_accounts[account_id] = expires_at
            logger.warning(
                f"Paused account {account_id} for {self.pause_duration_seconds}s "
                f"(until {expires_at.isoformat()})"
            )

    async def resume(self, account_id: str) -> bool:
        """
        Manually resume a paused account.

        Args:
            account_id: The account identifier to resume

        Returns:
            True if account was paused and is now resumed, False if not paused
        """
        async with self._lock:
            if account_id in self._paused_accounts:
                del self._paused_accounts[account_id]
                logger.info(f"Manually resumed account {account_id}")
                return True
            return False

    async def get_remaining_pause_time(self, account_id: str) -> Optional[int]:
        """
        Get the remaining pause time in seconds for an account.

        Args:
            account_id: The account identifier

        Returns:
            Seconds remaining in pause, or None if not paused
        """
        async with self._lock:
            if account_id not in self._paused_accounts:
                return None

            expires_at = self._paused_accounts[account_id]
            now = datetime.utcnow()

            if now >= expires_at:
                # Pause has expired
                del self._paused_accounts[account_id]
                return None

            remaining = int((expires_at - now).total_seconds())
            return remaining

    async def check_session_error(self, error_code: int, account_id: str) -> bool:
        """
        Check if error is session expired and pause the account if so.

        Args:
            error_code: The error code from API response
            account_id: The account identifier

        Returns:
            True if error was session expired (-14) and account is now paused,
            False otherwise
        """
        if error_code == -14:
            await self.pause(account_id)
            return True
        return False

    async def get_all_paused_accounts(self) -> Dict[str, int]:
        """
        Get all currently paused accounts with their remaining seconds.

        Returns:
            Dict mapping account_id to remaining seconds
        """
        async with self._lock:
            result = {}
            now = datetime.utcnow()
            expired_accounts = []

            for account_id, expires_at in self._paused_accounts.items():
                if now >= expires_at:
                    expired_accounts.append(account_id)
                else:
                    remaining = int((expires_at - now).total_seconds())
                    result[account_id] = remaining

            # Clean up expired accounts
            for account_id in expired_accounts:
                del self._paused_accounts[account_id]
                logger.info(
                    f"Auto-resumed account {account_id} (get_all_paused_accounts)"
                )

            return result

    async def clear_all(self) -> None:
        """Clear all paused accounts (useful for testing or reset)."""
        async with self._lock:
            count = len(self._paused_accounts)
            self._paused_accounts.clear()
            logger.info(f"Cleared all {count} paused accounts")
