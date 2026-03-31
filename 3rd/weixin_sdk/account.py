"""
Account management and state persistence for Weixin SDK.

This module provides functionality similar to TypeScript's auth/accounts.ts
and messaging/inbound.ts for account management and state persistence.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from .exceptions import WeixinError

logger = logging.getLogger(__name__)


# Default state directory
DEFAULT_STATE_DIR = Path.home() / ".weixin_sdk"


@dataclass
class WeixinAccount:
    """
    Represents a Weixin account configuration.

    Similar to TypeScript's ResolvedWeixinAccount.
    """

    account_id: str
    name: Optional[str] = None
    enabled: bool = True
    base_url: str = "https://ilinkai.weixin.qq.com"
    cdn_base_url: str = "https://novac2c.cdn.weixin.qq.com/c2c"
    token: Optional[str] = None
    route_tag: Optional[int] = None
    configured: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeixinAccount":
        """Create from dictionary."""
        return cls(**data)


class WeixinAccountManager:
    """
    Manages Weixin accounts and their state.

    Similar to TypeScript's auth/accounts.ts functionality.
    """

    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize account manager.

        Args:
            state_dir: Directory for storing account state.
                        Defaults to ~/.weixin_sdk
        """
        self.state_dir = state_dir or DEFAULT_STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._accounts: Dict[str, WeixinAccount] = {}
        self._context_tokens: Dict[str, str] = {}

        # Load existing accounts
        self._load_accounts()
        self._load_all_context_tokens()

    # -------------------------------------------------------------------------
    # Account Management
    # -------------------------------------------------------------------------

    def list_accounts(self) -> List[str]:
        """
        List all registered account IDs.

        Returns:
            List of account IDs
        """
        return list(self._accounts.keys())

    def register_account(
        self,
        account_id: str,
        account: Optional[WeixinAccount] = None,
    ) -> None:
        """
        Register a new account.

        Args:
            account_id: Unique account identifier
            account: Account configuration (optional)
        """
        if account is None:
            account = WeixinAccount(account_id=account_id)
        else:
            account.account_id = account_id

        self._accounts[account_id] = account
        self._persist_accounts()

        logger.info(f"Registered account: {account_id}")

    def unregister_account(self, account_id: str) -> None:
        """
        Unregister an account.

        Args:
            account_id: Account ID to unregister
        """
        if account_id in self._accounts:
            del self._accounts[account_id]
            self._persist_accounts()

            # Also clear context tokens
            self._clear_context_tokens_for_account(account_id)

            logger.info(f"Unregistered account: {account_id}")

    def get_account(self, account_id: str) -> Optional[WeixinAccount]:
        """
        Get account by ID.

        Args:
            account_id: Account ID

        Returns:
            Account configuration or None if not found
        """
        return self._accounts.get(account_id)

    def update_account(self, account_id: str, **kwargs) -> None:
        """
        Update account configuration.

        Args:
            account_id: Account ID
            **kwargs: Fields to update
        """
        if account_id not in self._accounts:
            raise WeixinError(f"Account not found: {account_id}")

        account = self._accounts[account_id]
        for key, value in kwargs.items():
            if hasattr(account, key):
                setattr(account, key, value)

        self._persist_accounts()
        logger.debug(f"Updated account: {account_id}")

    def resolve_account(
        self,
        account_id: Optional[str] = None,
    ) -> WeixinAccount:
        """
        Resolve account configuration.

        If account_id is not provided, returns the first configured account
        or raises an error if no accounts exist.

        Args:
            account_id: Optional account ID

        Returns:
            Resolved account configuration

        Raises:
            WeixinError: If no accounts configured or account not found
        """
        if account_id:
            account = self.get_account(account_id)
            if not account:
                raise WeixinError(f"Account not found: {account_id}")
            return account

        # Try to find first configured account
        configured = [
            acc for acc in self._accounts.values() if acc.configured and acc.enabled
        ]

        if not configured:
            if not self._accounts:
                raise WeixinError(
                    "No accounts registered. Use register_account() to add an account."
                )
            # Return first account even if not configured
            return list(self._accounts.values())[0]

        return configured[0]

    # -------------------------------------------------------------------------
    # Context Token Management (State Persistence)
    # -------------------------------------------------------------------------

    def set_context_token(
        self,
        account_id: str,
        user_id: str,
        token: str,
    ) -> None:
        """
        Store context token for account+user pair.

        Persists to disk for survival across restarts.

        Args:
            account_id: Account ID
            user_id: User ID
            token: Context token
        """
        key = f"{account_id}:{user_id}"
        self._context_tokens[key] = token
        self._persist_context_tokens(account_id)

        logger.debug(f"Set context token: {key}")

    def get_context_token(
        self,
        account_id: str,
        user_id: str,
    ) -> Optional[str]:
        """
        Get context token for account+user pair.

        Args:
            account_id: Account ID
            user_id: User ID

        Returns:
            Context token or None if not found
        """
        key = f"{account_id}:{user_id}"
        return self._context_tokens.get(key)

    def find_accounts_by_context_token(
        self,
        user_id: str,
    ) -> List[str]:
        """
        Find all account IDs that have a context token for the given user.

        Args:
            user_id: User ID

        Returns:
            List of account IDs
        """
        accounts = []
        for key in self._context_tokens.keys():
            if key.endswith(f":{user_id}"):
                account_id = key.rsplit(":", 1)[0]
                accounts.append(account_id)
        return accounts

    def clear_context_tokens(self, account_id: str) -> None:
        """
        Clear all context tokens for an account.

        Args:
            account_id: Account ID
        """
        self._clear_context_tokens_for_account(account_id)
        self._persist_context_tokens(account_id)

        logger.info(f"Cleared context tokens for account: {account_id}")

    # -------------------------------------------------------------------------
    # Persistence (Private Methods)
    # -------------------------------------------------------------------------

    def _get_accounts_file(self) -> Path:
        """Get path to accounts index file."""
        return self.state_dir / "accounts.json"

    def _get_context_tokens_file(self, account_id: str) -> Path:
        """Get path to context tokens file for an account."""
        accounts_dir = self.state_dir / "accounts"
        accounts_dir.mkdir(exist_ok=True)
        return accounts_dir / f"{account_id}.tokens.json"

    def _persist_accounts(self) -> None:
        """Persist accounts to disk."""
        accounts_file = self._get_accounts_file()

        data = {
            account_id: account.to_dict()
            for account_id, account in self._accounts.items()
        }

        # Write atomically
        temp_file = accounts_file.with_suffix(".tmp")
        temp_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temp_file.replace(accounts_file)

        logger.debug(f"Persisted {len(data)} accounts")

    def _load_accounts(self) -> None:
        """Load accounts from disk."""
        accounts_file = self._get_accounts_file()

        if not accounts_file.exists():
            return

        try:
            data = json.loads(accounts_file.read_text(encoding="utf-8"))
            self._accounts = {
                account_id: WeixinAccount.from_dict(account_data)
                for account_id, account_data in data.items()
            }
            logger.info(f"Loaded {len(self._accounts)} accounts")
        except Exception as e:
            logger.error(f"Failed to load accounts: {e}")
            self._accounts = {}

    def _persist_context_tokens(self, account_id: str) -> None:
        """Persist context tokens for an account to disk."""
        tokens_file = self._get_context_tokens_file(account_id)

        # Collect tokens for this account
        prefix = f"{account_id}:"
        tokens = {}
        for key, token in self._context_tokens.items():
            if key.startswith(prefix):
                user_id = key[len(prefix) :]
                tokens[user_id] = token

        # Write atomically
        temp_file = tokens_file.with_suffix(".tmp")
        temp_file.write_text(
            json.dumps(tokens, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temp_file.replace(tokens_file)

        logger.debug(f"Persisted {len(tokens)} tokens for account {account_id}")

    def _load_all_context_tokens(self) -> None:
        """Load all context tokens from disk."""
        accounts_dir = self.state_dir / "accounts"

        if not accounts_dir.exists():
            return

        total_loaded = 0
        for tokens_file in accounts_dir.glob("*.tokens.json"):
            try:
                account_id = tokens_file.stem.replace(".tokens", "")
                data = json.loads(tokens_file.read_text(encoding="utf-8"))

                for user_id, token in data.items():
                    key = f"{account_id}:{user_id}"
                    self._context_tokens[key] = token
                    total_loaded += 1
            except Exception as e:
                logger.error(f"Failed to load tokens from {tokens_file}: {e}")

        if total_loaded > 0:
            logger.info(f"Loaded {total_loaded} context tokens")

    def _clear_context_tokens_for_account(self, account_id: str) -> None:
        """Clear context tokens from memory for an account."""
        prefix = f"{account_id}:"
        keys_to_remove = [
            key for key in self._context_tokens.keys() if key.startswith(prefix)
        ]
        for key in keys_to_remove:
            del self._context_tokens[key]

        # Also delete file
        tokens_file = self._get_context_tokens_file(account_id)
        if tokens_file.exists():
            tokens_file.unlink()


# Convenience functions


def get_default_account_manager() -> WeixinAccountManager:
    """Get default account manager instance."""
    return WeixinAccountManager()
