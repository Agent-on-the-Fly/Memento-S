"""
Sync Buffer Manager for Weixin SDK.

Provides persistence for sync cursors to resume message polling across restarts.
Supports atomic writes, thread-safe operations, and graceful handling of corrupted files.
"""

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Try to import platform-specific file locking
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    logger.debug("fcntl not available (Windows), using threading lock only")


class SyncBufferManager:
    """
    Manager for persisting sync cursors to disk.

    Provides thread-safe operations for saving and loading sync cursors,
    with atomic writes and graceful handling of corrupted files.

    File format: JSON with structure {"cursor": "...", "timestamp": 1234567890}

    Example:
        manager = SyncBufferManager(state_dir=Path("~/.weixin_sdk"))

        # Save cursor
        manager.save_cursor("account_123", "cursor_xyz")

        # Load cursor
        cursor = manager.load_cursor("account_123")

        # List all cursors
        cursors = manager.list_cursors()

        # Delete cursor when done
        manager.delete_cursor("account_123")
    """

    def __init__(self, state_dir: Path):
        """
        Initialize the sync buffer manager.

        Args:
            state_dir: Directory to store cursor files (will be created if not exists)
        """
        self._state_dir = Path(state_dir).expanduser().resolve()
        self._cursors_dir = self._state_dir / "cursors"
        self._lock = threading.RLock()

        # Ensure directories exist
        self._cursors_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"SyncBufferManager initialized with state_dir: {self._state_dir}")

    def _get_cursor_path(self, account_id: str) -> Path:
        """
        Get the file path for a cursor.

        Args:
            account_id: Account identifier

        Returns:
            Path to the cursor file
        """
        # Sanitize account_id to prevent directory traversal
        safe_id = self._sanitize_filename(account_id)
        return self._cursors_dir / f"{safe_id}.json"

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename to prevent directory traversal and invalid characters.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename safe for filesystem use
        """
        # Replace path separators and other unsafe characters
        unsafe_chars = '/\\<>:"|?*\x00-\x1f'
        sanitized = "".join("_" if c in unsafe_chars else c for c in filename)
        # Limit length to avoid filesystem issues
        return sanitized[:255] if sanitized else "_"

    def _atomic_write(self, file_path: Path, data: dict) -> None:
        """
        Write data atomically using temp file and rename.

        Args:
            file_path: Target file path
            data: Data to serialize as JSON

        Raises:
            IOError: If write fails
        """
        # Write to temp file in same directory for atomic rename
        temp_fd = None
        temp_path = None

        try:
            # Create temp file in the same directory for atomic rename
            temp_fd, temp_path = tempfile.mkstemp(
                dir=file_path.parent, prefix=f".tmp_{file_path.stem}_", suffix=".json"
            )

            # Write JSON data
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            temp_fd = None  # File is now closed

            # Atomic rename
            os.replace(temp_path, file_path)

            # Ensure directory is synced (best effort, Unix only)
            try:
                if hasattr(os, "O_DIRECTORY"):
                    dir_fd = os.open(file_path.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
            except OSError:
                pass  # Ignore directory sync failures

        except Exception:
            # Clean up temp file on failure
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except OSError:
                    pass
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            raise

    def _read_cursor_file(self, file_path: Path) -> Optional[dict]:
        """
        Read and parse a cursor file with error handling.

        Args:
            file_path: Path to cursor file

        Returns:
            Parsed JSON data or None if file is corrupted/missing
        """
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # Acquire shared lock for reading (Unix only)
                if HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    return json.load(f)
                finally:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted cursor file {file_path}: {e}")
            return None
        except OSError as e:
            logger.warning(f"Error reading cursor file {file_path}: {e}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # Acquire shared lock for reading
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    return json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted cursor file {file_path}: {e}")
            return None
        except OSError as e:
            logger.warning(f"Error reading cursor file {file_path}: {e}")
            return None

    def save_cursor(self, account_id: str, cursor: str) -> None:
        """
        Save sync cursor to disk.

        Uses atomic writes (write to temp file, then rename) to ensure
        data integrity even if process crashes during write.

        Args:
            account_id: Account identifier
            cursor: Sync cursor from getUpdates response

        Raises:
            IOError: If write fails
            ValueError: If account_id or cursor is empty
        """
        if not account_id:
            raise ValueError("account_id cannot be empty")
        if not cursor:
            raise ValueError("cursor cannot be empty")

        with self._lock:
            file_path = self._get_cursor_path(account_id)

            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data with metadata
            data = {
                "cursor": cursor,
                "account_id": account_id,
                "timestamp": int(os.path.getmtime(file_path))
                if file_path.exists()
                else None,
            }
            # Set timestamp to current time
            data["timestamp"] = int(__import__("time").time())

            try:
                self._atomic_write(file_path, data)
                logger.debug(f"Saved cursor for account {account_id}: {cursor[:20]}...")
            except OSError as e:
                logger.error(f"Failed to save cursor for account {account_id}: {e}")
                raise

    def load_cursor(self, account_id: str) -> Optional[str]:
        """
        Load sync cursor from disk.

        Gracefully handles corrupted files by returning None and logging a warning.

        Args:
            account_id: Account identifier

        Returns:
            Sync cursor string if found, None otherwise
        """
        if not account_id:
            return None

        with self._lock:
            file_path = self._get_cursor_path(account_id)
            data = self._read_cursor_file(file_path)

            if data is None:
                logger.debug(f"No cursor found for account {account_id}")
                return None

            cursor = data.get("cursor")
            if cursor:
                logger.debug(
                    f"Loaded cursor for account {account_id}: {cursor[:20]}..."
                )
            return cursor

    def delete_cursor(self, account_id: str) -> bool:
        """
        Delete saved cursor for an account.

        Args:
            account_id: Account identifier

        Returns:
            True if cursor was deleted, False if it didn't exist
        """
        if not account_id:
            return False

        with self._lock:
            file_path = self._get_cursor_path(account_id)

            if not file_path.exists():
                logger.debug(f"No cursor to delete for account {account_id}")
                return False

            try:
                file_path.unlink()
                logger.debug(f"Deleted cursor for account {account_id}")
                return True
            except OSError as e:
                logger.error(f"Failed to delete cursor for account {account_id}: {e}")
                return False

    def list_cursors(self) -> Dict[str, str]:
        """
        List all saved cursors.

        Returns:
            Dictionary mapping account_id to cursor string.
            Corrupted files are skipped and logged as warnings.
        """
        cursors: Dict[str, str] = {}

        with self._lock:
            if not self._cursors_dir.exists():
                return cursors

            for file_path in self._cursors_dir.glob("*.json"):
                account_id = file_path.stem
                data = self._read_cursor_file(file_path)

                if data and "cursor" in data:
                    cursors[account_id] = data["cursor"]
                else:
                    logger.warning(f"Skipping corrupted cursor file: {file_path}")

        logger.debug(f"Listed {len(cursors)} cursors")
        return cursors

    def cursor_exists(self, account_id: str) -> bool:
        """
        Check if a cursor exists for an account.

        Args:
            account_id: Account identifier

        Returns:
            True if cursor exists and is readable
        """
        if not account_id:
            return False

        with self._lock:
            file_path = self._get_cursor_path(account_id)
            return self._read_cursor_file(file_path) is not None

    def clear_all_cursors(self) -> int:
        """
        Delete all saved cursors.

        Returns:
            Number of cursors deleted
        """
        count = 0

        with self._lock:
            if not self._cursors_dir.exists():
                return 0

            for file_path in self._cursors_dir.glob("*.json"):
                try:
                    file_path.unlink()
                    count += 1
                except OSError as e:
                    logger.warning(f"Failed to delete cursor file {file_path}: {e}")

        logger.info(f"Cleared {count} cursors")
        return count

    def get_cursor_metadata(self, account_id: str) -> Optional[Dict]:
        """
        Get full metadata for a cursor.

        Args:
            account_id: Account identifier

        Returns:
            Dictionary with cursor, timestamp, etc. or None if not found
        """
        if not account_id:
            return None

        with self._lock:
            file_path = self._get_cursor_path(account_id)
            return self._read_cursor_file(file_path)
