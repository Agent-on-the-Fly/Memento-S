"""
Auto Update Manager for Memento-S GUI.

Handles automatic update checking, downloading, and installation.
Features:
    - Automatic update check on startup
    - Background silent download
    - Download progress persistence (resume support)
    - Notification when download completes
    - User confirmation before installation
    - Cross-platform support (macOS, Windows, Linux)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import shlex
import shutil
import subprocess
import sys
import time
import tarfile
import zipfile
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

import httpx
from packaging.version import parse as parse_version

from gui.i18n import t
from middleware.config import g_config
from utils.logger import logger


class UpdateStatus(Enum):
    """Update process status."""

    IDLE = auto()
    CHECKING = auto()
    AVAILABLE = auto()
    DOWNLOADING = auto()
    PAUSED = auto()
    DOWNLOADED = auto()
    INSTALLING = auto()
    COMPLETED = auto()
    ERROR = auto()
    CANCELLED = auto()


@dataclass
class UpdateInfo:
    """Information about available update."""

    version: str
    current_version: str
    download_url: str
    release_notes: str = ""
    force_update: bool = False
    published_at: str = ""
    size: int = 0
    checksum: str | None = None


@dataclass
class DownloadProgress:
    """Download progress tracking."""

    total_size: int = 0
    downloaded: int = 0
    start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    speed: float = 0.0

    @property
    def percentage(self) -> float:
        """Download percentage (0.0 to 1.0)."""
        if self.total_size <= 0:
            return 0.0
        return min(1.0, self.downloaded / self.total_size)

    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining in seconds."""
        if self.speed <= 0 or self.total_size <= 0:
            return float("inf")
        remaining = self.total_size - self.downloaded
        return remaining / self.speed


@dataclass
class CallbackGroup:
    """A group of event callbacks registered by a single listener."""

    on_status_change: Callable[[UpdateStatus], None] | None = None
    on_progress: Callable[[DownloadProgress], None] | None = None
    on_download_complete: Callable[[UpdateInfo], None] | None = None
    on_error: Callable[[str], None] | None = None


@dataclass
class UpdateCache:
    """Persistent update cache metadata."""

    version: str
    download_path: Path
    checksum: str | None = None
    downloaded_size: int = 0
    total_size: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    installed: bool = False
    force_update: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "download_path": str(self.download_path),
            "checksum": self.checksum,
            "downloaded_size": self.downloaded_size,
            "total_size": self.total_size,
            "timestamp": self.timestamp,
            "installed": self.installed,
            "force_update": self.force_update,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UpdateCache":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            download_path=Path(data["download_path"]),
            checksum=data.get("checksum"),
            downloaded_size=data.get("downloaded_size", 0),
            total_size=data.get("total_size", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            installed=data.get("installed", False),
            force_update=data.get("force_update", False),
        )


class AutoUpdateManager:
    """
    Manages automatic application updates.
    """

    CACHE_DIR = Path.home() / "memento_s" / "updates"
    CACHE_METADATA_FILE = CACHE_DIR / "cache.json"
    TEMP_DIR = Path.home() / "memento_s" / "temp" / "updates"

    STARTUP_CHECK_DELAY = 10
    MACOS_APP_TRANSLOCATION_MARKER = "/AppTranslocation/"

    def __init__(self):
        """Initialize the update manager."""
        self._status = UpdateStatus.IDLE
        self._current_update: UpdateInfo | None = None
        self._progress = DownloadProgress()
        self._download_task: asyncio.Task | None = None
        self._cancelled = False
        self._paused = False
        self._cache: UpdateCache | None = None
        self._config = g_config

        # True while running an auto (silent) check; False for manual checks
        self._is_auto_check = False

        # Timestamp of last successful check (epoch seconds).
        # Only lives in memory — resets on app restart so startup always checks.
        self._last_check_time: float = 0.0

        self._listeners: list[CallbackGroup] = []

        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)

        self._cleanup_stale_runtime_artifacts()
        self._load_cache()

    @property
    def status(self) -> UpdateStatus:
        """Current update status."""
        return self._status

    @property
    def is_auto_check(self) -> bool:
        """Whether the current check/download cycle was triggered automatically."""
        return self._is_auto_check

    @property
    def current_update(self) -> UpdateInfo | None:
        """Current update information."""
        return self._current_update

    @property
    def progress(self) -> DownloadProgress:
        """Current download progress."""
        return self._progress

    def get_current_app_bundle_path(self) -> Path | None:
        """Return the current macOS .app bundle path when running from a bundle."""
        if platform.system().lower() != "darwin":
            return None

        exe_path = Path(sys.executable).resolve()
        for parent in exe_path.parents:
            if parent.suffix == ".app" and parent.is_dir():
                return parent
        return None

    def get_macos_update_block_reason(self) -> str | None:
        """Return a user-facing reason why self-update should be blocked on macOS."""
        if platform.system().lower() != "darwin":
            return None

        current_app = self.get_current_app_bundle_path()
        if not current_app:
            return None

        if self._is_macos_app_translocated(current_app):
            return t(
                "update.blocked_translocated",
                default="当前处于隔离环境，无法升级。请将应用移到 Applications 后重新打开。",
            )

        return None

    def get_runtime_environment_info(self) -> dict[str, Any]:
        """Collect runtime diagnostics useful for update/relaunch troubleshooting."""
        info: dict[str, Any] = {
            "platform": platform.system().lower(),
            "sys_executable": str(Path(sys.executable).resolve()),
            "frozen": bool(getattr(sys, "frozen", False)),
        }

        if info["platform"] != "darwin":
            return info

        current_app = self.get_current_app_bundle_path()
        info["app_bundle"] = str(current_app) if current_app else ""
        info["app_translocated"] = (
            self._is_macos_app_translocated(current_app) if current_app else False
        )
        info["quarantine"] = self._read_macos_quarantine(current_app)
        return info

    def log_runtime_environment_info(self) -> None:
        """Write runtime environment diagnostics into the application log."""
        info = self.get_runtime_environment_info()
        logger.info(
            "[AutoUpdate] Runtime environment: "
            f"platform={info.get('platform')} "
            f"frozen={info.get('frozen')} "
            f"sys_executable={info.get('sys_executable')}"
        )

        if info.get("platform") == "darwin":
            logger.info(
                "[AutoUpdate] macOS app bundle: "
                f"path={info.get('app_bundle') or '<none>'} "
                f"translocated={info.get('app_translocated')} "
                f"quarantine={info.get('quarantine')}"
            )

    def _cleanup_stale_runtime_artifacts(self) -> None:
        """Remove stale update artifacts that can confuse macOS app identity/UI."""
        if platform.system().lower() != "darwin":
            return

        try:
            patterns = ("*.backup.app", "*.backup_bundle")
            for pattern in patterns:
                for backup_app in self.TEMP_DIR.glob(pattern):
                    shutil.rmtree(backup_app, ignore_errors=True)
                    logger.info(
                        f"[AutoUpdate] Removed stale backup app artifact: {backup_app}"
                    )
        except Exception as e:
            logger.warning(f"[AutoUpdate] Failed to cleanup stale artifacts: {e}")

    @property
    def has_cached_update(self) -> bool:
        """Check if there's a cached update ready to install.

        Returns True only when the cached file is fully downloaded AND passes
        archive validation.  A corrupt file is auto-cleared; a partial file
        is left for resume.
        """
        if self._cache is None or self._cache.installed:
            return False
        if self._status == UpdateStatus.DOWNLOADING:
            return False
        if not self._cache.download_path.exists():
            return False
        if not self._is_valid_archive(self._cache.download_path):
            if self.has_partial_cache:
                logger.info(
                    "[AutoUpdate] Cached file is partial download, keeping for resume"
                )
                return False
            logger.warning("[AutoUpdate] Cached file is corrupted, clearing cache")
            self.clear_cache()
            return False
        return True

    @property
    def has_partial_cache(self) -> bool:
        """Check if there's a partially downloaded cache that can be resumed.

        Distinguishes a partial file from a corrupt complete file by checking:
        the file exists, is non-empty, is smaller than total_size (if known),
        and does NOT pass archive validation (i.e. it's genuinely incomplete).
        """
        if self._cache is None or self._cache.installed:
            return False
        if self._status == UpdateStatus.DOWNLOADING:
            return False
        if not self._cache.download_path.exists():
            return False
        file_size = self._cache.download_path.stat().st_size
        if file_size <= 0:
            return False
        if self._cache.total_size > 0 and file_size >= self._cache.total_size:
            return False
        if self._is_valid_archive(self._cache.download_path):
            return False
        return True

    def add_listener(self, **kwargs: Any) -> CallbackGroup:
        """Register a group of event callbacks. Returns the group for later removal."""
        group = CallbackGroup(**kwargs)
        self._listeners.append(group)
        return group

    def remove_listener(self, group: CallbackGroup) -> None:
        """Remove a previously registered callback group."""
        try:
            self._listeners.remove(group)
        except ValueError:
            pass

    def _emit(self, event: str, *args: Any) -> None:
        """Dispatch an event to all registered listeners."""
        for listener in self._listeners:
            cb = getattr(listener, event, None)
            if cb is not None:
                try:
                    cb(*args)
                except Exception as e:
                    logger.error(f"[AutoUpdate] Callback error ({event}): {e}")

    def _set_status(self, status: UpdateStatus):
        """Set status and trigger callback.

        Important: _emit() fires BEFORE _is_auto_check is reset.  Listeners
        that inspect ``is_auto_check`` during the on_status_change callback
        will still see the value that was active during the check/download
        cycle.  Do not reorder these two steps.
        """
        self._status = status
        self._emit("on_status_change", status)
        # Reset the auto-check flag on terminal states so subsequent manual
        # checks won't be mistaken for automatic ones.
        if status in (
            UpdateStatus.IDLE,
            UpdateStatus.DOWNLOADED,
            UpdateStatus.ERROR,
            UpdateStatus.CANCELLED,
        ):
            self._is_auto_check = False
        logger.info(f"[AutoUpdate] Status: {status.name}")

    def _notify_progress(self):
        """Notify progress update."""
        self._emit("on_progress", self._progress)

    def _notify_error(self, message: str):
        """Notify error."""
        logger.error(f"[AutoUpdate] Error: {message}")
        self._emit("on_error", message)

    def _notify_download_complete(self, update_info: UpdateInfo):
        """Notify download complete."""
        self._emit("on_download_complete", update_info)

    def _load_cache(self):
        """Load cached update metadata."""
        try:
            if self.CACHE_METADATA_FILE.exists():
                with open(self.CACHE_METADATA_FILE, "r") as f:
                    data = json.load(f)
                    self._cache = UpdateCache.from_dict(data)
                    logger.info(f"[AutoUpdate] Loaded cached: {self._cache.version}")
        except Exception as e:
            logger.warning(f"[AutoUpdate] Failed to load cache: {e}")
            self._cache = None

    def _save_cache(self):
        """Save update cache metadata."""
        try:
            if self._cache:
                with open(self.CACHE_METADATA_FILE, "w") as f:
                    json.dump(self._cache.to_dict(), f, indent=2)
                logger.info(f"[AutoUpdate] Saved cache: {self._cache.version}")
        except Exception as e:
            logger.error(f"[AutoUpdate] Failed to save cache: {e}")

    def clear_cache(self):
        """Clear update cache."""
        try:
            if self._cache and self._cache.download_path.exists():
                self._cache.download_path.unlink()
            if self.CACHE_METADATA_FILE.exists():
                self.CACHE_METADATA_FILE.unlink()
            self._cache = None
            logger.info("[AutoUpdate] Cache cleared")
        except Exception as e:
            logger.error(f"[AutoUpdate] Failed to clear cache: {e}")

    def _get_current_version(self) -> str:
        """Get current application version."""
        import sys

        # 版本号管理：优先从 version.py 读取（开发模式），失败则从包元数据读取（打包模式）
        try:
            import version

            current_version = version.version
            logger.info(
                f"[AutoUpdate] Current version (from version.py): {current_version}"
            )
            return current_version
        except ImportError:
            try:
                from importlib.metadata import version as _pkg_version

                current_version = _pkg_version("memento-s")
                logger.info(
                    f"[AutoUpdate] Current version (from package metadata): {current_version}"
                )
                return current_version
            except Exception as e:
                logger.error(f"[AutoUpdate] Failed to get version: {e}")
                return "0.2.0"

    def _mark_checked(self) -> None:
        """Record that a check cycle has completed (regardless of path taken)."""
        self._last_check_time = time.time()

    def _is_within_check_interval(self) -> bool:
        """Return True if the last check was recent enough to skip a new one."""
        if self._last_check_time <= 0:
            return False
        cfg = self._config.load() if hasattr(self._config, "load") else self._config
        interval_hours = (
            getattr(cfg.ota, "check_interval_hours", 24) if cfg and cfg.ota else 24
        )
        if interval_hours <= 0:
            return False
        return (time.time() - self._last_check_time) < interval_hours * 3600

    def _is_cache_stale(self) -> bool:
        """Return True if the cached update is too old and should be re-verified."""
        if not self._cache:
            return False
        try:
            cache_ts = datetime.fromisoformat(self._cache.timestamp).timestamp()
        except (ValueError, TypeError):
            return True
        cfg = self._config.load() if hasattr(self._config, "load") else self._config
        interval_hours = (
            getattr(cfg.ota, "check_interval_hours", 24) if cfg and cfg.ota else 24
        )
        stale_threshold = max(interval_hours, 24) * 3600
        return (time.time() - cache_ts) > stale_threshold

    async def start_auto_check(self):
        """Start automatic update check after startup delay.

        Startup flow (in priority order):
        1. If a fully-downloaded cached update exists → notify directly.
        2. If a partially-downloaded cache exists → resume download.
        3. Otherwise → hit the OTA server for a fresh check.
        """
        cfg = self._config.load() if hasattr(self._config, "load") else self._config
        if not cfg or not cfg.ota or not cfg.ota.url:
            logger.info("[AutoUpdate] OTA not configured")
            return

        if not getattr(cfg.ota, "auto_check", True):
            logger.info("[AutoUpdate] Auto check disabled")
            return

        logger.info(f"[AutoUpdate] Check in {self.STARTUP_CHECK_DELAY}s")
        await asyncio.sleep(self.STARTUP_CHECK_DELAY)

        if self._is_within_check_interval():
            logger.info(
                "[AutoUpdate] Auto-check skipped: within check_interval_hours cooldown"
            )
            return

        self._is_auto_check = True

        # Defensive cleanup: remove leftovers from a previous successful install
        if self._cache and self._cache.installed:
            logger.info("[AutoUpdate] Found stale installed cache, cleaning up")
            self.clear_cache()

        # Priority 1: fully cached update ready to install
        if self.has_cached_update:
            if self._is_cache_stale():
                logger.info("[AutoUpdate] Cached update is stale, re-checking")
                self.clear_cache()
            else:
                logger.info(f"[AutoUpdate] Found cached: {self._cache.version}")
                self._current_update = UpdateInfo(
                    version=self._cache.version,
                    current_version=self._get_current_version(),
                    download_url="",
                    force_update=self._cache.force_update,
                )
                self._set_status(UpdateStatus.DOWNLOADED)
                self._mark_checked()
                if self._current_update:
                    self._notify_download_complete(self._current_update)
                return

        # Priority 2: partially downloaded – try to resume
        if self.has_partial_cache:
            logger.info(
                f"[AutoUpdate] Found partial cache: {self._cache.version}, "
                f"downloaded={self._cache.downloaded_size}, total={self._cache.total_size}"
            )
            await self._resume_partial_download()
            return

        # Priority 3: fresh server check
        await self.check_for_update()

    async def _resume_partial_download(self):
        """Resume a partially downloaded update.

        Always re-checks the server first: if the server now advertises a
        different version than the cached partial, the old partial is discarded
        and a fresh download begins.
        """
        if not self._cache:
            return

        cfg = self._config.load() if hasattr(self._config, "load") else self._config
        if not cfg or not cfg.ota or not cfg.ota.url:
            return

        current_version = self._get_current_version()

        try:
            params = {
                "current_version": current_version,
                "platform": "mac"
                if platform.system().lower() == "darwin"
                else platform.system().lower(),
            }
            async with httpx.AsyncClient() as client:
                response = await client.get(cfg.ota.url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()

            self._mark_checked()

            if not data.get("update_available"):
                logger.info("[AutoUpdate] No update available, clearing partial cache")
                self.clear_cache()
                self._set_status(UpdateStatus.IDLE)
                return

            server_version = data.get("latest_version", "")
            download_url = data.get("download_url", "")

            if server_version != self._cache.version:
                logger.info(
                    f"[AutoUpdate] Server version {server_version} != cached {self._cache.version}, restarting"
                )
                self.clear_cache()
                update_info = UpdateInfo(
                    version=server_version,
                    current_version=current_version,
                    download_url=download_url,
                    release_notes=data.get("description", ""),
                    force_update=data.get("force_update", False),
                )
                self._current_update = update_info
                self._set_status(UpdateStatus.AVAILABLE)
                await self.download_update(update_info, resume=False)
                return

            update_info = UpdateInfo(
                version=self._cache.version,
                current_version=current_version,
                download_url=download_url,
                release_notes=data.get("description", ""),
                force_update=data.get("force_update", False),
            )
            self._current_update = update_info
            logger.info(
                f"[AutoUpdate] Resuming partial download for {self._cache.version}"
            )
            await self.download_update(update_info, resume=True)

        except Exception as e:
            logger.error(
                f"[AutoUpdate] Failed to resume partial download: {e}", exc_info=True
            )
            self._set_status(UpdateStatus.ERROR)
            self._notify_error(f"Resume download failed: {e}")

    async def check_for_update(self) -> UpdateInfo | None:
        """Check for available updates from OTA server."""
        # 防止重复检查：如果已在检查或下载中，直接返回当前更新信息
        if self._status == UpdateStatus.CHECKING:
            logger.info("[AutoUpdate] Already checking, skip duplicate check")
            return self._current_update
        if self._status == UpdateStatus.DOWNLOADING:
            logger.info("[AutoUpdate] Already downloading, skip check")
            return self._current_update
        # 如果已有缓存更新且未过旧，直接返回
        if self.has_cached_update:
            if self._is_cache_stale():
                logger.info(
                    "[AutoUpdate] Cached update is stale, clearing and re-checking"
                )
                self.clear_cache()
            else:
                logger.info(f"[AutoUpdate] Has cached update: {self._cache.version}")
                self._mark_checked()
                # Ensure _current_update reflects the cached force_update flag so
                # that callers (UpdateNotifier, SettingsPanel) can read is_force_update.
                if self._current_update is None and self._cache:
                    self._current_update = UpdateInfo(
                        version=self._cache.version,
                        current_version=self._get_current_version(),
                        download_url="",
                        force_update=self._cache.force_update,
                    )
                return self._current_update

        self._set_status(UpdateStatus.CHECKING)

        try:
            cfg = self._config.load() if hasattr(self._config, "load") else self._config
            if not cfg or not cfg.ota or not cfg.ota.url:
                logger.info("[AutoUpdate] OTA not configured")
                self._set_status(UpdateStatus.IDLE)
                return None

            current_version = self._get_current_version()
            platformVal = platform.system().lower()
            params = {
                "current_version": current_version,
                "platform": "mac" if platformVal == "darwin" else platformVal,
                "arch": platform.machine().lower() if platformVal == "darwin" else "",
            }

            logger.info(f"[AutoUpdate] Checking: {cfg.ota.url}, params: {params}")

            async with httpx.AsyncClient() as client:
                response = await client.get(cfg.ota.url, params=params, timeout=10.0)
                response.raise_for_status()
                resp = response.json()
                logger.info(f"[AutoUpdate] Checking: {resp}")

            self._mark_checked()

            code = resp.get("code")
            if code != 200:
                logger.error(f"[AutoUpdate] OTA server error: {code}")
                self._set_status(UpdateStatus.ERROR)
                return None

            data = resp.get("data")
            if not data.get("update_available"):
                logger.info("[AutoUpdate] No updates")
                self._set_status(UpdateStatus.IDLE)
                return None

            update_info = UpdateInfo(
                version=data.get("latest_version", ""),
                current_version=current_version,
                download_url=data.get("download_url", ""),
                release_notes=data.get("description", ""),
                force_update=data.get("force_update", False),
                published_at=data.get("published_at", ""),
                size=data.get("size", 0),
                checksum=data.get("checksum"),
            )

            if not update_info.version or not update_info.download_url:
                logger.error("[AutoUpdate] Invalid update info")
                self._set_status(UpdateStatus.ERROR)
                return None

            try:
                current = parse_version(current_version)
                latest = parse_version(update_info.version)
                if latest <= current:
                    logger.info(f"[AutoUpdate] No newer version")
                    self._set_status(UpdateStatus.IDLE)
                    return None
            except Exception as e:
                logger.warning(f"[AutoUpdate] Version compare: {e}")

            logger.info(f"[AutoUpdate] Available: {update_info.version}")
            self._current_update = update_info
            self._set_status(UpdateStatus.AVAILABLE)

            blocked_reason = self.get_macos_update_block_reason()
            if blocked_reason:
                logger.warning(
                    f"[AutoUpdate] Update available but download/install is blocked: {blocked_reason}"
                )
                return update_info

            auto_download = getattr(cfg.ota, "auto_download", True)
            if update_info.force_update:
                logger.info(
                    "[AutoUpdate] force_update=True, downloading regardless of auto_download"
                )
                await self.download_update(update_info)
            elif auto_download:
                await self.download_update(update_info)

            return update_info

        except httpx.RequestError as e:
            logger.error(f"[AutoUpdate] Network error: {e}")
            self._set_status(UpdateStatus.ERROR)
            self._notify_error(f"Network error: {e}")
        except Exception as e:
            logger.error(f"[AutoUpdate] Check failed: {e}", exc_info=True)
            self._set_status(UpdateStatus.ERROR)
            self._notify_error(f"Check failed: {e}")

        return None

    async def download_update(
        self,
        update_info: UpdateInfo | None = None,
        resume: bool = True,
    ) -> bool:
        """Download the update package.

        Wraps the actual download in an asyncio.Task so that cancel_download()
        can interrupt network I/O immediately via Task.cancel().
        """
        if update_info:
            self._current_update = update_info
        elif not self._current_update:
            self._notify_error("No update info")
            return False

        update_info = self._current_update

        blocked_reason = self.get_macos_update_block_reason()
        if blocked_reason:
            self._notify_error(blocked_reason)
            return False

        if self.has_cached_update and self._cache.version == update_info.version:
            logger.info(f"[AutoUpdate] Already downloaded: {update_info.version}")
            self._set_status(UpdateStatus.DOWNLOADED)
            self._notify_download_complete(update_info)
            return True

        if self._status == UpdateStatus.DOWNLOADING:
            logger.info("[AutoUpdate] Already downloading, skip duplicate download")
            return True

        self._cancelled = False
        self._paused = False

        # Store the current task reference so cancel_download() can call
        # Task.cancel() to immediately interrupt blocking network I/O,
        # rather than relying solely on the _cancelled flag polled between chunks.
        self._download_task = asyncio.current_task()
        try:
            return await self._do_download(update_info, resume)
        except asyncio.CancelledError:
            logger.info("[AutoUpdate] Download task cancelled")
            self._set_status(UpdateStatus.CANCELLED)
            return False
        finally:
            self._download_task = None

    async def _do_download(self, update_info: UpdateInfo, resume: bool) -> bool:
        """Internal download implementation."""
        self._set_status(UpdateStatus.DOWNLOADING)

        url = update_info.download_url
        file_ext = Path(url).suffix or self._get_platform_extension()
        download_path = self.CACHE_DIR / f"update_{update_info.version}{file_ext}"
        logger.info(f"[AutoUpdate] Download path: {download_path}")

        self._progress = DownloadProgress(total_size=update_info.size)

        resume_byte_pos = 0
        if resume and download_path.exists():
            resume_byte_pos = download_path.stat().st_size
            if update_info.size > 0 and resume_byte_pos >= update_info.size:
                logger.info("[AutoUpdate] Already complete")
                self._progress.downloaded = resume_byte_pos
                self._finish_download(download_path, update_info)
                return True
            logger.info(f"[AutoUpdate] Resume from {resume_byte_pos}")
            self._progress.downloaded = resume_byte_pos

        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "User-Agent": f"Memento-S/{self._get_current_version()}",
                }
                if resume_byte_pos > 0:
                    headers["Range"] = f"bytes={resume_byte_pos}-"

                async with client.stream(
                    "GET",
                    url,
                    headers=headers,
                    timeout=300.0,
                    follow_redirects=True,
                ) as response:
                    response.raise_for_status()

                    # 206 Partial Content = server honoured the Range header;
                    # 200 OK = server ignored it and sent the full file.
                    server_supports_resume = response.status_code == 206

                    if "Content-Length" in response.headers:
                        content_length = int(response.headers["Content-Length"])
                        if resume_byte_pos > 0 and server_supports_resume:
                            # Content-Length is the remaining bytes, not total
                            self._progress.total_size = resume_byte_pos + content_length
                        else:
                            self._progress.total_size = content_length
                    elif update_info.size > 0:
                        self._progress.total_size = update_info.size

                    if resume_byte_pos > 0 and not server_supports_resume:
                        logger.warning(
                            "[AutoUpdate] Server does not support resume, restarting download"
                        )
                        resume_byte_pos = 0
                        self._progress.downloaded = 0

                    mode = (
                        "ab" if resume_byte_pos > 0 and server_supports_resume else "wb"
                    )
                    with open(download_path, mode) as f:
                        last_progress_time = time.time()
                        bytes_since_last = 0
                        # Avoids emitting PAUSED status every 0.5s while paused;
                        # status is set once on entering paused state.
                        _paused_notified = False

                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            if self._cancelled:
                                logger.info("[AutoUpdate] Cancelled")
                                self._set_status(UpdateStatus.CANCELLED)
                                return False

                            while self._paused:
                                if not _paused_notified:
                                    self._set_status(UpdateStatus.PAUSED)
                                    _paused_notified = True
                                await asyncio.sleep(0.5)
                                if self._cancelled:
                                    return False

                            if _paused_notified:
                                _paused_notified = False

                            if self._status != UpdateStatus.DOWNLOADING:
                                self._set_status(UpdateStatus.DOWNLOADING)

                            f.write(chunk)
                            self._progress.downloaded += len(chunk)
                            bytes_since_last += len(chunk)

                            current_time = time.time()
                            if current_time - last_progress_time >= 0.5:
                                time_diff = current_time - last_progress_time
                                if time_diff > 0:
                                    self._progress.speed = bytes_since_last / time_diff
                                self._progress.last_update_time = current_time
                                self._notify_progress()
                                last_progress_time = current_time
                                bytes_since_last = 0

                    if update_info.checksum:
                        if not self._verify_checksum(
                            download_path, update_info.checksum
                        ):
                            logger.error("[AutoUpdate] Checksum failed")
                            download_path.unlink(missing_ok=True)
                            self._set_status(UpdateStatus.ERROR)
                            self._notify_error("Verification failed")
                            return False

                    if not self._is_valid_archive(download_path):
                        logger.error(
                            "[AutoUpdate] Downloaded file is not a valid archive"
                        )
                        download_path.unlink(missing_ok=True)
                        self._set_status(UpdateStatus.ERROR)
                        self._notify_error("Downloaded file is corrupted, please retry")
                        return False

                    logger.info(f"[AutoUpdate] Downloaded: {download_path}")
                    self._finish_download(download_path, update_info)
                    return True

        except asyncio.CancelledError:
            # Propagate to the outer handler in download_update()
            raise
        except Exception as e:
            logger.error(f"[AutoUpdate] Download failed: {e}", exc_info=True)
            self._set_status(UpdateStatus.ERROR)
            self._notify_error(f"Download failed: {e}")
            return False

    def _finish_download(self, download_path: Path, update_info: UpdateInfo):
        """Complete download and save cache."""
        self._cache = UpdateCache(
            version=update_info.version,
            download_path=download_path,
            checksum=update_info.checksum,
            downloaded_size=download_path.stat().st_size,
            total_size=self._progress.total_size,
            timestamp=datetime.now().isoformat(),
            installed=False,
            force_update=update_info.force_update,
        )
        self._save_cache()
        self._set_status(UpdateStatus.DOWNLOADED)

        self._notify_download_complete(update_info)

    @staticmethod
    def _is_valid_archive(file_path: Path) -> bool:
        """Quick check that a downloaded archive is structurally complete."""
        try:
            suffix = file_path.suffix.lower()
            if suffix == ".zip":
                with zipfile.ZipFile(file_path, "r") as zf:
                    zf.testzip()
                return True
            if suffix in (".gz", ".tgz") or file_path.name.endswith(".tar.gz"):
                with tarfile.open(file_path, "r:gz") as tf:
                    tf.getmembers()
                return True
            return True
        except Exception as e:
            logger.warning(f"[AutoUpdate] Archive validation failed: {e}")
            return False

    @staticmethod
    def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
        """Extract a zip archive with path traversal protection.

        Python's ``zipfile.extractall()`` does NOT restore Unix permission
        bits stored in the zip's ``external_attr`` field.  This causes
        executables (e.g. the main binary inside a .app bundle) to lose
        their +x flag.  After extracting, we iterate over all entries and
        restore the original permission mode when available.
        """
        dest_dir_resolved = dest_dir.resolve()
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                target = (dest_dir / member.filename).resolve()
                if not str(target).startswith(str(dest_dir_resolved)):
                    raise RuntimeError(
                        f"Zip entry '{member.filename}' would escape destination directory"
                    )
            zf.extractall(dest_dir)

            # Restore Unix file permissions that zipfile.extractall() ignores.
            # The Unix mode is stored in the upper 16 bits of external_attr.
            for member in zf.infolist():
                if member.is_dir():
                    continue
                unix_mode = member.external_attr >> 16
                if unix_mode != 0:
                    extracted = dest_dir / member.filename
                    try:
                        extracted.chmod(unix_mode)
                    except OSError:
                        pass

    @staticmethod
    def _safe_extract_tar(tar_path: Path, dest_dir: Path) -> None:
        """Extract a tar archive with path traversal protection.

        Uses the 'data' filter on Python >= 3.12 for built-in safety,
        falls back to manual validation on older versions.
        """
        dest_dir_resolved = dest_dir.resolve()
        with tarfile.open(tar_path, "r:gz") as tf:
            if sys.version_info >= (3, 12):
                tf.extractall(dest_dir, filter="data")
            else:
                for member in tf.getmembers():
                    target = (dest_dir / member.name).resolve()
                    if not str(target).startswith(str(dest_dir_resolved)):
                        raise RuntimeError(
                            f"Tar entry '{member.name}' would escape destination directory"
                        )
                    if member.islnk() or member.issym():
                        link_target = (dest_dir / member.linkname).resolve()
                        if not str(link_target).startswith(str(dest_dir_resolved)):
                            raise RuntimeError(
                                f"Tar symlink '{member.name}' -> '{member.linkname}' escapes destination"
                            )
                tf.extractall(dest_dir)

    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum.

        Auto-detects hash algorithm by the length of the expected hex string:
        32 chars → MD5, 40 → SHA-1, 64 → SHA-256.  Unknown lengths are
        treated as valid (pass-through) with a warning.
        """
        try:
            if len(expected_checksum) == 32:
                hash_obj = hashlib.md5()
            elif len(expected_checksum) == 40:
                hash_obj = hashlib.sha1()
            elif len(expected_checksum) == 64:
                hash_obj = hashlib.sha256()
            else:
                logger.warning(f"[AutoUpdate] Unknown checksum format")
                return True

            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)

            return hash_obj.hexdigest().lower() == expected_checksum.lower()
        except Exception as e:
            logger.error(f"[AutoUpdate] Checksum error: {e}")
            return False

    def _get_platform_extension(self) -> str:
        """Get file extension for current platform."""
        system = platform.system().lower()
        if system == "darwin":
            return ".zip"
        elif system == "windows":
            return ".zip"
        else:
            return ".tar.gz"

    @staticmethod
    def _find_graphical_privilege_helper() -> str:
        """Return the best available graphical privilege escalation command.

        Prefers ``pkexec`` (Polkit, works with most desktop environments),
        falls back to ``pkexec`` even if not found (will produce a clear
        error at runtime).  Avoids bare ``sudo`` which silently fails
        without a terminal.
        """
        for candidate in ("pkexec", "kdesudo", "gksudo"):
            if shutil.which(candidate):
                return candidate
        logger.warning(
            "[AutoUpdate] No graphical privilege helper found, falling back to pkexec"
        )
        return "pkexec"

    def pause_download(self):
        """Pause ongoing download."""
        if self._status == UpdateStatus.DOWNLOADING:
            self._paused = True
            self._set_status(UpdateStatus.PAUSED)
            logger.info("[AutoUpdate] Paused")

    def resume_download(self):
        """Resume paused download."""
        if self._status == UpdateStatus.PAUSED:
            self._paused = False
            self._set_status(UpdateStatus.DOWNLOADING)
            logger.info("[AutoUpdate] Resumed")

    def cancel_download(self):
        """Cancel ongoing download.

        Sets the cancelled flag (checked between chunks) and also cancels the
        underlying asyncio Task to interrupt any blocking network I/O.
        """
        self._cancelled = True
        self._paused = False
        if self._download_task and not self._download_task.done():
            self._download_task.cancel()
        if self._status not in (UpdateStatus.CANCELLED, UpdateStatus.IDLE):
            self._set_status(UpdateStatus.CANCELLED)
        logger.info("[AutoUpdate] Cancelled")

    def get_ota_config_value(self, key: str, default: Any = None) -> Any:
        """Read a single OTA config field, returning *default* if absent."""
        try:
            cfg = self._config.load() if hasattr(self._config, "load") else self._config
            return getattr(cfg.ota, key, default) if cfg and cfg.ota else default
        except Exception:
            return default

    @property
    def is_force_update(self) -> bool:
        """Whether the current pending update is marked as forced by the server."""
        return bool(self._current_update and self._current_update.force_update)

    async def manual_check_for_update(self) -> UpdateInfo | None:
        """Manually triggered update check (non-silent)."""
        self._is_auto_check = False
        logger.info("[AutoUpdate] Manual check triggered")
        return await self.check_for_update()

    async def install_update(
        self,
        page: Any | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> bool:
        """Install the downloaded update.

        Notes:
        - 各平台的安装都是通过异步脚本/安装程序完成的；
          这里返回 True 只表示“安装脚本已成功启动”，不代表安装已完成。
        - 不在这里 clear_cache / COMPLETED，因为当前进程即将退出，
          真正清理由安装脚本在成功后完成。
        """
        if not self.has_cached_update:
            self._notify_error("No update to install")
            return False

        self._set_status(UpdateStatus.INSTALLING)

        try:
            download_path = self._cache.download_path
            version = self._cache.version

            logger.info(f"[AutoUpdate] Installing: {version}")

            system = platform.system().lower()

            if system == "darwin":
                success = await self._install_macos(download_path, version)
            elif system == "windows":
                success = await self._install_windows(download_path, version)
            elif system == "linux":
                success = await self._install_linux(download_path, version)
            else:
                raise RuntimeError(f"Unsupported: {system}")

            if success:
                logger.info(
                    f"[AutoUpdate] Installer script launched successfully for {version}; "
                    "final replacement/cleanup will be handled by the script"
                )
                if on_complete:
                    on_complete()
                return True
            else:
                self._set_status(UpdateStatus.ERROR)
                return False

        except Exception as e:
            logger.error(f"[AutoUpdate] Install failed: {e}", exc_info=True)
            is_archive_error = isinstance(e, (zipfile.BadZipFile, RuntimeError)) and (
                "zip" in str(e).lower()
                or "extract" in str(e).lower()
                or "corrupt" in str(e).lower()
            )
            if is_archive_error:
                self.clear_cache()
                logger.info("[AutoUpdate] Cleared corrupted cache after archive error")
            self._set_status(UpdateStatus.ERROR)
            self._notify_error(f"Install failed: {e}")
            return False

    async def _install_macos(self, download_path: Path, version: str) -> bool:
        """Install update on macOS by replacing the current .app bundle safely.

        Flow:
        1. Extract archive to temp dir
        2. Find the best .app bundle
        3. Validate bundle structure strictly
        4. Generate shell script for delayed self-replacement
        5. Start script and return True (script performs final cleanup/relaunch)
        """
        logger.info(f"[AutoUpdate] macOS install: {version}")

        extract_dir = self.TEMP_DIR / f"extract_{version}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            if download_path.suffix == ".zip":
                logger.info("[AutoUpdate] Extracting ZIP")
                self._safe_extract_zip(download_path, extract_dir)
            elif str(download_path).endswith(".tar.gz") or str(download_path).endswith(
                ".tgz"
            ):
                logger.info("[AutoUpdate] Extracting TAR.GZ")
                self._safe_extract_tar(download_path, extract_dir)
            elif download_path.suffix == ".dmg":
                logger.info("[AutoUpdate] Opening DMG")
                subprocess.Popen(["open", str(download_path)])
                return True
            else:
                logger.warning(
                    f"[AutoUpdate] Unknown type: {download_path.suffix}, trying ZIP extraction"
                )
                try:
                    self._safe_extract_zip(download_path, extract_dir)
                except zipfile.BadZipFile:
                    raise RuntimeError(
                        f"Cannot extract update package: {download_path}"
                    )

            macosx_dir = extract_dir / "__MACOSX"
            if macosx_dir.exists():
                shutil.rmtree(macosx_dir, ignore_errors=True)

            new_app = self._find_macos_app_bundle(extract_dir)
            if not new_app:
                raise RuntimeError("No valid .app bundle found in extracted package")
            logger.info(f"[AutoUpdate] New app: {new_app}")

            exe_path = Path(sys.executable).resolve()
            current_app = None
            for parent in exe_path.parents:
                if parent.suffix == ".app" and parent.is_dir():
                    current_app = parent
                    break
            if not current_app:
                raise RuntimeError(
                    "Current app not found (running outside .app bundle?)"
                )
            logger.info(f"[AutoUpdate] Current app: {current_app}")

            if self._is_macos_app_translocated(current_app):
                raise RuntimeError(
                    "macOS 已将当前应用放在 App Translocation 隔离路径中运行，"
                    "无法执行原地更新。请先将 Memento-S.app 移动到 /Applications "
                    "或 ~/Applications，并从新位置重新打开一次后再更新。"
                )

            backup_app = self.TEMP_DIR / f"{current_app.stem}.backup_bundle"
            script_path = self.TEMP_DIR / f"update_{version}.sh"

            app_pid = os.getpid()
            log_path = self.TEMP_DIR / f"update_{version}.log"

            _q = shlex.quote
            script_lines = [
                "#!/bin/bash",
                "",
                "# Clear PyInstaller env vars inherited from the frozen parent process",
                "# BEFORE set -u, since these vars may not exist.",
                "unset _MEIPASS2 _PYI_SPLASH_IPC 2>/dev/null || true",
                "",
                "set -u",
                "",
                f"CURRENT_APP={_q(str(current_app))}",
                f"BACKUP_APP={_q(str(backup_app))}",
                f"NEW_APP={_q(str(new_app))}",
                f"EXTRACT_DIR={_q(str(extract_dir))}",
                f"DOWNLOAD_PKG={_q(str(download_path))}",
                f"CACHE_META={_q(str(self.CACHE_METADATA_FILE))}",
                f"SCRIPT_PATH={_q(str(script_path))}",
                f"LOG_PATH={_q(str(log_path))}",
                f"APP_PID={app_pid}",
                "",
                'log() { echo "$(date "+%Y-%m-%d %H:%M:%S") [AutoUpdate] $*" >> "$LOG_PATH"; }',
                "",
                "cleanup_temp() {",
                '    rm -rf "$EXTRACT_DIR" 2>/dev/null || true',
                '    rm -f "$DOWNLOAD_PKG" 2>/dev/null || true',
                '    rm -f "$CACHE_META" 2>/dev/null || true',
                '    rm -f "$SCRIPT_PATH" 2>/dev/null || true',
                "}",
                "",
                "fail() {",
                '    log "$1"',
                "    rollback",
                "}",
                "",
                "rollback() {",
                '    log "Update failed, starting rollback"',
                '    if [ -d "$BACKUP_APP" ]; then',
                '        rm -rf "$CURRENT_APP" 2>/dev/null || true',
                '        mv "$BACKUP_APP" "$CURRENT_APP" || { log "Rollback move failed: $BACKUP_APP -> $CURRENT_APP"; exit 1; }',
                '        log "Rollback complete"',
                '        env -i HOME="$HOME" PATH="$PATH" USER="$USER" SHELL="$SHELL" TMPDIR="${TMPDIR:-/tmp}" open -n "$CURRENT_APP" >/dev/null 2>&1 || true',
                "    fi",
                "    cleanup_temp",
                "    exit 1",
                "}",
                "",
                'log "Update script started"',
                'log "CURRENT_APP=$CURRENT_APP"',
                'log "NEW_APP=$NEW_APP"',
                'log "APP_PID=$APP_PID"',
                "",
                "ELAPSED=0",
                'while kill -0 "$APP_PID" 2>/dev/null; do',
                "    sleep 1",
                "    ELAPSED=$((ELAPSED + 1))",
                '    if [ "$ELAPSED" -ge 30 ]; then',
                '        log "Timeout waiting for app process to exit, force killing app pid"',
                '        kill -9 "$APP_PID" 2>/dev/null || true',
                "        sleep 2",
                "        break",
                "    fi",
                "done",
                'log "All processes exited (waited ${ELAPSED}s)"',
                "",
                "sleep 2",
                "",
                'TMPBASE=$(getconf DARWIN_USER_TEMP_DIR 2>/dev/null || echo "${TMPDIR:-/tmp}")',
                'if [ -d "$TMPBASE" ]; then',
                '    for d in "$TMPBASE"/_MEI*; do',
                '        [ -d "$d" ] || continue',
                '        rm -rf "$d" 2>/dev/null || true',
                '        log "Removed stale PyInstaller temp dir: $d"',
                "    done",
                "fi",
                "",
                'if [ ! -d "$NEW_APP" ] || [ ! -d "$NEW_APP/Contents/MacOS" ]; then',
                '    fail "New app bundle invalid"',
                "fi",
                "",
                'xattr -rd com.apple.quarantine "$NEW_APP" 2>/dev/null || true',
                'chmod +x "$NEW_APP/Contents/MacOS/"* 2>/dev/null || true',
                "",
                'rm -rf "$BACKUP_APP" 2>/dev/null || true',
                "",
                'if [ ! -w "$(dirname "$CURRENT_APP")" ]; then',
                '    fail "Current app parent directory is not writable: $(dirname "$CURRENT_APP")"',
                "fi",
                "",
                'if [ -d "$CURRENT_APP" ]; then',
                '    mv "$CURRENT_APP" "$BACKUP_APP" || fail "Failed to move current app to backup: $CURRENT_APP -> $BACKUP_APP"',
                '    log "Moved current app to backup"',
                "fi",
                "",
                'mv "$NEW_APP" "$CURRENT_APP" || fail "Failed to move new app into place: $NEW_APP -> $CURRENT_APP"',
                'log "Moved new app into place"',
                "",
                'xattr -rd com.apple.quarantine "$CURRENT_APP" 2>/dev/null || true',
                'chmod +x "$CURRENT_APP/Contents/MacOS/"* 2>/dev/null || true',
                "",
                'EXE_NAME=$(/usr/libexec/PlistBuddy -c "Print :CFBundleExecutable" "$CURRENT_APP/Contents/Info.plist" 2>/dev/null)',
                'EXE_PATH="$CURRENT_APP/Contents/MacOS/${EXE_NAME}"',
                'if [ ! -x "$EXE_PATH" ]; then',
                '    fail "Executable not found: $EXE_PATH"',
                "fi",
                'log "Executable: $EXE_PATH"',
                "",
                'log "Launching new app bundle via LaunchServices..."',
                "unset _MEIPASS2 2>/dev/null || true",
                'if ! env -i HOME="$HOME" PATH="$PATH" USER="$USER" SHELL="$SHELL" TMPDIR="${TMPDIR:-/tmp}" open -n "$CURRENT_APP" >/dev/null 2>&1; then',
                '    fail "Failed to launch app bundle with open -n"',
                "fi",
                'log "LaunchServices accepted app relaunch request"',
                "",
                "APP_STARTED=false",
                "for wait_sec in $(seq 1 30); do",
                "    sleep 1",
                '    if pgrep -f "$CURRENT_APP/Contents/MacOS/$EXE_NAME" >/dev/null 2>&1; then',
                '        log "App process found after ${wait_sec}s"',
                "        APP_STARTED=true",
                "        break",
                "    fi",
                "done",
                "",
                'if [ "$APP_STARTED" = false ]; then',
                '    fail "Timed out waiting for relaunched app process"',
                "fi",
                "",
                'log "New app launched successfully"',
                'rm -rf "$BACKUP_APP" 2>/dev/null || true',
                "cleanup_temp",
                'log "Update finished successfully"',
                "",
                "exit 0",
            ]
            script_path.write_text("\n".join(script_lines))
            script_path.chmod(0o755)

            logger.info(f"[AutoUpdate] Launching macOS update script: {script_path}")

            subprocess.Popen(
                ["bash", str(script_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            return True

        except Exception as e:
            logger.error(f"[AutoUpdate] macOS error: {e}", exc_info=True)
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
            raise

    def _find_macos_app_bundle(self, extract_dir: Path) -> Path | None:
        """Find a valid .app bundle from the extracted package."""
        app_bundles = [
            p
            for p in extract_dir.rglob("*.app")
            if "__MACOSX" not in p.parts and self._validate_macos_app_bundle(p)
        ]
        return app_bundles[0] if app_bundles else None

    @classmethod
    def _is_macos_app_translocated(cls, app_path: Path) -> bool:
        """Whether the running app bundle is under macOS App Translocation."""
        return cls.MACOS_APP_TRANSLOCATION_MARKER in str(app_path)

    @staticmethod
    def _read_macos_quarantine(app_path: Path | None) -> str:
        """Read the quarantine xattr value for the current macOS app bundle."""
        if not app_path:
            return ""
        try:
            result = subprocess.run(
                ["xattr", "-p", "com.apple.quarantine", str(app_path)],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return ""
        except Exception:
            return ""

    @staticmethod
    def _validate_macos_app_bundle(app_path: Path) -> bool:
        """Validate basic .app bundle structure (Contents/MacOS must exist)."""
        if not app_path.is_dir():
            return False
        macos_dir = app_path / "Contents" / "MacOS"
        if not macos_dir.is_dir():
            return False
        for exe_file in macos_dir.iterdir():
            if exe_file.is_file():
                exe_file.chmod(exe_file.stat().st_mode | 0o755)
        return True

    async def _install_windows(self, download_path: Path, version: str) -> bool:
        """Install update on Windows.

        For .exe/.msi installers, launches them directly.
        For .zip packages, replaces the *entire* application directory
        (not just a single exe) with rollback support.

        The generated batch script performs:
          1. Wait for the app process to exit.
          2. Move the whole app directory to a .backup sibling.
          3. Move extracted new directory into the original location.
          4. Validate the main executable exists in the new directory.
          5. On any failure: restore the backup directory.
          6. On success: remove backup, relaunch, clean up.
        """
        logger.info(f"[AutoUpdate] Windows install: {version}")

        extract_dir = self.TEMP_DIR / f"extract_{version}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            if download_path.suffix == ".zip":
                logger.info("[AutoUpdate] Extracting ZIP")
                self._safe_extract_zip(download_path, extract_dir)
            elif download_path.suffix == ".exe":
                # The downloaded file is the new application executable directly
                # (not a wrapper installer). Generate a batch script that waits
                # for the current process to exit, replaces the exe, then
                # relaunches — mirroring the .zip directory-replace approach.
                logger.info("[AutoUpdate] Replacing exe via batch script")

                # Move the downloaded exe into TEMP_DIR before generating the
                # batch script.  install_update() calls clear_cache() right
                # after this method returns, which would delete download_path
                # (in CACHE_DIR) before the batch script has a chance to use
                # it.  By moving it to TEMP_DIR first, clear_cache() finds
                # nothing to delete and the batch script can still find the
                # file 5 seconds later.
                staged_exe = self.TEMP_DIR / f"new_{version}.exe"
                shutil.move(str(download_path), str(staged_exe))
                logger.info(f"[AutoUpdate] Staged new exe: {staged_exe}")

                current_exe = Path(sys.executable).resolve()
                backup_exe = current_exe.with_suffix(".exe.backup")
                script_path = self.TEMP_DIR / f"update_{version}.bat"
                script_lines = [
                    "@echo off",
                    "setlocal",
                    "",
                    f'set "CURRENT_EXE={current_exe}"',
                    f'set "BACKUP_EXE={backup_exe}"',
                    f'set "NEW_EXE={staged_exe}"',
                    f'set "CACHE_META={self.CACHE_METADATA_FILE}"',
                    f'set "SCRIPT_PATH={script_path}"',
                    "",
                    "goto :main",
                    "",
                    ":delete_file_retry",
                    'if not exist "%~1" exit /b 0',
                    "for /l %%i in (1,1,5) do (",
                    '    del /f /q "%~1" 2>nul',
                    '    if not exist "%~1" exit /b 0',
                    "    timeout /t 1 /nobreak >nul",
                    ")",
                    "exit /b 1",
                    "",
                    ":main",
                    "timeout /t 5 /nobreak >nul",
                    "",
                    'if exist "%BACKUP_EXE%" call :delete_file_retry "%BACKUP_EXE%"',
                    "",
                    'move "%CURRENT_EXE%" "%BACKUP_EXE%"',
                    "if errorlevel 1 (",
                    "    echo [AutoUpdate] Failed to backup old exe",
                    "    goto :launch",
                    ")",
                    "",
                    'move "%NEW_EXE%" "%CURRENT_EXE%"',
                    "if errorlevel 1 (",
                    "    echo [AutoUpdate] Failed to move new exe, rolling back...",
                    '    move "%BACKUP_EXE%" "%CURRENT_EXE%"',
                    "    goto :launch",
                    ")",
                    "",
                    'if not exist "%CURRENT_EXE%" (',
                    "    echo [AutoUpdate] New exe missing, rolling back...",
                    '    move "%BACKUP_EXE%" "%CURRENT_EXE%"',
                    "    goto :launch",
                    ")",
                    "",
                    'if exist "%BACKUP_EXE%" call :delete_file_retry "%BACKUP_EXE%"',
                    "",
                    ":launch",
                    'set "_MEIPASS2="',
                    'set "PYINSTALLER_RESET_ENVIRONMENT=1"',
                    'set "_PYI_ARCHIVE_FILE="',
                    'set "_PYI_APPLICATION_HOME_DIR="',
                    'set "_PYI_PARENT_PROCESS_LEVEL="',
                    'set "_PYI_SPLASH_IPC="',
                    'set "PYTHONHOME="',
                    'set "PYTHONPATH="',
                    'set "SSL_CERT_FILE="',
                    'set "SSL_CERT_DIR="',
                    'set "REQUESTS_CA_BUNDLE="',
                    'set "CURL_CA_BUNDLE="',
                    'if exist "%CURRENT_EXE%" (',
                    '    start "" "%CURRENT_EXE%"',
                    ') else if exist "%BACKUP_EXE%" (',
                    '    start "" "%BACKUP_EXE%"',
                    ")",
                    "",
                    ":cleanup",
                    'if exist "%NEW_EXE%" call :delete_file_retry "%NEW_EXE%"',
                    'if exist "%CACHE_META%" call :delete_file_retry "%CACHE_META%"',
                    'del "%SCRIPT_PATH%" 2>nul',
                ]
                script_path.write_text("\n".join(script_lines))

                DETACHED_PROCESS = 0x00000008
                CREATE_NO_WINDOW = 0x08000000
                subprocess.Popen(
                    ["cmd", "/c", str(script_path)],
                    shell=False,
                    creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
                    close_fds=True,
                )
                return True
            elif download_path.suffix == ".msi":
                logger.info("[AutoUpdate] Running MSI installer")
                CREATE_NO_WINDOW = 0x08000000
                subprocess.Popen(
                    ["msiexec", "/i", str(download_path), "/quiet"],
                    shell=False,
                    creationflags=CREATE_NO_WINDOW,
                )
                return True
            else:
                raise RuntimeError(f"Unsupported format: {download_path.suffix}")

            current_exe = Path(sys.executable).resolve()
            app_dir = current_exe.parent
            backup_dir = app_dir.parent / f"{app_dir.name}.backup"
            logger.info(f"[AutoUpdate] App directory: {app_dir}")

            # Use the first top-level directory inside the zip as the new app
            # source; fall back to extract_dir itself if the zip is flat.
            new_app_dir = None
            for candidate in extract_dir.iterdir():
                if candidate.is_dir():
                    new_app_dir = candidate
                    break
            if new_app_dir is None:
                new_app_dir = extract_dir

            script_path = self.TEMP_DIR / f"update_{version}.bat"
            script_lines = [
                "@echo off",
                "setlocal",
                "",
                f'set "APP_DIR={app_dir}"',
                f'set "BACKUP_DIR={backup_dir}"',
                f'set "NEW_APP_DIR={new_app_dir}"',
                f'set "EXTRACT_DIR={extract_dir}"',
                f'set "DOWNLOAD_PKG={download_path}"',
                f'set "CACHE_META={self.CACHE_METADATA_FILE}"',
                f'set "SCRIPT_PATH={script_path}"',
                f'set "EXE_NAME={current_exe.name}"',
                "",
                "goto :main",
                "",
                ":delete_file_retry",
                'if not exist "%~1" exit /b 0',
                "for /l %%i in (1,1,5) do (",
                '    del /f /q "%~1" 2>nul',
                '    if not exist "%~1" exit /b 0',
                "    timeout /t 1 /nobreak >nul",
                ")",
                "exit /b 1",
                "",
                ":delete_dir_retry",
                'if not exist "%~1" exit /b 0',
                "for /l %%i in (1,1,5) do (",
                '    rmdir /s /q "%~1" 2>nul',
                '    if not exist "%~1" exit /b 0',
                "    timeout /t 1 /nobreak >nul",
                ")",
                "exit /b 1",
                "",
                ":main",
                "timeout /t 3 /nobreak >nul",
                "",
                'if exist "%BACKUP_DIR%" call :delete_dir_retry "%BACKUP_DIR%"',
                "",
                'move "%APP_DIR%" "%BACKUP_DIR%"',
                "if errorlevel 1 (",
                "    echo [AutoUpdate] Failed to move app dir to backup",
                "    goto :launch",
                ")",
                "",
                'move "%NEW_APP_DIR%" "%APP_DIR%"',
                "if errorlevel 1 (",
                "    echo [AutoUpdate] Failed to move new app, rolling back...",
                '    move "%BACKUP_DIR%" "%APP_DIR%"',
                "    goto :launch",
                ")",
                "",
                'if not exist "%APP_DIR%\\%EXE_NAME%" (',
                "    echo [AutoUpdate] New app invalid, rolling back...",
                '    call :delete_dir_retry "%APP_DIR%"',
                '    move "%BACKUP_DIR%" "%APP_DIR%"',
                "    goto :launch",
                ")",
                "",
                'if exist "%BACKUP_DIR%" call :delete_dir_retry "%BACKUP_DIR%"',
                "",
                ":launch",
                'set "_MEIPASS2="',
                'set "PYINSTALLER_RESET_ENVIRONMENT=1"',
                'set "_PYI_ARCHIVE_FILE="',
                'set "_PYI_APPLICATION_HOME_DIR="',
                'set "_PYI_PARENT_PROCESS_LEVEL="',
                'set "_PYI_SPLASH_IPC="',
                'set "PYTHONHOME="',
                'set "PYTHONPATH="',
                'set "SSL_CERT_FILE="',
                'set "SSL_CERT_DIR="',
                'set "REQUESTS_CA_BUNDLE="',
                'set "CURL_CA_BUNDLE="',
                'if exist "%APP_DIR%\\%EXE_NAME%" (',
                f'    start "" "%APP_DIR%\\%EXE_NAME%"',
                ') else if exist "%BACKUP_DIR%\\%EXE_NAME%" (',
                f'    start "" "%BACKUP_DIR%\\%EXE_NAME%"',
                ")",
                "",
                ":cleanup",
                'if exist "%EXTRACT_DIR%" call :delete_dir_retry "%EXTRACT_DIR%"',
                'if exist "%DOWNLOAD_PKG%" call :delete_file_retry "%DOWNLOAD_PKG%"',
                'if exist "%CACHE_META%" call :delete_file_retry "%CACHE_META%"',
                'del "%SCRIPT_PATH%" 2>nul',
            ]
            script_path.write_text("\n".join(script_lines))

            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen(
                ["cmd", "/c", str(script_path)],
                shell=False,
                creationflags=CREATE_NO_WINDOW,
            )

            return True

        except Exception as e:
            logger.error(f"[AutoUpdate] Windows error: {e}")
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
            raise

    async def _install_linux(self, download_path: Path, version: str) -> bool:
        """Install update on Linux.

        Supports multiple package formats:
          - .zip / .tar.gz / .appimage: replaces the running binary via a
            shell script with delayed self-replacement and rollback.
          - .deb / .rpm: delegates to the system package manager via
            graphical privilege escalation (pkexec).

        Flow (for .zip / .tar.gz shell scripts):
        1. Poll-wait for the app process to exit (up to 30 s)
        2. Backup current binary, copy new binary into place
        3. Launch new binary and verify it started (up to 30 s)
        4. On any failure: rollback from backup and relaunch old version
        5. On success: remove backup, clean up temp files
        """
        logger.info(f"[AutoUpdate] Linux install: {version}")

        extract_dir = self.TEMP_DIR / f"extract_{version}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            file_name = download_path.name.lower()

            is_appimage = file_name.endswith(".appimage")

            if download_path.suffix == ".zip":
                logger.info("[AutoUpdate] Extracting ZIP")
                self._safe_extract_zip(download_path, extract_dir)
            elif file_name.endswith(".tar.gz") or file_name.endswith(".tgz"):
                logger.info("[AutoUpdate] Extracting TAR.GZ")
                self._safe_extract_tar(download_path, extract_dir)
            elif is_appimage:
                logger.info("[AutoUpdate] Staging AppImage")
                staged = extract_dir / download_path.name
                shutil.copy2(download_path, staged)
                staged.chmod(0o755)
            elif file_name.endswith(".deb"):
                logger.info("[AutoUpdate] Installing DEB")
                priv_cmd = self._find_graphical_privilege_helper()
                subprocess.Popen(
                    [priv_cmd, "dpkg", "-i", str(download_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            elif file_name.endswith(".rpm"):
                logger.info("[AutoUpdate] Installing RPM")
                priv_cmd = self._find_graphical_privilege_helper()
                subprocess.Popen(
                    [priv_cmd, "rpm", "-Uvh", str(download_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            else:
                raise RuntimeError(f"Unsupported format: {download_path.suffix}")

            if is_appimage:
                new_exe = staged
            else:
                exe_name = "memento-s"
                new_exe = None
                for path in extract_dir.rglob("*"):
                    if path.is_file() and path.name.lower() == exe_name.lower():
                        new_exe = path
                        break

            if not new_exe:
                raise RuntimeError("Executable not found in update package")

            current_exe = Path(sys.executable)
            backup_exe = Path(f"{current_exe}.backup")
            logger.info(f"[AutoUpdate] Current: {current_exe}")

            app_pid = os.getpid()
            script_path = self.TEMP_DIR / f"update_{version}.sh"
            _q = shlex.quote
            script_lines = [
                "#!/bin/bash",
                "set -u",
                "",
                f"CURRENT_EXE={_q(str(current_exe))}",
                f"BACKUP_EXE={_q(str(backup_exe))}",
                f"NEW_EXE={_q(str(new_exe))}",
                f"EXTRACT_DIR={_q(str(extract_dir))}",
                f"DOWNLOAD_PKG={_q(str(download_path))}",
                f"CACHE_META={_q(str(self.CACHE_METADATA_FILE))}",
                f"SCRIPT_PATH={_q(str(script_path))}",
                f"APP_PID={app_pid}",
                "",
                "cleanup_temp() {",
                '    rm -rf "$EXTRACT_DIR" 2>/dev/null || true',
                '    rm -f "$DOWNLOAD_PKG" 2>/dev/null || true',
                '    rm -f "$CACHE_META" 2>/dev/null || true',
                '    rm -f "$SCRIPT_PATH" 2>/dev/null || true',
                "}",
                "",
                "rollback() {",
                '    if [ -f "$BACKUP_EXE" ]; then',
                '        rm -f "$CURRENT_EXE" 2>/dev/null || true',
                '        mv "$BACKUP_EXE" "$CURRENT_EXE" || exit 1',
                '        chmod +x "$CURRENT_EXE"',
                '        "$CURRENT_EXE" --updated-rollback &',
                "    fi",
                "    cleanup_temp",
                "    exit 1",
                "}",
                "",
                "# Wait for the app process to actually exit",
                "ELAPSED=0",
                'while kill -0 "$APP_PID" 2>/dev/null; do',
                "    sleep 1",
                "    ELAPSED=$((ELAPSED + 1))",
                '    if [ "$ELAPSED" -ge 30 ]; then',
                "        cleanup_temp",
                "        exit 1",
                "    fi",
                "done",
                "",
                'if [ ! -f "$NEW_EXE" ]; then',
                "    rollback",
                "fi",
                "",
                'rm -f "$BACKUP_EXE" 2>/dev/null || true',
                "",
                'if [ -f "$CURRENT_EXE" ]; then',
                '    mv "$CURRENT_EXE" "$BACKUP_EXE" || rollback',
                "fi",
                "",
                'cp "$NEW_EXE" "$CURRENT_EXE" || rollback',
                'chmod +x "$CURRENT_EXE"',
                "",
                'if ! [ -x "$CURRENT_EXE" ]; then',
                "    rollback",
                "fi",
                "",
                '"$CURRENT_EXE" --updated &',
                "",
                "APP_STARTED=false",
                "for wait_sec in $(seq 1 30); do",
                "    sleep 1",
                '    if pgrep -f "$CURRENT_EXE" >/dev/null 2>&1; then',
                "        APP_STARTED=true",
                "        break",
                "    fi",
                "done",
                "",
                'if [ "$APP_STARTED" = false ]; then',
                "    rollback",
                "fi",
                "",
                'rm -f "$BACKUP_EXE" 2>/dev/null || true',
                "cleanup_temp",
                "",
                "exit 0",
            ]
            script_path.write_text("\n".join(script_lines))
            script_path.chmod(0o755)

            logger.info(f"[AutoUpdate] Launching Linux update script: {script_path}")

            subprocess.Popen(
                ["bash", str(script_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            return True

        except Exception as e:
            logger.error(f"[AutoUpdate] Linux error: {e}", exc_info=True)
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
            raise


__all__ = [
    "AutoUpdateManager",
    "CallbackGroup",
    "UpdateStatus",
    "UpdateInfo",
    "DownloadProgress",
    "UpdateCache",
]
