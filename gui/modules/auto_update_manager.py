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
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

import httpx
from packaging.version import parse as parse_version

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
class UpdateCache:
    """Persistent update cache metadata."""

    version: str
    download_path: Path
    checksum: str | None = None
    downloaded_size: int = 0
    total_size: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    installed: bool = False

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
        )


class AutoUpdateManager:
    """
    Manages automatic application updates.
    """

    CACHE_DIR = Path.home() / ".memento-s" / "updates"
    CACHE_METADATA_FILE = CACHE_DIR / "cache.json"
    TEMP_DIR = Path.home() / ".memento-s" / "temp" / "updates"

    STARTUP_CHECK_DELAY = 10

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

        self._on_status_change: Callable[[UpdateStatus], None] | None = None
        self._on_progress: Callable[[DownloadProgress], None] | None = None
        self._on_download_complete: Callable[[UpdateInfo], None] | None = None
        self._on_error: Callable[[str], None] | None = None

        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)

        self._load_cache()

    @property
    def status(self) -> UpdateStatus:
        """Current update status."""
        return self._status

    @property
    def current_update(self) -> UpdateInfo | None:
        """Current update information."""
        return self._current_update

    @property
    def progress(self) -> DownloadProgress:
        """Current download progress."""
        return self._progress

    @property
    def has_cached_update(self) -> bool:
        """Check if there's a cached update ready to install."""
        if self._cache is None or self._cache.installed:
            return False
        return self._cache.download_path.exists()

    def set_callbacks(
        self,
        on_status_change: Callable[[UpdateStatus], None] | None = None,
        on_progress: Callable[[DownloadProgress], None] | None = None,
        on_download_complete: Callable[[UpdateInfo], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ):
        """Set event callbacks."""
        self._on_status_change = on_status_change
        self._on_progress = on_progress
        self._on_download_complete = on_download_complete
        self._on_error = on_error

    def _set_status(self, status: UpdateStatus):
        """Set status and trigger callback."""
        self._status = status
        if self._on_status_change:
            try:
                self._on_status_change(status)
            except Exception as e:
                logger.error(f"[AutoUpdate] Status callback error: {e}")
        logger.info(f"[AutoUpdate] Status: {status.name}")

    def _notify_progress(self):
        """Notify progress update."""
        if self._on_progress:
            try:
                self._on_progress(self._progress)
            except Exception as e:
                logger.error(f"[AutoUpdate] Progress callback error: {e}")

    def _notify_error(self, message: str):
        """Notify error."""
        logger.error(f"[AutoUpdate] Error: {message}")
        if self._on_error:
            try:
                self._on_error(message)
            except Exception as e:
                logger.error(f"[AutoUpdate] Error callback error: {e}")

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
        try:
            from importlib.metadata import version as _pkg_version

            return _pkg_version("memento-s")
        except Exception:
            return "0.1.0"

    async def start_auto_check(self):
        """Start automatic update check after startup delay."""
        cfg = (
            self._config.load()
            if hasattr(self._config, "load")
            else self._config
        )
        if not cfg or not cfg.ota or not cfg.ota.url:
            logger.info("[AutoUpdate] OTA not configured")
            return

        if not getattr(cfg.ota, "auto_check", True):
            logger.info("[AutoUpdate] Auto check disabled")
            return

        logger.info(f"[AutoUpdate] Check in {self.STARTUP_CHECK_DELAY}s")
        await asyncio.sleep(self.STARTUP_CHECK_DELAY)

        if self.has_cached_update:
            logger.info(f"[AutoUpdate] Found cached: {self._cache.version}")
            self._current_update = UpdateInfo(
                version=self._cache.version,
                current_version=self._get_current_version(),
                download_url="",
            )
            self._set_status(UpdateStatus.DOWNLOADED)
            if self._on_download_complete:
                self._on_download_complete(self._current_update)
            return

        await self.check_for_update()

    async def check_for_update(self) -> UpdateInfo | None:
        """Check for available updates from OTA server."""
        self._set_status(UpdateStatus.CHECKING)

        try:
            cfg = (
                self._config.load()
                if hasattr(self._config, "load")
                else self._config
            )
            if not cfg or not cfg.ota or not cfg.ota.url:
                logger.info("[AutoUpdate] OTA not configured")
                self._set_status(UpdateStatus.IDLE)
                return None

            current_version = self._get_current_version()
            params = {
                "current_version": current_version,
                "platform": platform.system().lower(),
                "arch": platform.machine().lower(),
            }

            logger.info(f"[AutoUpdate] Checking: {cfg.ota.url}")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    cfg.ota.url, params=params, timeout=10.0
                )
                response.raise_for_status()
                data = response.json()

            if not data.get("update_available"):
                logger.info("[AutoUpdate] No updates")
                self._set_status(UpdateStatus.IDLE)
                return None

            update_info = UpdateInfo(
                version=data.get("latest_version", ""),
                current_version=current_version,
                download_url=data.get("download_url", ""),
                release_notes=data.get("release_notes", ""),
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

            auto_download = getattr(cfg.ota, "auto_download", True)
            if auto_download:
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
        """Download the update package."""
        if update_info:
            self._current_update = update_info
        elif not self._current_update:
            self._notify_error("No update info")
            return False

        update_info = self._current_update

        if self.has_cached_update and self._cache.version == update_info.version:
            logger.info(f"[AutoUpdate] Already downloaded: {update_info.version}")
            self._set_status(UpdateStatus.DOWNLOADED)
            if self._on_download_complete:
                self._on_download_complete(update_info)
            return True

        self._set_status(UpdateStatus.DOWNLOADING)
        self._cancelled = False
        self._paused = False

        url = update_info.download_url
        file_ext = Path(url).suffix or self._get_platform_extension()
        download_path = self.CACHE_DIR / f"update_{update_info.version}{file_ext}"

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

                    if "Content-Length" in response.headers:
                        content_length = int(response.headers["Content-Length"])
                        if resume_byte_pos > 0 and response.status_code == 206:
                            self._progress.total_size = resume_byte_pos + content_length
                        else:
                            self._progress.total_size = content_length
                    elif update_info.size > 0:
                        self._progress.total_size = update_info.size

                    mode = "ab" if resume_byte_pos > 0 else "wb"
                    with open(download_path, mode) as f:
                        last_progress_time = time.time()
                        bytes_since_last = 0

                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            if self._cancelled:
                                logger.info("[AutoUpdate] Cancelled")
                                self._set_status(UpdateStatus.CANCELLED)
                                return False

                            while self._paused:
                                self._set_status(UpdateStatus.PAUSED)
                                await asyncio.sleep(0.5)
                                if self._cancelled:
                                    return False

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
                        if not self._verify_checksum(download_path, update_info.checksum):
                            logger.error("[AutoUpdate] Checksum failed")
                            download_path.unlink(missing_ok=True)
                            self._set_status(UpdateStatus.ERROR)
                            self._notify_error("Verification failed")
                            return False

                    logger.info(f"[AutoUpdate] Downloaded: {download_path}")
                    self._finish_download(download_path, update_info)
                    return True

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
        )
        self._save_cache()
        self._set_status(UpdateStatus.DOWNLOADED)

        if self._on_download_complete:
            self._on_download_complete(update_info)

    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum."""
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
        """Cancel ongoing download."""
        self._cancelled = True
        self._paused = False
        if self._download_task and not self._download_task.done():
            self._download_task.cancel()
        self._set_status(UpdateStatus.CANCELLED)
        logger.info("[AutoUpdate] Cancelled")

    async def install_update(
        self,
        page: Any | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> bool:
        """Install the downloaded update."""
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
                self._cache.installed = True
                self._save_cache()
                self._set_status(UpdateStatus.COMPLETED)
                logger.info(f"[AutoUpdate] Installed: {version}")
                if on_complete:
                    on_complete()
                return True
            else:
                self._set_status(UpdateStatus.ERROR)
                return False

        except Exception as e:
            logger.error(f"[AutoUpdate] Install failed: {e}", exc_info=True)
            self._set_status(UpdateStatus.ERROR)
            self._notify_error(f"Install failed: {e}")
            return False

    async def _install_macos(self, download_path: Path, version: str) -> bool:
        """Install update on macOS."""
        logger.info(f"[AutoUpdate] macOS install: {version}")

        extract_dir = self.TEMP_DIR / f"extract_{version}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Extract
            if download_path.suffix == ".zip":
                logger.info("[AutoUpdate] Extracting ZIP")
                with zipfile.ZipFile(download_path, "r") as zf:
                    zf.extractall(extract_dir)
            elif str(download_path).endswith(".tar.gz") or str(download_path).endswith(".tgz"):
                import tarfile
                logger.info("[AutoUpdate] Extracting TAR.GZ")
                with tarfile.open(download_path, "r:gz") as tar:
                    tar.extractall(extract_dir)
            elif download_path.suffix == ".dmg":
                logger.info("[AutoUpdate] Opening DMG")
                subprocess.Popen(["open", str(download_path)])
                return True
            else:
                logger.warning(f"[AutoUpdate] Unknown type: {download_path.suffix}")
                try:
                    with zipfile.ZipFile(download_path, "r") as zf:
                        zf.extractall(extract_dir)
                except zipfile.BadZipFile:
                    raise RuntimeError(f"Cannot extract: {download_path}")

            # Find .app bundle
            app_bundles = list(extract_dir.rglob("*.app"))
            if not app_bundles:
                raise RuntimeError("No .app bundle found")

            new_app = app_bundles[0]
            logger.info(f"[AutoUpdate] New app: {new_app}")

            # Find current app
            current_app = None
            app_dir = Path(sys.executable).parent
            for path in [app_dir, app_dir.parent]:
                apps = list(path.glob("*.app"))
                if apps:
                    current_app = apps[0]
                    break

            if not current_app:
                raise RuntimeError("Current app not found")

            logger.info(f"[AutoUpdate] Current app: {current_app}")

            # Create update script
            script_path = self.TEMP_DIR / f"update_{version}.sh"
            script_lines = [
                "#!/bin/bash",
                "set -e",
                "",
                "sleep 2",
                "",
                f'if [ -d "{current_app}" ]; then',
                f'    mv "{current_app}" "{current_app}.backup"',
                "fi",
                "",
                f'mv "{new_app}" "{current_app}"',
                "",
                f'rm -rf "{current_app}.backup"',
                "",
                f'open "{current_app}"',
                "",
                f'rm -rf "{extract_dir}"',
                f'rm "{script_path}"',
            ]
            script_path.write_text("\n".join(script_lines))
            script_path.chmod(0o755)

            subprocess.Popen(
                ["bash", str(script_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            return True

        except Exception as e:
            logger.error(f"[AutoUpdate] macOS error: {e}")
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
            raise

    async def _install_windows(self, download_path: Path, version: str) -> bool:
        """Install update on Windows."""
        logger.info(f"[AutoUpdate] Windows install: {version}")

        extract_dir = self.TEMP_DIR / f"extract_{version}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            if download_path.suffix == ".zip":
                logger.info("[AutoUpdate] Extracting ZIP")
                with zipfile.ZipFile(download_path, "r") as zf:
                    zf.extractall(extract_dir)
            elif download_path.suffix in [".exe", ".msi"]:
                logger.info("[AutoUpdate] Running installer")
                subprocess.Popen([str(download_path)], shell=True)
                return True
            else:
                raise RuntimeError(f"Unsupported format: {download_path.suffix}")

            # Find executable
            exe_name = "memento-s.exe"
            new_exe = None
            for path in extract_dir.rglob("*.exe"):
                if path.name.lower() == exe_name.lower():
                    new_exe = path
                    break

            if not new_exe:
                raise RuntimeError(f"{exe_name} not found")

            current_exe = Path(sys.executable)
            logger.info(f"[AutoUpdate] Current: {current_exe}")

            # Create batch script
            script_path = self.TEMP_DIR / f"update_{version}.bat"
            script_lines = [
                "@echo off",
                "timeout /t 2 /nobreak >nul",
                "",
                f'move "{current_exe}" "{current_exe}.backup"',
                "",
                f'copy "{new_exe}" "{current_exe}"',
                "",
                f'del "{current_exe}.backup"',
                "",
                f'start "" "{current_exe}"',
                "",
                f'rmdir /s /q "{extract_dir}"',
                f'del "{script_path}"',
            ]
            script_path.write_text("\n".join(script_lines))

            subprocess.Popen(
                ["cmd", "/c", str(script_path)],
                shell=False,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )

            return True

        except Exception as e:
            logger.error(f"[AutoUpdate] Windows error: {e}")
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
            raise

    async def _install_linux(self, download_path: Path, version: str) -> bool:
        """Install update on Linux."""
        logger.info(f"[AutoUpdate] Linux install: {version}")

        extract_dir = self.TEMP_DIR / f"extract_{version}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            file_name = download_path.name.lower()

            if download_path.suffix == ".zip":
                logger.info("[AutoUpdate] Extracting ZIP")
                with zipfile.ZipFile(download_path, "r") as zf:
                    zf.extractall(extract_dir)
            elif file_name.endswith(".tar.gz") or file_name.endswith(".tgz"):
                import tarfile
                logger.info("[AutoUpdate] Extracting TAR.GZ")
                with tarfile.open(download_path, "r:gz") as tar:
                    tar.extractall(extract_dir)
            elif file_name.endswith(".appimage"):
                logger.info("[AutoUpdate] Installing AppImage")
                current_exe = Path(sys.executable)
                shutil.copy2(download_path, current_exe)
                current_exe.chmod(0o755)
                return True
            elif file_name.endswith(".deb"):
                logger.info("[AutoUpdate] Installing DEB")
                subprocess.Popen(
                    ["sudo", "dpkg", "-i", str(download_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            elif file_name.endswith(".rpm"):
                logger.info("[AutoUpdate] Installing RPM")
                subprocess.Popen(
                    ["sudo", "rpm", "-Uvh", str(download_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            else:
                raise RuntimeError(f"Unsupported format: {download_path.suffix}")

            # Find executable
            exe_name = "memento-s"
            new_exe = None
            for path in extract_dir.rglob("*"):
                if path.is_file() and path.name.lower() == exe_name.lower():
                    new_exe = path
                    break

            if not new_exe:
                raise RuntimeError(f"{exe_name} not found")

            current_exe = Path(sys.executable)
            logger.info(f"[AutoUpdate] Current: {current_exe}")

            # Create shell script
            script_path = self.TEMP_DIR / f"update_{version}.sh"
            script_lines = [
                "#!/bin/bash",
                "set -e",
                "",
                "sleep 2",
                "",
                f'mv "{current_exe}" "{current_exe}.backup"',
                "",
                f'cp "{new_exe}" "{current_exe}"',
                f'chmod +x "{current_exe}"',
                "",
                f'rm "{current_exe}.backup"',
                "",
                f'"{current_exe}" &',
                "",
                f'rm -rf "{extract_dir}"',
                f'rm "{script_path}"',
            ]
            script_path.write_text("\n".join(script_lines))
            script_path.chmod(0o755)

            subprocess.Popen(
                ["bash", str(script_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            return True

        except Exception as e:
            logger.error(f"[AutoUpdate] Linux error: {e}")
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
            raise


__all__ = [
    "AutoUpdateManager",
    "UpdateStatus",
    "UpdateInfo",
    "DownloadProgress",
    "UpdateCache",
]
