"""
Log upload functionality for Weixin SDK.

Provides CLI and programmatic interfaces for uploading logs for troubleshooting.
Supports filtering by account, date, and uploading to a remote endpoint.
"""

import json
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin


# Default log directory (relative to state directory)
DEFAULT_LOG_DIR = "logs"

# Environment variable for upload URL
UPLOAD_URL_ENV = "WEIXIN_LOG_UPLOAD_URL"

# Config file name
CONFIG_FILE = "openclaw.json"

# Log file patterns
LOG_FILE_PATTERN = re.compile(r"weixin_sdk_.*\.log(?:\.\d+)?$")
ACCOUNT_LOG_PATTERN = re.compile(r"weixin_sdk_(\w+)_.*\.log(?:\.\d+)?$")


class LogUploadError(Exception):
    """Exception raised for log upload errors."""

    pass


class LogUploader:
    """
    Handles log file collection, compression, and upload.

    Example:
        uploader = LogUploader(state_dir=Path.home() / ".weixin_sdk")
        log_files = uploader.collect_logs(account_id="bot1", date="2026-03-25")
        archive = uploader.compress_logs(log_files)
        uploader.send_logs("https://logs.example.com/upload", archive)
    """

    def __init__(
        self, state_dir: Optional[Path] = None, log_dir: Optional[Path] = None
    ):
        """
        Initialize the log uploader.

        Args:
            state_dir: Base state directory (default: ~/.weixin_sdk)
            log_dir: Log directory (default: {state_dir}/logs)
        """
        self.state_dir = state_dir or Path.home() / ".weixin_sdk"
        self.log_dir = log_dir or self.state_dir / DEFAULT_LOG_DIR

    def collect_logs(
        self,
        account_id: Optional[str] = None,
        date: Optional[str] = None,
        all_logs: bool = False,
    ) -> List[Path]:
        """
        Collect log files from the log directory.

        Args:
            account_id: Filter by account ID (optional)
            date: Filter by date in YYYY-MM-DD format (optional)
            all_logs: If True, ignore filters and collect all logs

        Returns:
            List of paths to matching log files

        Raises:
            LogUploadError: If log directory doesn't exist or date format is invalid
        """
        if not self.log_dir.exists():
            raise LogUploadError(f"Log directory does not exist: {self.log_dir}")

        log_files: List[Path] = []

        # Get all log files
        all_log_files = [
            f
            for f in self.log_dir.iterdir()
            if f.is_file() and LOG_FILE_PATTERN.match(f.name)
        ]

        for log_file in all_log_files:
            # Check account filter
            if account_id and not all_logs:
                match = ACCOUNT_LOG_PATTERN.match(log_file.name)
                if match:
                    file_account_id = match.group(1)
                    if file_account_id != account_id:
                        continue
                else:
                    # File doesn't have account pattern but we're filtering by account
                    continue

            # Check date filter
            if date and not all_logs:
                # Parse date from filename: weixin_sdk_YYYYMMDD_HHMMSS.log
                # or weixin_sdk_{account}_YYYYMMDD_HHMMSS.log
                date_match = re.search(r"(\d{8})_\d{6}", log_file.name)
                if date_match:
                    file_date_str = date_match.group(1)
                    # Convert YYYYMMDD to YYYY-MM-DD for comparison
                    file_date = (
                        f"{file_date_str[:4]}-{file_date_str[4:6]}-{file_date_str[6:8]}"
                    )
                    if file_date != date:
                        continue
                else:
                    # Check file modification time as fallback
                    try:
                        mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                        file_date = mtime.strftime("%Y-%m-%d")
                        if file_date != date:
                            continue
                    except (OSError, ValueError):
                        continue

            log_files.append(log_file)

        # Sort by modification time (newest first)
        log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        return log_files

    def compress_logs(
        self, log_files: List[Path], archive_name: Optional[str] = None
    ) -> Path:
        """
        Compress log files into a ZIP archive.

        Args:
            log_files: List of log file paths to compress
            archive_name: Optional name for the archive (default: auto-generated)

        Returns:
            Path to the created archive

        Raises:
            LogUploadError: If no files to compress or compression fails
        """
        if not log_files:
            raise LogUploadError("No log files to compress")

        # Generate archive name if not provided
        if not archive_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"weixin_sdk_logs_{timestamp}.zip"

        # Create archive in temp directory
        archive_path = self.state_dir / archive_name

        try:
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for log_file in log_files:
                    # Add file with relative path for cleaner archive
                    zf.write(log_file, arcname=log_file.name)

            return archive_path
        except Exception as e:
            # Clean up partial archive
            if archive_path.exists():
                archive_path.unlink()
            raise LogUploadError(f"Failed to compress logs: {e}") from e

    def send_logs(
        self,
        upload_url: str,
        archive_path: Path,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Send logs to upload endpoint via HTTP POST.

        Args:
            upload_url: URL to POST the archive to
            archive_path: Path to the ZIP archive
            progress_callback: Optional callback for progress updates

        Returns:
            Response data from the server

        Raises:
            LogUploadError: If upload fails
        """
        if not archive_path.exists():
            raise LogUploadError(f"Archive not found: {archive_path}")

        try:
            import requests
        except ImportError:
            # Fallback to urllib if requests not available
            return self._send_logs_urllib(upload_url, archive_path, progress_callback)

        try:
            file_size = archive_path.stat().st_size

            # Create multipart form data
            with open(archive_path, "rb") as f:
                files = {
                    "file": (archive_path.name, f, "application/zip"),
                }
                data = {
                    "timestamp": datetime.now().isoformat(),
                    "filename": archive_path.name,
                    "size": str(file_size),
                }

                # Progress callback wrapper
                if progress_callback:
                    progress_callback(0, file_size, "starting")

                response = requests.post(
                    upload_url,
                    files=files,
                    data=data,
                    timeout=300,  # 5 minute timeout for large uploads
                )

                if progress_callback:
                    progress_callback(file_size, file_size, "complete")

            response.raise_for_status()

            # Try to parse JSON response
            try:
                return response.json()
            except ValueError:
                return {"status": "success", "response": response.text}

        except requests.exceptions.RequestException as e:
            raise LogUploadError(f"Upload failed: {e}") from e
        except Exception as e:
            raise LogUploadError(f"Unexpected error during upload: {e}") from e

    def _send_logs_urllib(
        self,
        upload_url: str,
        archive_path: Path,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Send logs using urllib (fallback when requests is not available).

        Args:
            upload_url: URL to POST the archive to
            archive_path: Path to the ZIP archive
            progress_callback: Optional callback for progress updates

        Returns:
            Response data from the server
        """
        import urllib.request
        import urllib.error
        from urllib.parse import urlparse
        import mimetypes
        import uuid

        file_size = archive_path.stat().st_size

        if progress_callback:
            progress_callback(0, file_size, "starting")

        # Build multipart form data
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"

        def encode_multipart_formdata():
            """Encode the multipart form data."""
            lines = []

            # Add form fields
            fields = {
                "timestamp": datetime.now().isoformat(),
                "filename": archive_path.name,
                "size": str(file_size),
            }

            for key, value in fields.items():
                lines.append(f"--{boundary}".encode())
                lines.append(f'Content-Disposition: form-data; name="{key}"'.encode())
                lines.append(b"")
                lines.append(value.encode())

            # Add file
            lines.append(f"--{boundary}".encode())
            lines.append(
                f'Content-Disposition: form-data; name="file"; filename="{archive_path.name}"'.encode()
            )
            content_type = (
                mimetypes.guess_type(str(archive_path))[0] or "application/zip"
            )
            lines.append(f"Content-Type: {content_type}".encode())
            lines.append(b"")

            with open(archive_path, "rb") as f:
                file_content = f.read()
            lines.append(file_content)

            lines.append(f"--{boundary}--".encode())
            lines.append(b"")

            return b"\r\n".join(lines)

        body = encode_multipart_formdata()

        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        }

        try:
            request = urllib.request.Request(
                upload_url,
                data=body,
                headers=headers,
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=300) as response:
                response_body = response.read().decode("utf-8")

            if progress_callback:
                progress_callback(file_size, file_size, "complete")

            # Try to parse JSON response
            try:
                return json.loads(response_body)
            except ValueError:
                return {"status": "success", "response": response_body}

        except urllib.error.HTTPError as e:
            raise LogUploadError(f"Upload failed: HTTP {e.code} - {e.reason}") from e
        except urllib.error.URLError as e:
            raise LogUploadError(f"Upload failed: {e.reason}") from e
        except Exception as e:
            raise LogUploadError(f"Unexpected error during upload: {e}") from e

    def upload_logs(
        self,
        upload_url: str,
        log_files: List[Path],
        progress_callback: Optional[Any] = None,
        remove_archive: bool = True,
    ) -> Dict[str, Any]:
        """
        Complete upload workflow: compress and send logs.

        Args:
            upload_url: URL to upload to
            log_files: List of log files to upload
            progress_callback: Optional callback(current, total, status)
            remove_archive: Whether to remove the archive after upload (default: True)

        Returns:
            Response data from the server

        Raises:
            LogUploadError: If upload fails
        """
        # Compress logs
        archive_path = self.compress_logs(log_files)

        try:
            # Send logs
            result = self.send_logs(upload_url, archive_path, progress_callback)

            # Add archive info to result
            result["archive_path"] = str(archive_path)
            result["files_count"] = len(log_files)
            result["files"] = [f.name for f in log_files]

            return result
        finally:
            # Remove archive if requested
            if remove_archive and archive_path.exists():
                try:
                    archive_path.unlink()
                except OSError:
                    pass  # Ignore cleanup errors


def get_upload_url(
    cli_url: Optional[str] = None,
    config_path: Optional[Path] = None,
    state_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    Resolve upload URL from multiple sources (in order of priority):
    1. CLI argument (--url)
    2. Environment variable (WEIXIN_LOG_UPLOAD_URL)
    3. Config file (openclaw.json -> logUploadUrl)

    Args:
        cli_url: URL from command line argument
        config_path: Path to config file
        state_dir: State directory for finding config file

    Returns:
        Upload URL or None if not found
    """
    # Priority 1: CLI argument
    if cli_url:
        return cli_url

    # Priority 2: Environment variable
    env_url = os.environ.get(UPLOAD_URL_ENV)
    if env_url:
        return env_url

    # Priority 3: Config file
    if config_path and config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            url = config.get("logUploadUrl")
            if url:
                return url
        except (json.JSONDecodeError, OSError):
            pass

    # Try to find config in state directory
    if state_dir:
        default_config = state_dir / CONFIG_FILE
        if default_config.exists():
            try:
                config = json.loads(default_config.read_text(encoding="utf-8"))
                url = config.get("logUploadUrl")
                if url:
                    return url
            except (json.JSONDecodeError, OSError):
                pass

    return None


def format_size(size_bytes: int) -> str:
    """Format byte size as human readable string."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def print_progress(current: int, total: int, status: str) -> None:
    """Default progress printer for CLI."""
    if status == "starting":
        print(f"Uploading {format_size(total)}...", end="", flush=True)
    elif status == "complete":
        print(" done!")
    else:
        pct = (current / total) * 100 if total > 0 else 0
        print(f"\rUploading: {pct:.1f}%", end="", flush=True)
