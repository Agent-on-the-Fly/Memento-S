"""
Test script for auto-update functionality.

This script tests various components of the auto-update system
without requiring the full GUI application.

Usage:
    # First, start the mock OTA server:
    python tests/auto_update/mock_ota_server.py

    # Then run tests:
    python tests/auto_update/test_auto_update.py check      # Test update check
    python tests/auto_update/test_auto_update.py download   # Test download
    python tests/auto_update/test_auto_update.py full       # Test full flow
    python tests/auto_update/test_auto_update.py cache      # Test cache management
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock g_config for testing
import unittest.mock
from dataclasses import dataclass


@dataclass
class MockOTAConfig:
    """Mock OTA config for testing."""

    url: str = "http://localhost:8888/api/check"
    auto_check: bool = True
    auto_download: bool = True


@dataclass
class MockConfig:
    """Mock config for testing."""

    ota: MockOTAConfig = None

    def __post_init__(self):
        if self.ota is None:
            self.ota = MockOTAConfig()


# Create global mock config
mock_config = MockConfig()

# Patch g_config before importing
with unittest.mock.patch.dict(
    "sys.modules",
    {
        "middleware.config": unittest.mock.MagicMock(g_config=mock_config),
    },
):
    from gui.modules.auto_update_manager import (
        AutoUpdateManager,
        UpdateStatus,
        UpdateInfo,
    )


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{'=' * 60}")
    print(f" {text}")
    print(f"{'=' * 60}\n")


def print_status(manager: AutoUpdateManager):
    """Print current status."""
    print(f"Status: {manager.status.name}")
    if manager.current_update:
        print(f"Current Update: {manager.current_update.version}")
    if manager.has_cached_update:
        print(f"Cached Update: Yes ({manager._cache.version})")
    else:
        print(f"Cached Update: No")
    print()


async def test_check_update():
    """Test update checking."""
    print_header("Testing Update Check")

    manager = AutoUpdateManager()

    # Set callbacks
    def on_status(status: UpdateStatus):
        print(f"  → Status changed: {status.name}")

    def on_error(msg: str):
        print(f"  ✗ Error: {msg}")

    manager.add_listener(on_status_change=on_status, on_error=on_error)

    print("Checking for updates...")
    print(f"  OTA URL: {mock_config.ota.url}")

    try:
        update_info = await manager.check_for_update()

        if update_info:
            print(f"\n✓ Update found!")
            print(f"  Version: {update_info.version}")
            print(f"  Current: {update_info.current_version}")
            print(f"  Download URL: {update_info.download_url}")
            print(
                f"  Size: {update_info.size:,} bytes ({update_info.size / (1024 * 1024):.1f} MB)"
            )
            if update_info.checksum:
                print(f"  Checksum: {update_info.checksum}")
            print(f"\nRelease Notes:\n{update_info.release_notes}")
            return update_info
        else:
            print("\n✗ No updates available")
            return None

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_download():
    """Test downloading update."""
    print_header("Testing Download")

    manager = AutoUpdateManager()

    # First check for updates
    update_info = await manager.check_for_update()
    if not update_info:
        print("No update available, cannot test download")
        return

    print(f"Starting download of {update_info.version}...")
    print(f"Cache directory: {manager.CACHE_DIR}")

    progress_count = 0

    def on_progress(progress):
        nonlocal progress_count
        progress_count += 1
        if progress_count % 10 == 0:  # Print every 10th update
            pct = progress.percentage * 100
            mb = progress.downloaded / (1024 * 1024)
            speed = progress.speed / (1024 * 1024) if progress.speed > 0 else 0
            print(f"  Progress: {pct:.1f}% ({mb:.2f} MB) @ {speed:.2f} MB/s")

    def on_complete(info):
        print(f"\n✓ Download complete: {info.version}")

    def on_error(msg):
        print(f"\n✗ Download error: {msg}")

    manager.add_listener(
        on_progress=on_progress,
        on_download_complete=on_complete,
        on_error=on_error,
    )

    try:
        success = await manager.download_update()

        if success:
            print(f"\n✓ Download successful")
            print_status(manager)

            # Show cache info
            if manager._cache:
                print(f"Cached file: {manager._cache.download_path}")
                print(f"File size: {manager._cache.downloaded_size:,} bytes")
        else:
            print(f"\n✗ Download failed")

    except Exception as e:
        print(f"\n✗ Exception: {e}")
        import traceback

        traceback.print_exc()


async def test_cache_management():
    """Test cache management."""
    print_header("Testing Cache Management")

    manager = AutoUpdateManager()

    print("Current cache status:")
    print_status(manager)

    # Check for update and download
    await manager.check_for_update()
    if manager.current_update:
        await manager.download_update()

    print("\nAfter download:")
    print_status(manager)

    if manager.has_cached_update:
        print(f"\nCache details:")
        print(f"  Version: {manager._cache.version}")
        print(f"  Path: {manager._cache.download_path}")
        print(f"  Size: {manager._cache.downloaded_size:,} bytes")
        print(f"  Timestamp: {manager._cache.timestamp}")
        print(f"  Installed: {manager._cache.installed}")

        # Clear cache
        print("\nClearing cache...")
        manager.clear_cache()
        print("Cache cleared")

        print("\nAfter clearing:")
        print_status(manager)
    else:
        print("\nNo cache to test")


async def test_full_flow():
    """Test complete update flow."""
    print_header("Testing Full Update Flow")

    manager = AutoUpdateManager()

    # Clear any existing cache
    manager.clear_cache()
    print("✓ Cache cleared")

    # Step 1: Check for updates
    print("\n[Step 1] Checking for updates...")
    update_info = await manager.check_for_update()

    if not update_info:
        print("✗ No update available")
        return

    print(f"✓ Update available: {update_info.version}")

    # Step 2: Download
    print("\n[Step 2] Downloading update...")

    def on_progress(progress):
        pct = progress.percentage * 100
        mb = progress.downloaded / (1024 * 1024)
        print(f"  Progress: {pct:.1f}% ({mb:.2f} MB)", end="\r")

    manager.add_listener(on_progress=on_progress)

    success = await manager.download_update()
    print()  # New line after progress

    if not success:
        print("✗ Download failed")
        return

    print("✓ Download complete")

    # Step 3: Verify cache
    print("\n[Step 3] Verifying cache...")
    if manager.has_cached_update:
        print(f"✓ Update cached: {manager._cache.download_path}")
        print(f"  Size: {manager._cache.downloaded_size:,} bytes")
    else:
        print("✗ Cache not found")
        return

    # Step 4: Show what would happen on install
    print("\n[Step 4] Install simulation:")
    print(f"  Would install: {manager._cache.version}")
    print(f"  From: {manager._cache.download_path}")
    print(f"  Platform: {__import__('platform').system()}")
    print("  ⚠ Note: Actual installation would restart the app")

    print("\n✓ Full flow test completed successfully!")


async def interactive_demo():
    """Interactive demo with user prompts."""
    print_header("Auto-Update Interactive Demo")

    print("This demo will test the auto-update system.")
    print(
        f"Make sure the mock server is running: python tests/auto_update/mock_ota_server.py"
    )
    print()

    manager = AutoUpdateManager()

    # Check
    input("Press Enter to check for updates...")
    update_info = await manager.check_for_update()

    if not update_info:
        print("No updates available")
        return

    print(f"\nUpdate found: {update_info.version}")
    print(f"Size: {update_info.size / (1024 * 1024):.1f} MB")

    # Download
    response = input("\nDownload now? (y/n): ")
    if response.lower() != "y":
        print("Download cancelled")
        return

    print("\nDownloading...")
    success = await manager.download_update()

    if not success:
        print("Download failed")
        return

    print("✓ Download complete!")
    print(f"Cached at: {manager._cache.download_path}")

    # Install
    response = input("\nInstall now? This will restart the app (y/n): ")
    if response.lower() != "y":
        print("Installation deferred. Update is cached and ready.")
        return

    print("\nInstalling...")
    # Note: In real scenario, this would install and restart
    print("✓ Installation would proceed here (simulated)")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <command>")
        print("\nCommands:")
        print("  check     - Test update check")
        print("  download  - Test download")
        print("  full      - Test full flow")
        print("  cache     - Test cache management")
        print("  demo      - Interactive demo")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "check":
        asyncio.run(test_check_update())
    elif command == "download":
        asyncio.run(test_download())
    elif command == "full":
        asyncio.run(test_full_flow())
    elif command == "cache":
        asyncio.run(test_cache_management())
    elif command == "demo":
        asyncio.run(interactive_demo())
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
