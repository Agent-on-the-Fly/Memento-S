#!/usr/bin/env python3
"""
Quick demo for auto-update functionality.

This is a standalone demo that simulates the update process
without requiring a real OTA server.

Usage:
    python tests/auto_update/demo.py
"""

import asyncio
import random
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class DemoStatus(Enum):
    IDLE = auto()
    CHECKING = auto()
    AVAILABLE = auto()
    DOWNLOADING = auto()
    PAUSED = auto()
    DOWNLOADED = auto()
    INSTALLING = auto()
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class DemoUpdateInfo:
    version: str
    current_version: str
    size: int
    release_notes: str


class AutoUpdateDemo:
    """Demo auto-updater with simulated network operations."""

    def __init__(self):
        self.status = DemoStatus.IDLE
        self.progress = 0.0
        self.speed = 0.0
        self.update_info: DemoUpdateInfo | None = None
        self.cancelled = False

    def print_header(self, text: str):
        """Print header."""
        print(f"\n{'=' * 50}")
        print(f" {text}")
        print(f"{'=' * 50}\n")

    def print_progress(self):
        """Print progress bar."""
        bar_length = 40
        filled = int(bar_length * self.progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        pct = self.progress * 100
        speed_mb = self.speed / (1024 * 1024)
        print(f"\r[{bar}] {pct:.1f}% ({speed_mb:.1f} MB/s)", end="", flush=True)

    async def check_for_updates(self) -> bool:
        """Simulate checking for updates."""
        self.print_header("Checking for Updates")

        self.status = DemoStatus.CHECKING
        print("🌐 Connecting to update server...")
        await asyncio.sleep(0.5)

        print("📡 Sending version info...")
        await asyncio.sleep(0.5)

        print("📥 Waiting for response...")
        await asyncio.sleep(1)

        # Simulate finding an update
        self.update_info = DemoUpdateInfo(
            version="1.1.0",
            current_version="1.0.0",
            size=25_165_824,  # ~24 MB
            release_notes="""
版本 1.1.0 更新内容：

✨ 新功能
  - 新增自动更新功能
  - 支持断点续传
  - 后台静默下载

🚀 性能优化
  - 启动速度提升 30%
  - 内存占用降低 20%

🐛 问题修复
  - 修复若干已知问题
  - 提升稳定性
            """.strip(),
        )

        self.status = DemoStatus.AVAILABLE
        print(f"\n✅ Update available!")
        print(f"   Version: {self.update_info.version}")
        print(f"   Current: {self.update_info.current_version}")
        print(f"   Size: {self.update_info.size / (1024 * 1024):.1f} MB")
        print(f"\n📝 Release Notes:\n{self.update_info.release_notes}")

        return True

    async def download_update(self) -> bool:
        """Simulate downloading update."""
        self.print_header("Downloading Update")

        if not self.update_info:
            print("❌ No update information")
            return False

        self.status = DemoStatus.DOWNLOADING
        print(f"📦 Downloading: {self.update_info.version}")
        print(f"💾 Size: {self.update_info.size / (1024 * 1024):.1f} MB")
        print()

        # Simulate download progress
        total_size = self.update_info.size
        downloaded = 0
        chunk_size = 512 * 1024  # 512 KB chunks

        while downloaded < total_size and not self.cancelled:
            # Simulate network speed variation
            self.speed = random.uniform(2 * 1024 * 1024, 5 * 1024 * 1024)  # 2-5 MB/s

            # Calculate next chunk
            chunk = min(chunk_size, total_size - downloaded)
            download_time = chunk / self.speed

            await asyncio.sleep(download_time)
            downloaded += chunk
            self.progress = downloaded / total_size

            self.print_progress()

        if self.cancelled:
            print("\n\n❌ Download cancelled")
            return False

        print("\n")
        self.status = DemoStatus.DOWNLOADED
        print("✅ Download complete!")
        print(
            f"💾 Cached to: ~/memento_s/updates/update_{self.update_info.version}.zip"
        )

        return True

    def cancel(self):
        """Cancel download."""
        self.cancelled = True

    async def install_update(self) -> bool:
        """Simulate installing update."""
        self.print_header("Installing Update")

        if self.status != DemoStatus.DOWNLOADED:
            print("❌ No update to install")
            return False

        self.status = DemoStatus.INSTALLING
        print("🔧 Preparing installation...")
        await asyncio.sleep(0.5)

        print("📂 Extracting update package...")
        await asyncio.sleep(0.5)

        print("💾 Creating backup of current version...")
        await asyncio.sleep(0.5)

        print("🔄 Replacing files...")
        await asyncio.sleep(0.5)

        print("🧹 Cleaning up...")
        await asyncio.sleep(0.3)

        self.status = DemoStatus.COMPLETED
        print("\n✅ Installation complete!")
        print("🚀 Application will now restart...")

        return True

    async def show_notification(self):
        """Show download complete notification."""
        self.print_header("Update Notification")

        print("┌" + "─" * 48 + "┐")
        print("│" + " " * 48 + "│")
        print("│  🔔 Update Ready" + " " * 32 + "│")
        print("│" + " " * 48 + "│")
        print(
            f"│  Version {self.update_info.version} has been downloaded"
            + " " * 10
            + "│"
        )
        print("│" + " " * 48 + "│")
        print("│  [Install Now]              [Later]             │")
        print("│" + " " * 48 + "│")
        print("└" + "─" * 48 + "┘")

    async def run_demo(self):
        """Run complete demo."""
        print("\n" + "🚀" * 25)
        print("    Memento-S Auto-Update Demo")
        print("🚀" * 25)

        # Step 1: Check
        if not await self.check_for_updates():
            print("\n❌ Demo failed: Could not check for updates")
            return

        input("\n⏎ Press Enter to start download...")

        # Step 2: Download
        if not await self.download_update():
            print("\n❌ Demo failed: Download error")
            return

        # Step 3: Show notification
        await self.show_notification()

        response = input("\n❓ Install now? (y/n): ")
        if response.lower() != "y":
            print("\n⏸️  Installation deferred. Update is cached for later.")
            print("   It will be installed on next app restart.")
            return

        # Step 4: Install
        if not await self.install_update():
            print("\n❌ Demo failed: Installation error")
            return

        self.print_header("Demo Complete")
        print("✨ All steps completed successfully!")
        print("\nIn a real scenario, the app would now restart")
        print("and run the new version automatically.")


async def quick_demo():
    """Quick demo showing all states."""
    demo = AutoUpdateDemo()

    print("\n⚡ Quick Demo - Auto-Update Flow\n")

    # Simulate states
    states = [
        ("IDLE", "Waiting for user...", 0.5),
        ("CHECKING", "🔍 Checking for updates...", 1.0),
        ("AVAILABLE", "📢 New version found!", 0.5),
        ("DOWNLOADING", "⬇️  Downloading update...", 2.0),
        ("DOWNLOADED", "✅ Download complete!", 0.5),
        ("INSTALLING", "🔧 Installing update...", 1.5),
        ("COMPLETED", "🎉 Update installed!", 0.5),
    ]

    for state, description, delay in states:
        print(f"{state:12} | {description}")
        await asyncio.sleep(delay)

    print("\n✨ Demo complete!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto-Update Demo")
    parser.add_argument("--quick", action="store_true", help="Run quick state demo")
    parser.add_argument(
        "--step",
        choices=["check", "download", "install"],
        help="Run specific step",
    )

    args = parser.parse_args()

    if args.quick:
        asyncio.run(quick_demo())
    elif args.step:
        demo = AutoUpdateDemo()
        if args.step == "check":
            asyncio.run(demo.check_for_updates())
        elif args.step == "download":
            demo.update_info = DemoUpdateInfo(
                version="1.1.0",
                current_version="1.0.0",
                size=25_165_824,
                release_notes="Test",
            )
            asyncio.run(demo.download_update())
        elif args.step == "install":
            demo.status = DemoStatus.DOWNLOADED
            demo.update_info = DemoUpdateInfo(
                version="1.1.0",
                current_version="1.0.0",
                size=25_165_824,
                release_notes="Test",
            )
            asyncio.run(demo.install_update())
    else:
        # Full interactive demo
        demo = AutoUpdateDemo()
        try:
            asyncio.run(demo.run_demo())
        except KeyboardInterrupt:
            print("\n\n⚠️  Demo cancelled")


if __name__ == "__main__":
    main()
