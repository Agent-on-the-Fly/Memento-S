"""
Over-the-Air (OTA) Update Checker for Memento-S GUI.
"""

from __future__ import annotations

import asyncio
import platform
from importlib.metadata import version as get_version

import httpx
from packaging.version import parse as parse_version
from pydantic import BaseModel, Field

from middleware.config import g_config

from utils.logger import logger


class UpdateInfo(BaseModel):
    """Holds information about a new available version."""

    is_available: bool = Field(False, alias="update_available")
    current_version: str
    latest_version: str | None = None
    download_url: str | None = None


async def check_for_updates(timeout: int = 5) -> UpdateInfo | None:
    """
    Checks for a new version of the application.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        An UpdateInfo object if the check is successful, otherwise None.
    """
    cfg = g_config
    ota_url = cfg.ota.url
    if not ota_url:
        logger.debug("[OTA] Check skipped: API URL is not configured.")
        return None

    try:
        current_version_str = get_version("memento-s")
    except Exception:
        logger.warning(
            "[OTA] Could not determine current application version from config."
        )
        return None

    # Prepare request parameters
    params = {
        "current_version": current_version_str,
        "platform": platform.system().lower(),
    }

    logger.debug(f"[OTA] Request parameters: {params}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(ota_url, params=params, timeout=timeout)
            response.raise_for_status()  # Raise an exception for 4xx/5xx responses
            data = response.json()

        update_info = UpdateInfo(current_version=current_version_str, **data)

        if update_info.is_available:
            logger.info(f"[OTA] New version available: {update_info.latest_version}")
        else:
            logger.info("[OTA] Application is up to date.")

        return update_info

    except httpx.RequestError as e:
        logger.warning(f"[OTA] Network error while checking for updates: {e}")
    except (ValueError, TypeError, KeyError) as e:
        logger.warning(f"[OTA] Error parsing API response: {e}")
    except Exception as e:
        logger.error(f"[OTA] An unexpected error occurred: {e}", exc_info=True)

    return None


if __name__ == "__main__":
    # Example of how to use the checker
    async def main():
        print("Checking for updates...")
        # Temporarily set a mock URL for testing
        global OTA_API_URL
        # OTA_API_URL = "https://api.github.com/repos/some/repo/releases/latest" # Mock
        update = await check_for_updates()
        if update:
            if update.is_available:
                print(f"🎉 New version found: {update.latest_version}")
                print(f"   Release notes: {update.release_notes_url}")
                print(f"   Download: {update.download_url}")
            else:
                print("✅ You are on the latest version.")
        else:
            print("❌ Update check failed.")

    asyncio.run(main())
