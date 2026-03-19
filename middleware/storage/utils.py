"""Utility functions for storage module."""

from datetime import datetime, timezone, timedelta


def get_east_8_time() -> datetime:
    """Get current time in East 8 timezone (UTC+8) without timezone info.

    Returns a naive datetime representing East 8 timezone time.
    """
    # Get current UTC time
    utc_now = datetime.now(timezone.utc)
    # Convert to East 8 (UTC+8)
    east_8 = timezone(timedelta(hours=8))
    east_8_now = utc_now.astimezone(east_8)
    # Return naive datetime (without timezone info)
    return east_8_now.replace(tzinfo=None)
