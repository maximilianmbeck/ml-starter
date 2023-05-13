"""Utilities for working with datetimes."""

import datetime


def format_timedelta(delta: datetime.timedelta) -> str:
    """Formats a time delta to human-readable format.

    Args:
        delta: The time delta to format

    Returns:
        The human-readable time delta
    """
    parts = []
    if delta.days > 0:
        parts += [f"{delta.days} day" if delta.days == 1 else f"{delta.days} days"]

    seconds = delta.seconds

    if seconds > 60 * 60:
        hours, seconds = seconds // (60 * 60), seconds % (60 * 60)
        parts += [f"{hours} hour" if hours == 1 else f"{hours} hours"]

    if seconds > 60:
        minutes, seconds = seconds // 60, seconds % 60
        parts += [f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"]

    parts += [f"{seconds} second" if seconds == 1 else f"{seconds} seconds"]

    return ", ".join(parts)


def format_datetime(dt: datetime.datetime) -> str:
    """Formats a datetime to human-readable format.

    Args:
        dt: The datetime to format

    Returns:
        The human-readable datetime
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")
