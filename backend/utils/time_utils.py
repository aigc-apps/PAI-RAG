import os
from datetime import datetime


def get_current_date_str() -> str:
    """Return today's date in the process-local timezone."""
    return datetime.now().astimezone().date().isoformat()


def get_local_timezone_name() -> str:
    """Return a stable configured timezone name, falling back to its abbreviation."""
    configured = os.environ.get("TZ", "").strip()
    if configured:
        return configured
    return str(datetime.now().astimezone().tzinfo or "UTC")


def get_current_time_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


def get_current_time_str_zh() -> str:
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S %A")
