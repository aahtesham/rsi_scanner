"""Local timezone timestamps for logs and results (replaces UTC labels)."""

from datetime import datetime


def _local_dt() -> datetime:
    return datetime.now().astimezone()


def scan_timestamp() -> str:
    """e.g. 2026-05-14 21:30 (+03) — for === MATCHES === headers."""
    dt = _local_dt()
    base = dt.strftime("%Y-%m-%d %H:%M")
    tz = dt.strftime("%z")
    if tz:
        return f"{base} ({tz[:3]}:{tz[3:]})"
    return base


def trade_time_stamp() -> str:
    """e.g. 21:30 (+03) — for BUY/SELL lines."""
    dt = _local_dt()
    base = dt.strftime("%H:%M")
    tz = dt.strftime("%z")
    if tz:
        return f"{base} ({tz[:3]}:{tz[3:]})"
    return base
