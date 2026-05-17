"""Send scanner alerts to Telegram (iPhone via Telegram app)."""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


def telegram_configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


def send_telegram(text: str, parse_mode: Optional[str] = None) -> bool:
    """
    Post a message via Bot API. Returns True on success.
    Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment (or .env loaded by run script).
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        logger.warning("Telegram skipped: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text[:4096]}
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200 and r.json().get("ok"):
            return True
        logger.warning("Telegram API error: %s %s", r.status_code, r.text[:300])
        return False
    except requests.RequestException as e:
        logger.warning("Telegram request failed: %s", e)
        return False


def format_matches_message(matches: List[dict], header: str) -> str:
    lines = [header, f"Total: {len(matches)}", ""]
    for m in matches[:15]:
        lines.append(
            f"{m['symbol']}\n"
            f"  1h RSI: {m['rsi_prev']} → {m['rsi_closed']}\n"
            f"  Live RSI: {m['indicator_live_rsi']} | 2d: {m.get('two_day_pct')}%"
            f" | ${m.get('current_price')}"
        )
    if len(matches) > 15:
        lines.append(f"\n… +{len(matches) - 15} more (see results.txt)")
    return "\n".join(lines)
