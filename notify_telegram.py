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


def format_matches_message(matches, title="RSI Matches"):
    lines = [f"🔔 *{title}*\n"]
    
    for m in matches:
        live_rsi   = m.get("indicator_live_rsi", m.get("live_rsi"))
        mode       = m.get("mode", "")
        score      = m.get("score", "")
        volume     = m.get("volume", "")
        quote_vol  = m.get("quote_volume", "")

        mode_part    = f"Mode: {mode} | "    if mode              else ""
        score_part   = f"Score: {score} | "  if score             else ""
        vol_part     = f"Vol: {volume} | "   if volume            else ""
        qvol_part    = f"QVol: {quote_vol}"  if quote_vol         else ""

        lines.append(
            f"📊 *{m['symbol']}* | {mode_part} | {score_part}\n"
            f"  RSI: `{m['rsi_prev']} → {m['rsi_closed']}` | Live: `{live_rsi}`\n"
            f"  Price: `{m.get('price')}`\n"
            f"   {vol_part} | {qvol_part}"
        )

    
    return "\n\n".join(lines)