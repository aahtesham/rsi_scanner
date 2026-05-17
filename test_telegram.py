#!/usr/bin/env python3
"""One-shot test: .venv/bin/python test_telegram.py  (reads .env in project folder)"""

import os
from pathlib import Path


def load_dotenv_simple(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    env_path = root / ".env"
    load_dotenv_simple(env_path)

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not env_path.is_file():
        print(f"FAILED — no .env file at:\n  {env_path}\n")
        print("Create it:  cp .env.example .env")
        print("Then edit .env and set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (numeric id, not @botname).")
        raise SystemExit(1)

    if not token or not chat_id:
        print("FAILED — .env exists but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is empty.")
        raise SystemExit(1)

    if not chat_id.lstrip("-").isdigit():
        print(
            f"FAILED — TELEGRAM_CHAT_ID must be a number (e.g. 123456789), not '{chat_id}'.\n"
            "Get it: message your bot, then open getUpdates in the browser (see Telegram setup steps)."
        )
        raise SystemExit(1)

    from notify_telegram import send_telegram

    ok = send_telegram("RSI scanner test — if you see this on iPhone, Telegram is wired correctly.")
    print("OK" if ok else "FAILED — check token (revoke if leaked), chat id, and that you messaged the bot once")
