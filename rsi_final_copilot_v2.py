"""
RSI Scanner V2 — Early Catch-Up / Early Tide only.

Run alongside rsi_final_copilot.py (V1 = HIGH / BALANCED / LOW).
V2 targets the first 2–3 green 1h candles + volume spike + breakout,
before RSI goes parabolic (e.g. avoid EDEN-style alerts at RSI 73+).

  python rsi_final_copilot_v2.py
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from ta.momentum import RSIIndicator

from notify_telegram import format_matches_message, send_telegram, telegram_configured
from time_util import scan_timestamp

MODE_EARLY = "EARLY_CATCHUP"


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_PROJECT_DIR = Path(__file__).resolve().parent
_load_dotenv(_PROJECT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

B_API = "https://api.binance.com"
session = requests.Session()
adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
session.mount("https://", adapter)
session.mount("http://", adapter)

last_alert_time: dict[str, float] = {}
_alert_lock = threading.Lock()
ALERT_COOLDOWN = int(os.environ.get("ALERT_COOLDOWN", "1800"))

NOTIFY_TELEGRAM = os.environ.get("NOTIFY_TELEGRAM", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# V2 defaults to faster scans (override in .env)
SCAN_SLEEP_S = int(os.environ.get("SCAN_SLEEP_S_V2", os.environ.get("SCAN_SLEEP_S", "90")))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def get_all_usdt_symbols() -> list[str]:
    url = f"{B_API}/api/v3/exchangeInfo"
    r = session.get(url, timeout=10)
    r.raise_for_status()
    return [
        s["symbol"]
        for s in r.json()["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
    ]


def get_live_rsi(symbol: str, klines: list, period: int = 14):
    """Inject ticker into forming candle; return (manual_rsi, ta_rsi, price)."""
    try:
        ticker = session.get(
            f"{B_API}/api/v3/ticker/price?symbol={symbol}", timeout=5
        ).json()
        current_price = float(ticker["price"])
    except Exception:
        return None, None, None

    closes = [float(x[4]) for x in klines]
    closes[-1] = current_price
    s = pd.Series(closes)
    auto_rsi = RSIIndicator(s, window=period).rsi()
    return (
        round(float(auto_rsi.iloc[-1]), 2),
        round(float(auto_rsi.iloc[-1]), 2),
        round(current_price, 8),
    )


def quote_volume_spike(df: pd.DataFrame) -> bool:
    mult = _env_float("EARLY_VOL_SPIKE_MULT", 1.45)
    avg_qvol = float(df["quote_volume"].iloc[-12:-2].mean())
    if avg_qvol <= 0:
        return False
    for idx in (-1, -2):
        if float(df["quote_volume"].iloc[idx]) >= mult * avg_qvol:
            return True
    return False


def buy_pressure_ok(df: pd.DataFrame, buy_ratio: float) -> bool:
    min_buy = _env_float("EARLY_MIN_BUY_RATIO", 0.53)
    for idx in (-1, -2):
        vol = float(df["volume"].iloc[idx])
        if vol > 0 and float(df["taker_base"].iloc[idx]) / vol >= min_buy:
            return True
    return buy_ratio >= min_buy


def green_candle_streak(df: pd.DataFrame, min_green: int = 2) -> bool:
    tail = df.iloc[-3:]
    return int((tail["close"] > tail["open"]).sum()) >= min_green


def is_late_rsi(rsi_closed: float, auto_live_rsi: float) -> bool:
    max_closed = _env_float("EARLY_MAX_RSI_CLOSED", 66)
    max_live = _env_float("EARLY_MAX_RSI_LIVE", 68)
    return rsi_closed > max_closed or auto_live_rsi > max_live


def passes_early_catchup(
    df: pd.DataFrame,
    *,
    rsi_closed: float,
    rsi_prev: float,
    rsi_series: pd.Series,
    auto_live_rsi: float,
    current_price: float,
    buy_ratio: float,
    short_slope: float,
    score: int,
) -> bool:
    if is_late_rsi(rsi_closed, auto_live_rsi):
        return False

    min_qvol = _env_float("EARLY_MIN_QUOTE_VOL", 20000)
    if not quote_volume_spike(df) and float(df["quote_volume"].tail(10).mean()) < min_qvol:
        return False

    if not buy_pressure_ok(df, buy_ratio):
        return False

    lookback = int(os.environ.get("EARLY_BREAKOUT_LOOKBACK", "20"))
    prior_high = float(df["high"].iloc[-lookback:-2].max())
    breakout_pct = _env_float("EARLY_BREAKOUT_PCT", 0.001)
    if current_price <= prior_high * (1 + breakout_pct):
        return False

    min_live = _env_float("EARLY_MIN_RSI_LIVE", 45)
    if auto_live_rsi < min_live:
        return False

    live_push = auto_live_rsi - rsi_closed
    if live_push < 0.3 and short_slope < 0.4:
        return False

    try:
        if float(rsi_series.iloc[-5]) > 62:
            return False
    except (IndexError, TypeError, ValueError):
        pass

    if rsi_closed <= rsi_prev and short_slope < 0.3:
        return False

    if not green_candle_streak(df, min_green=2):
        return False

    min_score = 1 if quote_volume_spike(df) else 2
    if score < min_score:
        return False

    return True


def process_symbol(symbol: str):
    try:
        url = f"{B_API}/api/v3/klines?symbol={symbol}&interval=1h&limit=120"
        klines = session.get(url, timeout=10).json()

        if not klines or len(klines) < 30:
            return None

        df = pd.DataFrame(
            klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "qav",
                "trades",
                "taker_base",
                "taker_quote",
                "ignore",
            ],
        )
        for col in ["open", "close", "high", "low", "volume", "taker_base"]:
            df[col] = df[col].astype(float)

        df["quote_volume"] = df["volume"] * df["close"]

        rsi = RSIIndicator(df["close"], window=14).rsi()
        rsi_closed = float(rsi.iloc[-2])
        rsi_prev = float(rsi.iloc[-3])
        short_slope = rsi_closed - rsi_prev
        long_slope = rsi_closed - float(rsi.iloc[-5])

        recent_buy = df["taker_base"].iloc[-6:-2].sum()
        recent_vol = df["volume"].iloc[-6:-2].sum()
        if recent_vol == 0:
            return None
        buy_ratio = float(recent_buy / recent_vol)

        _, auto_live_rsi, current_price = get_live_rsi(symbol, klines)
        if auto_live_rsi is None:
            return None

        score = 0
        if buy_ratio > 0.55:
            score += 1
        if short_slope > 1:
            score += 1
        if long_slope > 3:
            score += 1
        if auto_live_rsi > 52:
            score += 1

        if not passes_early_catchup(
            df,
            rsi_closed=rsi_closed,
            rsi_prev=rsi_prev,
            rsi_series=rsi,
            auto_live_rsi=auto_live_rsi,
            current_price=current_price,
            buy_ratio=buy_ratio,
            short_slope=short_slope,
            score=score,
        ):
            return None

        now = time.time()
        with _alert_lock:
            if symbol in last_alert_time:
                if now - last_alert_time[symbol] < ALERT_COOLDOWN:
                    return None
            last_alert_time[symbol] = now

        vol_spike = quote_volume_spike(df)
        return {
            "symbol": symbol,
            "mode": MODE_EARLY,
            "price": current_price,
            "rsi_prev": round(rsi_prev, 2),
            "rsi_closed": round(rsi_closed, 2),
            "live_rsi": auto_live_rsi,
            "indicator_live_rsi": auto_live_rsi,
            "buy_ratio": round(buy_ratio, 3),
            "score": score,
            "vol_spike": vol_spike,
            "quote_volume": round(float(df["quote_volume"].tail(10).mean()), 2),
            "volume": round(float(df["volume"].tail(10).mean()), 2),
        }

    except Exception as e:
        logger.error("%s error: %s", symbol, e)
        return None


def scan():
    logger.info("V2 Early Catch-Up scan starting")
    symbols = get_all_usdt_symbols()
    matches = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_symbol, s) for s in symbols]
        for future in as_completed(futures):
            result = future.result()
            if result:
                logger.info(
                    "EARLY: %s | RSI %s → %s | live=%s | score=%s | vol_spike=%s",
                    result["symbol"],
                    result["rsi_prev"],
                    result["rsi_closed"],
                    result["live_rsi"],
                    result["score"],
                    result["vol_spike"],
                )
                matches.append(result)

    elapsed = round(time.time() - start, 2)
    print(f"\nV2 scan done in {elapsed}s")

    if not matches:
        print("No early catch-up matches.\n")
        return

    ts = scan_timestamp()
    print(f"\n=== EARLY CATCH-UP (V2) === {ts}")
    print("-" * 55)
    for m in matches:
        print(
            f"{m['symbol']:<12} | "
            f"RSI: {m['rsi_prev']} → {m['rsi_closed']} | Live: {m['live_rsi']} | "
            f"Price: {m['price']} | Score: {m['score']} | "
            f"VolSpike: {m['vol_spike']} | QVol(10h): {m['quote_volume']}"
        )
    print("-" * 55)
    print(f"Total: {len(matches)}\n")

    results_path = _PROJECT_DIR / "results_v2_early.txt"
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== EARLY CATCH-UP V2 === {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        for m in matches:
            f.write(
                f"{m['symbol']:<12} | RSI: {m['rsi_prev']} → {m['rsi_closed']} | "
                f"Live: {m['live_rsi']} | Price: {m['price']} | "
                f"Score: {m['score']} | VolSpike: {m['vol_spike']}\n"
            )
        f.write(f"Total: {len(matches)}\n\n")

    if NOTIFY_TELEGRAM and telegram_configured():
        ok = send_telegram(
            format_matches_message(matches, f"⚡ Early Tide (V2) — {ts}")
        )
        if ok:
            logger.info("Telegram sent (%s early matches)", len(matches))
        else:
            logger.warning("Telegram send failed")
    elif NOTIFY_TELEGRAM:
        logger.warning(
            "Telegram skipped: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in %s",
            _PROJECT_DIR / ".env",
        )


if __name__ == "__main__":
    env_file = _PROJECT_DIR / ".env"
    if NOTIFY_TELEGRAM and not telegram_configured():
        logger.warning(
            "Telegram not configured — copy .env.example → .env and set bot token + chat id (%s)",
            env_file,
        )
    logger.info(
        "V2 Early Catch-Up scanner | SCAN_SLEEP_S=%s | results=%s",
        SCAN_SLEEP_S,
        _PROJECT_DIR / "results_v2_early.txt",
    )
    while True:
        scan()
        logger.info("Sleeping %ss (%.1f min)\n", SCAN_SLEEP_S, SCAN_SLEEP_S / 60)
        time.sleep(SCAN_SLEEP_S)
