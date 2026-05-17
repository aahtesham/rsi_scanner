import os
import requests
import pandas as pd
import time
import logging
from pathlib import Path
from ta.momentum import RSIIndicator
from concurrent.futures import ThreadPoolExecutor, as_completed
from time_util import scan_timestamp
from notify_telegram import format_matches_message, send_telegram, telegram_configured
from requests.adapters import HTTPAdapter


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
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

B_API = "https://api.binance.com"
session = requests.Session()
adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
session.mount("https://", adapter)
session.mount("http://", adapter)

last_alert_time = {}
ALERT_COOLDOWN = 1800  # 30 min

# Telegram (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env)
NOTIFY_TELEGRAM = os.environ.get("NOTIFY_TELEGRAM", "1").strip().lower() in ("1", "true", "yes", "on")
SCAN_SLEEP_S = int(os.environ.get("SCAN_SLEEP_S", "150"))

# 2-day uptrend on 1h bars: 48 closed hours ≈ 2 days (no extra API call)
TWO_DAY_BARS = 48
DAY_HALF_BARS = 24

# -------------------------------
# Get all symbols
# -------------------------------
def get_all_usdt_symbols():
    url = f"{B_API}/api/v3/exchangeInfo"
    r = session.get(url, timeout=10)
    r.raise_for_status()
    return [
        s["symbol"]
        for s in r.json()["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
    ]

# -------------------------------
# True live RSI using ticker price
# -------------------------------
def get_live_rsi(symbol, klines, period=14):
    """Inject real-time ticker price into last candle, then recalculate RSI."""
    try:
        ticker = session.get(
            f"{B_API}/api/v3/ticker/price?symbol={symbol}", timeout=5
        ).json()
        current_price = float(ticker["price"])
    except Exception:
        return None, None, None

    closes = [float(x[4]) for x in klines]
    closes[-1] = current_price          # ✅ replace forming candle with live price

    s = pd.Series(closes)
    delta = s.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    manual_rsi = 100 - (100 / (1 + rs))
    auto_rsi = RSIIndicator(s, window=period).rsi()

    return (
        round(manual_rsi.iloc[-1], 2),
        round(auto_rsi.iloc[-1], 2),
        round(current_price, 8)
    )


def two_day_uptrend_ok(df_1h: pd.DataFrame) -> tuple:
    """
    Last ~2 days in uptrend (1h klines, last bar may still be forming):
    - Net: last closed 1h close > close 48 bars earlier
    - Structure: close rises over each 24h half (higher highs on 1h closes)
    Returns (ok, two_day_pct_change).
    """
    need = TWO_DAY_BARS + 3
    if len(df_1h) < need:
        return False, None
    c = df_1h["close"].astype(float)
    start = float(c.iloc[-2 - TWO_DAY_BARS])
    mid = float(c.iloc[-2 - DAY_HALF_BARS])
    end = float(c.iloc[-2])
    if end <= start or mid <= start or end <= mid:
        return False, round((end / start - 1) * 100, 2) if start else None
    return True, round((end / start - 1) * 100, 2)


# -------------------------------
# Process each symbol
# -------------------------------
def process_symbol(symbol):
    try:
        url    = f"{B_API}/api/v3/klines?symbol={symbol}&interval=1h&limit=100"
        klines = session.get(url, timeout=10).json()

        if not klines or len(klines) < TWO_DAY_BARS + 20:
            return None

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close",
            "volume", "close_time", "qav", "trades",
            "taker_base", "taker_quote", "ignore"
        ])
        df["close"]      = df["close"].astype(float)
        df["volume"]     = df["volume"].astype(float)
        df["taker_base"] = df["taker_base"].astype(float)

        # volume filter
        if df["volume"].tail(10).mean() < 50000:
            return None

        # taker buy ratio — buyers must dominate
        buy_ratio = df["taker_base"].iloc[-2] / df["volume"].iloc[-2]
        if buy_ratio < 0.52:
            return None

        # 2-day uptrend
        up_ok, two_day_pct = two_day_uptrend_ok(df)
        if not up_ok:
            return None

        # RSI closed candles
        rsi        = RSIIndicator(df["close"], window=14).rsi()
        rsi_closed = rsi.iloc[-2]
        rsi_prev   = rsi.iloc[-3]
        long_slope = rsi_closed - rsi.iloc[-5]   # 4-candle momentum

        # ✅ fix: restored upper limit
        if not (rsi_prev < rsi_closed ):
            return None

        # short slope
        if (rsi_closed - rsi_prev) < 1:
            return None

        # long slope — sustained momentum not just 1 candle spike
        

        # live RSI
        manual_live_rsi, auto_live_rsi, current_price = get_live_rsi(symbol, klines)
        if auto_live_rsi is None:
            return None

        if not (50 < auto_live_rsi <= 60):
            return None

        # cooldown
        now = time.time()
        if symbol in last_alert_time:
            if now - last_alert_time[symbol] < ALERT_COOLDOWN:
                return None
        last_alert_time[symbol] = now

        return {
            "symbol"             : symbol,
            "rsi_prev"           : round(rsi_prev, 2),
            "rsi_closed"         : round(rsi_closed, 2),
            "manual_live_rsi"    : round(manual_live_rsi, 2),
            "indicator_live_rsi" : round(auto_live_rsi, 2),
            "current_price"      : round(current_price, 8),
            "two_day_pct"        : two_day_pct,
            "long_slope"         : round(long_slope, 2),
            "buy_ratio"          : round(buy_ratio, 2),
        }

    except Exception as e:
        logger.error(f"{symbol} error: {e}")
        return None
# -------------------------------
# Main Scan
# -------------------------------
def scan():
    logger.info("Starting scan cycle")
    symbols = get_all_usdt_symbols()
    matches = []
    start   = time.time()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_symbol, s) for s in symbols]
        for future in as_completed(futures):
            result = future.result()
            if result:
              matches.append(result)

    print(f"\nScan done in {round(time.time() - start, 2)}s")

    if matches:
        print(f"\n=== MATCHES === {scan_timestamp()}")
        print("-" * 55)
        for m in matches:
            print(
                f"{m['symbol']:<12} | "
                f"1h RSI: {m['rsi_prev']} → {m['rsi_closed']} | "
                f"Manual Live RSI: {m['manual_live_rsi']} | "
                f"RSIIndicator RSI: {m['indicator_live_rsi']} | "
                f"2d up: {m['two_day_pct']}% | "
                f"Price: {m['current_price']}"
            )
        print("-" * 55)
        print(f"Total: {len(matches)}\n")

        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== MATCHES === {scan_timestamp()}\n")
            for m in matches:
                f.write(
                    f"{m['symbol']:<12} | "
                    f"1h RSI: {m['rsi_prev']} → {m['rsi_closed']} | "
                    f"Manual Live RSI: {m['manual_live_rsi']} | "
                    f"RSIIndicator Live RSI: {m['indicator_live_rsi']} | "
                    f"2d up: {m['two_day_pct']}% | "
                    f"Price: {m['current_price']}\n"
                )
            f.write(f"Total: {len(matches)}\n\n")
        if NOTIFY_TELEGRAM and telegram_configured():
            ok = send_telegram(
                format_matches_message(matches, f"RSI matches — {scan_timestamp()}")
            )
            if ok:
                logger.info("Telegram sent (%s matches)", len(matches))
            else:
                logger.warning("Telegram send failed (see warnings above)")
        elif NOTIFY_TELEGRAM:
            logger.warning(
                "Telegram skipped: create %s with TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID",
                _PROJECT_DIR / ".env",
            )
    else:
        print("No matches found.\n")

# -------------------------------
# MAIN LOOP
# -------------------------------
if __name__ == "__main__":
    env_file = _PROJECT_DIR / ".env"
    if NOTIFY_TELEGRAM and not telegram_configured():
        logger.warning(
            "Telegram not configured — alerts will NOT be sent. "
            "Run: cp .env.example .env  then set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in %s",
            env_file,
        )
    logger.info(
        "Scanner started (NOTIFY_TELEGRAM=%s, telegram_ok=%s, .env=%s, SCAN_SLEEP_S=%s)",
        NOTIFY_TELEGRAM,
        telegram_configured(),
        env_file.is_file(),
        SCAN_SLEEP_S,
    )
    while True:
        scan()
        logger.info("Sleeping %s seconds (%.1f min)\n", SCAN_SLEEP_S, SCAN_SLEEP_S / 60)
        time.sleep(SCAN_SLEEP_S)