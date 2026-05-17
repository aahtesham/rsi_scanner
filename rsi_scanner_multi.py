import requests
import pandas as pd
import time
import logging
from ta.momentum import RSIIndicator
from concurrent.futures import ThreadPoolExecutor, as_completed
from time_util import scan_timestamp
from requests.adapters import HTTPAdapter


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
        # Step 1: fetch klines
        url = f"{B_API}/api/v3/klines?symbol={symbol}&interval=1h&limit=100"
        klines = session.get(url, timeout=10).json()

        if not klines or len(klines) < TWO_DAY_BARS + 20:
            return None

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close",
            "volume", "close_time", "qav", "trades",
            "taker_base", "taker_quote", "ignore"
        ])
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # Step 2: volume filter
        if df["volume"].tail(10).mean() < 50000:
            return None

        # Step 2b: last 2 days in uptrend (48h on 1h chart)
        up_ok, two_day_pct = two_day_uptrend_ok(df)
        if not up_ok:
            return None

        # Step 3: closed candle RSI (no live price yet)
        rsi        = RSIIndicator(df["close"], window=14).rsi()
        rsi_closed = rsi.iloc[-2]   # ✅ last fully closed candle
        rsi_prev   = rsi.iloc[-3]   # one before

        # Step 4: closed candles must show RSI rising but still below 53
        if not (rsi_prev < rsi_closed < 53):
            return None

        if (rsi_closed - rsi_prev) < 1:     # must be rising by at least 1 point
            return None

        # Step 5: NOW get live RSI values with current price injected
        manual_live_rsi, auto_live_rsi, current_price = get_live_rsi(symbol, klines)
        if auto_live_rsi is None:
            return None

        # Step 6: scan using RSIIndicator auto live RSI, not the manual SMA version
        if not (50 < auto_live_rsi <= 60):
            return None

        # Step 7: cooldown check
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
                logger.info(
                    f"MATCH: {result['symbol']} | "
                    f"1h: {result['rsi_prev']} → {result['rsi_closed']} | "
                    f"Manual Live RSI: {result['manual_live_rsi']} | "
                    f"RSIIndicator RSI: {result['indicator_live_rsi']} | "
                    f"2d%: {result['two_day_pct']} | "
                    f"Price: {result['current_price']}"
                )
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
    else:
        print("No matches found.\n")

# -------------------------------
# MAIN LOOP
# -------------------------------
if __name__ == "__main__":
    logger.info("Scanner started")
    while True:
        scan()
        logger.info("Sleeping 2.5 minutes...\n")
        time.sleep(150)