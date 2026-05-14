import requests
import pandas as pd
import time
import logging
from ta.momentum import RSIIndicator
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        return None

    closes = [float(x[4]) for x in klines]
    closes[-1] = current_price          # ✅ replace forming candle with live price

    s = pd.Series(closes)
    delta = s.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

# -------------------------------
# Process each symbol
# -------------------------------
def process_symbol(symbol):
    try:
        # Step 1: fetch klines
        url = f"{B_API}/api/v3/klines?symbol={symbol}&interval=1h&limit=100"
        klines = session.get(url, timeout=10).json()

        if not klines or len(klines) < 20:
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

        # Step 3: closed candle RSI (no live price yet)
        rsi        = RSIIndicator(df["close"], window=14).rsi()
        rsi_closed = rsi.iloc[-2]   # ✅ last fully closed candle
        rsi_prev   = rsi.iloc[-3]   # one before

        # Step 4: closed candles must show RSI rising but still below 53
        if not (rsi_prev < rsi_closed < 53):
            return None

        if (rsi_closed - rsi_prev) < 1:     # must be rising by at least 1 point
            return None

        # Step 5: NOW get true live RSI (ticker price injected)
        live_rsi = get_live_rsi(symbol, klines)
        if live_rsi is None:
            return None

        # Step 6: live RSI must have crossed above 53 into 53-60 zone
        if not (53 < live_rsi <= 60):
            return None

        # Step 7: cooldown check
        now = time.time()
        if symbol in last_alert_time:
            if now - last_alert_time[symbol] < ALERT_COOLDOWN:
                return None
        last_alert_time[symbol] = now

        return {
            "symbol"    : symbol,
            "rsi_prev"  : round(rsi_prev, 2),
            "rsi_closed": round(rsi_closed, 2),
            "live_rsi"  : round(live_rsi, 2),
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
                    f"Live: {result['live_rsi']}"
                )
                matches.append(result)

    print(f"\nScan done in {round(time.time() - start, 2)}s")

    if matches:
        print(f"\n=== MATCHES === {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print("-" * 55)
        for m in matches:
            print(
                f"{m['symbol']:<12} | "
                f"1h RSI: {m['rsi_prev']} → {m['rsi_closed']} | "
                f"Live RSI: {m['live_rsi']}"
            )
        print("-" * 55)
        print(f"Total: {len(matches)}\n")

        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== MATCHES === {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
            for m in matches:
                f.write(
                    f"{m['symbol']:<12} | "
                    f"1h RSI: {m['rsi_prev']} → {m['rsi_closed']} | "
                    f"Live RSI: {m['live_rsi']}\n"
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