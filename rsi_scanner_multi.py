import requests
import pandas as pd
import time
import logging
from ta.momentum import RSIIndicator
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ✅ increase connection pool
from requests.adapters import HTTPAdapter



# -------------------------------
# LOGGING CONFIGURATION
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# -------------------------------
# CONFIGURATION
# -------------------------------
B_API = "https://api.binance.com"
session = requests.Session()

adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
session.mount("https://", adapter)
session.mount("http://", adapter)


# cooldown tracking
last_alert_time = {}
ALERT_COOLDOWN = 1800  # 30 min

# -------------------------------
# Get all symbols
# -------------------------------
def get_all_usdt_symbols():
    url = f"{B_API}/api/v3/exchangeInfo"
    response = session.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()

    return [
        s["symbol"]
        for s in data["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
    ]

# -------------------------------
# RSI Calculation
# -------------------------------
def calculate_rsi(df):
    indicator = RSIIndicator(df["close"], window=14)
    return indicator.rsi()

# -------------------------------
# Process each symbol (THREAD)
# -------------------------------
def process_symbol(symbol):
    try:
        # ✅ SINGLE API CALL
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

        # ✅ volume filter
        if df["volume"].tail(10).mean() < 50000:
            return None

        rsi = calculate_rsi(df)

        live_rsi   = rsi.iloc[-1]
        rsi_closed = rsi.iloc[-2]
        rsi_prev   = rsi.iloc[-3]

        # ✅ RSI filters
        if not (53 <= live_rsi <= 60):
            return None

        if (rsi_closed - rsi_prev) < 1:
            return None

        if not (rsi_prev < rsi_closed < 52):
            return None

        # ✅ cooldown control
        current_time = time.time()
        if symbol in last_alert_time:
            if current_time - last_alert_time[symbol] < ALERT_COOLDOWN:
                return None

        last_alert_time[symbol] = current_time

        return {
            "symbol": symbol,
            "rsi_prev": round(rsi_prev, 2),
            "rsi_closed": round(rsi_closed, 2),
            "live_rsi": round(live_rsi, 2),
        }

    except Exception as e:
        logger.error(f"{symbol} error: {e}")

    return None

# -------------------------------
# Main Scan Function
# -------------------------------
def scan():
    logger.info("Starting scan cycle")

    symbols = get_all_usdt_symbols()
    matches = []

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_symbol, symbol) for symbol in symbols]

        for future in as_completed(futures):
            result = future.result()

            if result:
                logger.info(
                    f"MATCH: {result['symbol']} | "
                    f"{result['rsi_prev']} → {result['rsi_closed']} | "
                    f"Live: {result['live_rsi']}"
                )
                matches.append(result)

    end_time = time.time()

    print(f"\nScan completed in {round(end_time - start_time, 2)} seconds")

    # -------------------------------
    # Print Results
    # -------------------------------
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
        logger.info("Sleeping for 2.5 minutes...\n")
        time.sleep(150)