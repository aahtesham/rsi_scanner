import requests
import pandas as pd
import time
import logging
from ta.momentum import RSIIndicator
from datetime import datetime, timezone


# -------------------------------
# LOGGING CONFIGURATION
# -------------------------------
logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for very detailed logs
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

# -------------------------------
# CONFIGURATION
# -------------------------------
B_API = "https://api.binance.com"
INTERVAL = "1h"
CANDLE_LIMIT = 100
session = requests.Session()


# -------------------------------
# Get all symbols
# -------------------------------
def get_all_usdt_symbols():
    logger.info("Fetching all symbols")

    url = f"{B_API}/api/v3/exchangeInfo"
    response = session.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()
    symbols = [
        s["symbol"]
        for s in data["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
    ]

    logger.info(f"Fetched {len(symbols)} USDT symbols")
    return symbols


# -------------------------------
# Fetch candlestick data
# -------------------------------
def get_klines(symbol):
    logger.debug(f"Fetching klines for {symbol}")

    url = f"{B_API}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": CANDLE_LIMIT
    }

    r = session.get(url, params=params, timeout=10)
    if r.status_code != 200:
        logger.warning(f"Kline request failed for {symbol}")
        return None

    return r.json()


# -------------------------------
# Calculate RSI(14)
# -------------------------------
def calculate_rsi(df):
    logger.debug("Calculating RSI(14)")
    indicator = RSIIndicator(df["close"], window=14)
    return indicator.rsi()

# -------------------------------
# Calculate Live RSI
# -------------------------------

def get_live_rsi(symbol, interval="1h", period=14):
    data = session.get(
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=100"
    ).json()

    closes = [float(x[4]) for x in data]

    # live price
    current_price = float(
        session.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}").json()["price"]
    )

    closes[-1] = current_price

    s = pd.Series(closes)
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return round(rsi.iloc[-1], 2)

# -------------------------------
# Main scan logic
# -------------------------------
def scan():
    logger.info("Starting scan cycle")
    symbols = get_all_usdt_symbols()
    matches = []

    for idx, symbol in enumerate(symbols, start=1):
        try:
            # ✅ STEP 1: Check live RSI first (fast filter)
            live_rsi = get_live_rsi(symbol)

            # ✅ STEP 2: Only proceed if live RSI is in the 54-57 zone
            if not (51 <= live_rsi <= 57):
                continue  # skip immediately, no heavy work

            logger.info(f"[{idx}] {symbol} live RSI={live_rsi} → checking 1h closed candles...")

            # ✅ STEP 3: Now do the heavy work
            klines = get_klines(symbol)
            if not klines or len(klines) < 20:
                continue

            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close",
                "volume", "close_time", "qav", "trades",
                "taker_base", "taker_quote", "ignore"
            ])
            df["close"] = df["close"].astype(float)

            rsi        = calculate_rsi(df)
            rsi_closed = rsi.iloc[-2]   # last fully closed 1h candle
            rsi_prev   = rsi.iloc[-3]   # one before that

            # ✅ STEP 4: Confirm 1h RSI was below 53 (crossing happened inside live candle)
            if rsi_closed < 53 and rsi_prev < rsi_closed:
                logger.info(f"MATCH: {symbol} | 1h RSI: {rsi_prev:.2f}→{rsi_closed:.2f} | Live: {live_rsi}")
                matches.append({
                    "symbol"    : symbol,
                    "rsi_prev"  : round(rsi_prev, 2),
                    "rsi_closed": round(rsi_closed, 2),
                    "live_rsi"  : round(live_rsi, 2),
                })

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    # Print results
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
            f.write(f"\n=== MATCHES Reverse Logic=== {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
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
# Run every 1 hour
# -------------------------------
if __name__ == "__main__":
    logger.info(" Scanner started")

    while True:
        logger.info("Running 1 Hour RSI scanner")
        scan()
        logger.info("Sleeping for 1 hour \n")
        time.sleep(300)
