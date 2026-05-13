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
        logger.debug(f"[{idx}/{len(symbols)}] Processing {symbol}")

        try:
            klines = get_klines(symbol)
            if not klines or len(klines) < 20:
                logger.debug(f"Skipping {symbol} (insufficient data)")
                continue

            df = pd.DataFrame(
                klines,
                columns=[
                    "open_time", "open", "high", "low", "close",
                    "volume", "close_time", "qav", "trades",
                    "taker_base", "taker_quote", "ignore"
                ]
            )
            df["close"] = df["close"].astype(float)

            rsi = calculate_rsi(df)

            #rsi_now = rsi.iloc[-1]
            #rsi_prev = rsi.iloc[-2]
            #rsi_prev2 = rsi.iloc[-3]

            rsi_now   = rsi.iloc[-2]   # last CLOSED  ✅
            rsi_prev  = rsi.iloc[-3]   # one before it
            rsi_prev2 = rsi.iloc[-4]   # two before it


            logger.debug(
                f"{symbol} RSI values → "
                f"prev2={rsi_prev2:.2f}, prev1={rsi_prev:.2f}, now={rsi_now:.2f}"
            )

            # ✅ RSI rising & heading toward 57
            #if rsi_now > rsi_prev > rsi_prev2 and 54 <= rsi_now <= 55:
            if rsi_prev2 < rsi_prev < rsi_now and rsi_prev < 53 < rsi_now:

                logger.info(f"MATCH FOUND: {symbol} RSI={rsi_now:.2f}")
                # matches.append({
                #     "symbol": symbol,
                #     "rsi": round(rsi_now, 2)
                # })
                matches.append({
                    "symbol"  : symbol,
                    "rsi"     : round(rsi_now, 2),
                    "rsi_prev": round(rsi_prev, 2),
                })

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    # -------------------------------
    # Print results to console and file
    # -------------------------------
    logger.info("Scan cycle completed")

    if matches:
        print("\n=== RSI 53 →  MATCHES ===")
        print(f"Scan Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print("-" * 45)
        
        for m in matches:
            m["live_rsi"] = get_live_rsi(m["symbol"])

        for m in matches:
            print(f"{m['symbol']:<10} | RSI: {m['rsi']} | Live RSI: {m['live_rsi']}\n")

        print("-" * 45)
        print(f"Total Matches: {len(matches)}\n")

        # Write to file
        
        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== RSI 53 →  MATCHES ===\n")
            f.write(f"Scan Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
            f.write("-" * 45 + "\n")

            for m in matches:
                f.write(f"{m['symbol']:<10} | RSI: {m['rsi']} | Live RSI: {m['live_rsi']}\n")

            f.write("-" * 45 + "\n")
            f.write(f"Total Matches: {len(matches)}\n\n")
            f.write("=======\n\n")

    else:
        print("No matching tokens found.\n")
        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"No matching tokens found at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")


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
