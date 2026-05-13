import requests
import pandas as pd
import time
from ta.momentum import RSIIndicator
from datetime import datetime

BINANCE_API = "https://api.binance.com"
INTERVAL = "1h"
CANDLE_LIMIT = 100   # enough for RSI smoothing
OUTPUT_FILE = "rsi_52_55_scanner.xlsx"


# -------------------------------
# Get all USDT trading symbols
# -------------------------------
def get_all_usdt_symbols():
    url = f"{BINANCE_API}/api/v3/exchangeInfo"
    data = requests.get(url).json()
    return [
        s["symbol"]
        for s in data["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
    ]


# -------------------------------
# Fetch candles
# -------------------------------
def get_klines(symbol):
    url = f"{BINANCE_API}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": CANDLE_LIMIT
    }
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return None
    return r.json()


# -------------------------------
# Calculate RSI(14)
# -------------------------------
def calculate_rsi(df):
    rsi = RSIIndicator(df["close"], window=14).rsi()
    return rsi


# -------------------------------
# Main scan logic
# -------------------------------
def scan():
    symbols = get_all_usdt_symbols()
    results = []

    for symbol in symbols:
        try:
            klines = get_klines(symbol)
            if not klines or len(klines) < 20:
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

            rsi_now = rsi.iloc[-1]
            rsi_1 = rsi.iloc[-2]
            rsi_2 = rsi.iloc[-3]

            # ✅ RSI rising & heading toward 55
            if (
                rsi_now > rsi_1 > rsi_2 and
                52 <= rsi_now < 55
            ):
                results.append({
                    "Symbol": symbol,
                    "RSI_1H": round(rsi_now, 2),
                    "ScanTime": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
                })

        except Exception:
            continue

    return pd.DataFrame(results)


# -------------------------------
# Auto‑run every 1 hour
# -------------------------------
while True:
    print("Running 1H RSI scanner...")
    df_result = scan()

    if not df_result.empty:
        df_result.to_excel(OUTPUT_FILE, index=False)
        print(f"Found {len(df_result)} tokens → saved to {OUTPUT_FILE}")
    else:
        print("No matching tokens this hour.")

    # ✅ wait exactly 1 hour
    time.sleep(3600)