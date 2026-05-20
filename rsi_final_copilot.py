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

# -------------------------------
# Process each symbol
# -------------------------------
def process_symbol(symbol, trend_mode="AUTO"):
    try:
        # -------------------------------
        # Step 1: Fetch klines
        # -------------------------------
        url = f"{B_API}/api/v3/klines?symbol={symbol}&interval=1h&limit=250"
        klines = session.get(url, timeout=10).json()

        if not klines or len(klines) < 200:
            return None

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close",
            "volume", "close_time", "qav", "trades",
            "taker_base", "taker_quote", "ignore"
        ])

        for col in ["close", "high", "low", "volume", "taker_base"]:
            df[col] = df[col].astype(float)

        # -------------------------------
        # Step 2: Liquidity
        # -------------------------------
        df["quote_volume"] = df["volume"] * df["close"]

        # if df["quote_volume"].tail(10).mean() < 2_000_00:
        #     return None
        
         # ✅ better volume filter
        if df["quote_volume"].tail(10).mean() < 40000:
                return None


        # -------------------------------
        # Step 3: Buyer Dominance
        # -------------------------------
        recent_buy = df["taker_base"].iloc[-6:-2].sum()
        recent_vol = df["volume"].iloc[-6:-2].sum()

        if recent_vol == 0:
            return None

        buy_ratio = recent_buy / recent_vol
        if buy_ratio < 0.52:
            return None

        # -------------------------------
        # Step 4: RSI
        # -------------------------------
        rsi = RSIIndicator(df["close"], window=14).rsi()
        rsi_closed = rsi.iloc[-2]
        rsi_prev   = rsi.iloc[-3]

        short_slope = rsi_closed - rsi_prev
        long_slope  = rsi_closed - rsi.iloc[-5]

        # -------------------------------
        # Step 5: Trend Calculation
        # -------------------------------
        price = df["close"].iloc[-2]
        ma50  = df["close"].rolling(50).mean().iloc[-2]
        ma200 = df["close"].rolling(200).mean().iloc[-2]

        trend_strength = (ma50 - ma200) / ma200

        # -------------------------------
        # Step 6: AUTO MODE DETECTION
        # -------------------------------
        if trend_mode == "AUTO":
            if trend_strength > 0.05:
                mode = "HIGH"
            elif trend_strength < 0.01:
                mode = "LOW"
            else:
                mode = "BALANCED"
        else:
            mode = trend_mode

        # -------------------------------
        # Step 7: Structure Filter
        # -------------------------------
        recent_high = df["high"].iloc[-2]
        prev_high   = df["high"].iloc[-5]

        # -------------------------------
        # Step 8: Volatility
        # -------------------------------
        df["range"] = df["high"] - df["low"]
        avg_range = df["range"].rolling(14).mean().iloc[-2]

        # -------------------------------
        # Step 9: Live RSI
        # -------------------------------
        manual_live_rsi, auto_live_rsi, current_price = get_live_rsi(symbol, klines)
        if auto_live_rsi is None:
            return None

        # -------------------------------
        # ✅ Step 10: STRATEGY MODES
        # -------------------------------

        score = 0

        if buy_ratio > 0.55:
            score += 1
        if short_slope > 1:
            score += 1
        if long_slope > 3:
            score += 1
        if auto_live_rsi > 52:
            score += 1

        # -------------------------------
        # 🔴 HIGH MARKET (Strict / Breakout)
        # -------------------------------
        if mode == "HIGH":

            if not (price > ma50 > ma200):
                return None

            if recent_high <= prev_high:
                return None

            if auto_live_rsi <= rsi_closed:
                return None

            if (auto_live_rsi - rsi_closed) < 1.0:
                return None

            if score < 3:
                return None

        # -------------------------------
        # 🟢 LOW MARKET (1% Scalping)
        # -------------------------------
        elif mode == "LOW":

            if price < ma50:
                return None

            if not (40 <= rsi_closed <= 55):
                return None

            if rsi_closed <= rsi_prev:
                return None

            if short_slope < 0.5:
                return None

            if avg_range / price < 0.005:
                return None

            if score < 2:
                return None

        # -------------------------------
        # 🟡 BALANCED
        # -------------------------------
        else:

            if price < ma50:
                return None

            if not (42 <= rsi_closed <= 52):
                return None

            if rsi_closed <= rsi_prev:
                return None

            if short_slope < 0.8:
                return None

            if avg_range / price < 0.006:
                return None

            if recent_high <= prev_high * 0.99:
                return None

            if score < 2:
                return None

        # -------------------------------
        # Step 11: Cooldown
        # -------------------------------
        now = time.time()
        if symbol in last_alert_time:
            if now - last_alert_time[symbol] < ALERT_COOLDOWN:
                return None

        last_alert_time[symbol] = now

        # -------------------------------
        # ✅ FINAL RESULT
        # -------------------------------
            
        return {
            "symbol": symbol,
            "mode": mode,
            "price": round(current_price, 8),
            "rsi_prev": round(rsi_prev, 2),   # ✅ ADD THIS
            "rsi_closed": round(rsi_closed, 2),
            "manual_live_rsi": round(manual_live_rsi, 2),
            "live_rsi": round(auto_live_rsi, 2),
            "buy_ratio": round(buy_ratio, 3),
            "score": score,
            
            "volume": round(df["volume"].tail(10).mean(), 2),
            "quote_volume": round(df["quote_volume"].tail(10).mean(), 2)

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
                f"RSIIndicator RSI: {result['live_rsi']} | "
                f"Price: {result['price']} | Mode: {result['mode']} | Score: {result['score']} | "
                f"Volume: {result['volume']} | Quote Volume: {result['quote_volume']}"
                )
                matches.append(result)

    print(f"\nScan done in {round(time.time() - start, 2)}s")

    if matches:
        print(f"\n=== PRO LEVELS MATCHES === {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("-" * 55)
        for m in matches:
            print(
                f"{m['symbol']:<12} | "
                f"1h RSI: {m['rsi_prev']} → {m['rsi_closed']} | "
                f"Manual Live RSI: {m['manual_live_rsi']} | "
                f"RSIIndicator RSI: {m['live_rsi']} | "
                f"Price: {m['price']} | "
                f"Mode: {m['mode']} | Score: {m['score']} | "
                f"Volume: {m['volume']} | Quote Volume: {m['quote_volume']}"
            )

        print("-" * 55)
        print(f"Total: {len(matches)}\n")

        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== PRO LEVELS MATCHES === {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            for m in matches:
                f.write(
                    f"{m['symbol']:<12} | "
                    f"1h RSI: {m['rsi_prev']} → {m['rsi_closed']} | "
                    f"Manual Live RSI: {m['manual_live_rsi']} | "
                    f"RSIIndicator Live RSI: {m['live_rsi']} | "
                    f"Price: {m['price']} | Mode: {m['mode']} | Score: {m['score']} | "
                    f"Volume: {m['volume']} | Quote Volume: {m['quote_volume']}\n"
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

