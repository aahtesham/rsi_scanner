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

B_API   = "https://api.binance.com"
session = requests.Session()
adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
session.mount("https://", adapter)
session.mount("http://", adapter)

last_alert_time = {}
ALERT_COOLDOWN  = 1800  # 30 min

# ─────────────────────────────────────────
# Symbols
# ─────────────────────────────────────────
def get_all_usdt_symbols():
    url = f"{B_API}/api/v3/exchangeInfo"
    r = session.get(url, timeout=10)
    r.raise_for_status()
    return [
        s["symbol"]
        for s in r.json()["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
    ]

# ─────────────────────────────────────────
# Live RSI — inject ticker price
# ─────────────────────────────────────────
def get_live_rsi(symbol, klines, period=14):
    try:
        ticker = session.get(
            f"{B_API}/api/v3/ticker/price?symbol={symbol}", timeout=5
        ).json()
        current_price = float(ticker["price"])
    except Exception:
        return None, None, None

    closes     = [float(x[4]) for x in klines]
    closes[-1] = current_price

    s          = pd.Series(closes)
    delta      = s.diff()
    gain       = delta.where(delta > 0, 0).rolling(period).mean()
    loss       = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs         = gain / loss
    manual_rsi = 100 - (100 / (1 + rs))
    auto_rsi   = RSIIndicator(s, window=period).rsi()

    return (
        round(manual_rsi.iloc[-1], 2),
        round(auto_rsi.iloc[-1], 2),
        round(current_price, 8)
    )

# ─────────────────────────────────────────
# Process symbol
# ─────────────────────────────────────────
def process_symbol(symbol):
    try:
        klines = session.get(
            f"{B_API}/api/v3/klines?symbol={symbol}&interval=1h&limit=100",
            timeout=10
        ).json()

        if not klines or len(klines) < 55:
            return None

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close",
            "volume", "close_time", "qav", "trades",
            "taker_base", "taker_quote", "ignore"
        ])
        df["close"]  = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # ── 1. volume base filter ──
        avg_vol = df["volume"].tail(20).mean()
        if avg_vol < 50000:
            return None

        # ── 2. volume spike — real momentum ──
        last_vol = df["volume"].iloc[-2]
        if last_vol < avg_vol * 1.2:
            return None

        # ── 3. trend filter — above 50 SMA ──
        sma50 = df["close"].rolling(50).mean().iloc[-2]
        if df["close"].iloc[-2] < sma50:
            return None

        # ── 4. RSI closed candles ──
        rsi        = RSIIndicator(df["close"], window=14).rsi()
        rsi_closed = rsi.iloc[-2]
        rsi_prev   = rsi.iloc[-3]
        rsi_prev2  = rsi.iloc[-4]
        rsi_prev4  = rsi.iloc[-6]

        # ── 5. direction — rising and below 50 ──
        if not (rsi_prev < rsi_closed < 50):
            return None

        # ── 6. slope ──
        short_slope  = rsi_closed - rsi_prev
        long_slope   = rsi_closed - rsi_prev4
        acceleration = short_slope - (rsi_prev - rsi_prev2)

        if short_slope < 0.5:
            return None
        if long_slope < 2:
            return None
        if acceleration < 0:          # slope slowing down
            return None

        # ── 7. live RSI ──
        manual_live_rsi, auto_live_rsi, current_price = get_live_rsi(symbol, klines)
        if auto_live_rsi is None:
            return None

        if auto_live_rsi <= rsi_closed:
            return None

        if not (50 < auto_live_rsi <= 60):
            return None

        # ── 8. momentum cooldown ──
        if symbol in last_alert_rsi:
            rsi_moved     = abs(auto_live_rsi - last_alert_rsi[symbol]) >= 3
            slope_changed = abs(short_slope - last_alert_slope.get(symbol, 0)) >= 1
            if not rsi_moved and not slope_changed:
                return None

        last_alert_rsi[symbol]   = auto_live_rsi
        last_alert_slope[symbol] = short_slope

        # ── 9. score ──
        score = score_signal(
            rsi_prev, rsi_closed,
            short_slope, long_slope,
            auto_live_rsi, avg_vol
        )

        if score < 4:              # minimum quality threshold
            return None

        if auto_live_rsi <= 53:    zone = "just crossed 50"
        elif auto_live_rsi <= 56:  zone = "mid momentum"
        else:                      zone = "strong push"

        return {
            "symbol"          : symbol,
            "score"           : score,
            "rsi_prev"        : round(rsi_prev, 2),
            "rsi_closed"      : round(rsi_closed, 2),
            "short_slope"     : round(short_slope, 2),
            "long_slope"      : round(long_slope, 2),
            "acceleration"    : round(acceleration, 2),
            "live_rsi"        : round(auto_live_rsi, 2),
            "vol_ratio"       : round(last_vol / avg_vol, 2),
            "current_price"   : round(current_price, 8),
            "zone"            : zone,
        }

    except Exception as e:
        logger.error(f"{symbol} error: {e}")
        return None
# ─────────────────────────────────────────
# Scan
# ─────────────────────────────────────────
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
                    f"1h RSI: {result['rsi_prev']} → {result['rsi_closed']} | "
                    f"Slope: {result['short_slope']} / {result['long_slope']} | "
                    f"Live RSI: {result['live_rsi']} | "
                    f"Price: {result['current_price']} | {result['zone']}"
                )
                matches.append(result)

    print(f"\nScan done in {round(time.time() - start, 2)}s")

    if matches:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n=== MATCHES === {ts}")
        print("-" * 70)
        matches.sort(key=lambda x: x["score"], reverse=True)   # best first
        for m in matches:
             print(
                f"[{m['score']}/10] {m['symbol']:<12} | "
                f"RSI: {m['rsi_prev']} → {m['rsi_closed']} | "
                f"Slope: {m['short_slope']}/{m['long_slope']} accel={m['acceleration']} | "
                f"Vol: {m['vol_ratio']}x | "
                f"Live RSI: {m['live_rsi']} | {m['zone']}"
            )
        print("-" * 70)
        print(f"Total: {len(matches)}\n")

        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== MATCHES === {ts}\n")
            for m in matches:
                f.write(
                    f"[{m['score']}/10] {m['symbol']:<12} | "
                    f"RSI: {m['rsi_prev']} → {m['rsi_closed']} | "
                    f"Slope: {m['short_slope']}/{m['long_slope']} accel={m['acceleration']} | "
                    f"Live RSI: {m['live_rsi']} | "
                    f"Price: {m['current_price']} | {m['zone']}\n"
                )
            f.write(f"Total: {len(matches)}\n\n")
    else:
        print("No matches found.\n")

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Scanner started")
    while True:
        scan()
        logger.info("Sleeping 2.5 minutes...\n")
        time.sleep(150)