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

last_alert_rsi = {}
RSI_CHANGE_THRESHOLD = 3

LB_LEFT     = 5
LB_RIGHT    = 5
RANGE_LOWER = 5
RANGE_UPPER = 60

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
# 500 candles
# ─────────────────────────────────────────
def get_klines_500(symbol):
    r = session.get(
        f"{B_API}/api/v3/klines",
        params={"symbol": symbol, "interval": "1h", "limit": 500},
        timeout=10
    )
    return r.json() if r.status_code == 200 else None

# ─────────────────────────────────────────
# Live RSI
# ─────────────────────────────────────────
def get_live_rsi(symbol, klines, period=14):
    try:
        ticker = session.get(
            f"{B_API}/api/v3/ticker/price?symbol={symbol}", timeout=5
        ).json()
        current_price = float(ticker["price"])
    except Exception:
        return None, None

    closes     = [float(x[4]) for x in klines]
    closes[-1] = current_price
    s          = pd.Series(closes)
    live_rsi   = RSIIndicator(s, window=period).rsi()
    return round(live_rsi.iloc[-1], 2), round(current_price, 8)

# ─────────────────────────────────────────
# Pivot Low on RSI
# Mirrors: ta.pivotlow(osc, lbL, lbR)
# A pivot at index i means:
#   rsi[i] < all lbL bars to the LEFT
#   rsi[i] < all lbR bars to the RIGHT
# Then we READ price LOW at that same index i
# ─────────────────────────────────────────
def find_rsi_pivot_lows(rsi_vals: list, price_lows: list,
                         lb_left=5, lb_right=5):
    """
    Find pivot lows on RSI.
    At each pivot index, record BOTH the RSI value AND the price low.
    This is exactly what TradingView does:
      plFound = pivot on RSI
      osc[lbR] = RSI value at that pivot
      low[lbR] = price LOW at that same pivot bar
    """
    pivots = []
    n = len(rsi_vals)

    for i in range(lb_left, n - lb_right):
        rsi_val = rsi_vals[i]

        # RSI must be lower than all bars on left AND right
        left_ok  = all(rsi_val < rsi_vals[i - j] for j in range(1, lb_left + 1))
        right_ok = all(rsi_val < rsi_vals[i + j] for j in range(1, lb_right + 1))

        if left_ok and right_ok:
            pivots.append({
                "idx"      : i,
                "rsi_val"  : round(rsi_val, 4),
                "price_low": round(price_lows[i], 8),  # price at same bar
                "bars_ago" : n - 1 - i
            })

    return pivots

# ─────────────────────────────────────────
# Hidden Bullish — exact TradingView match
#
# From Pine source:
#   plFound  = RSI pivot low found
#   oscLL    = osc[lbR] < valuewhen(plFound, osc[lbR], 1)
#              → current pivot RSI < previous pivot RSI  = RSI LOWER LOW
#   priceHL  = low[lbR] > valuewhen(plFound, low[lbR], 1)
#              → current pivot price > previous pivot price = PRICE HIGHER LOW
#
# Hidden Bullish = Price Higher Low + RSI Lower Low
# (trend continuation signal — uptrend pullback)
# ─────────────────────────────────────────
def detect_hidden_bullish(rsi_vals: list, price_lows: list,
                           lb_left=5, lb_right=5,
                           range_lower=5, range_upper=60,
                           max_signal_age=30):
    """
    Finds all hidden bullish divergences.
    Returns most recent signal or None.
    max_signal_age = reject signals older than this many bars
    """
    pivots = find_rsi_pivot_lows(
        rsi_vals, price_lows, lb_left, lb_right
    )

    if len(pivots) < 2:
        return None

    signals = []

    # compare each consecutive pair of pivots
    # curr = more recent pivot (higher index)
    # prev = older pivot
    for k in range(1, len(pivots)):
        curr = pivots[k]
        prev = pivots[k - 1]

        bars_between = curr["idx"] - prev["idx"]

        # inRange: bars between pivots must be in [rangeLower, rangeUpper]
        if not (range_lower <= bars_between <= range_upper):
            continue

        # ── Hidden Bullish (exact Pine logic) ──
        osc_ll   = curr["rsi_val"]   < prev["rsi_val"]    # RSI lower low
        price_hl = curr["price_low"] > prev["price_low"]  # Price higher low

        if osc_ll and price_hl:
            signals.append({
                "curr_bars_ago"  : curr["bars_ago"],
                "prev_bars_ago"  : prev["bars_ago"],
                "curr_rsi"       : curr["rsi_val"],
                "prev_rsi"       : prev["rsi_val"],
                "curr_price_low" : curr["price_low"],
                "prev_price_low" : prev["price_low"],
                "bars_between"   : bars_between,
                "rsi_dropped_by" : round(prev["rsi_val"] - curr["rsi_val"], 2),
                "price_rose_by"  : round(
                    (curr["price_low"] - prev["price_low"])
                    / prev["price_low"] * 100, 2
                ),
            })

    if not signals:
        return None

    # most recent signal
    latest = signals[-1]

    # reject if too old
    # note: curr_bars_ago is always >= lb_right because pivot needs
    # lb_right bars to confirm. So minimum age = lb_right bars
    if latest["curr_bars_ago"] > max_signal_age:
        return None

    return latest

# ─────────────────────────────────────────
# Process symbol
# ─────────────────────────────────────────
def process_symbol(symbol):
    try:
        klines = get_klines_500(symbol)
        if not klines or len(klines) < 100:
            return None

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close",
            "volume", "close_time", "qav", "trades",
            "taker_base", "taker_quote", "ignore"
        ])
        df["close"]  = df["close"].astype(float)
        df["low"]    = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # volume filter
        if df["volume"].tail(10).mean() < 50000:
            return None

        # RSI on closed candles
        rsi      = RSIIndicator(df["close"], window=14).rsi()
        rsi_vals = rsi.tolist()
        price_lows = df["low"].tolist()

        # detect hidden bullish
        signal = detect_hidden_bullish(
            rsi_vals, price_lows,
            lb_left=LB_LEFT,
            lb_right=LB_RIGHT,
            range_lower=RANGE_LOWER,
            range_upper=RANGE_UPPER,
            max_signal_age=30    # signal pivot must be within last 30h
        )

        if signal is None:
            return None

        # live RSI
        live_rsi, current_price = get_live_rsi(symbol, klines)
        if live_rsi is None:
            return None

        # cooldown
        if symbol in last_alert_rsi:
            if abs(live_rsi - last_alert_rsi[symbol]) < RSI_CHANGE_THRESHOLD:
                return None
        last_alert_rsi[symbol] = live_rsi

        if live_rsi <= 40:
            zone = "30-40 oversold"
        elif live_rsi <= 50:
            zone = "40-50 mid recovery"
        else:
            zone = "50-60 mid zone"

        return {
            "symbol"          : symbol,
            "prev_price_low"  : signal["prev_price_low"],
            "curr_price_low"  : signal["curr_price_low"],
            "price_rose_by"   : signal["price_rose_by"],
            "prev_rsi"        : signal["prev_rsi"],
            "curr_rsi"        : signal["curr_rsi"],
            "rsi_dropped_by"  : signal["rsi_dropped_by"],
            "bars_between"    : signal["bars_between"],
            "signal_hours_ago": signal["curr_bars_ago"],
            "live_rsi"        : live_rsi,
            "current_price"   : current_price,
            "zone"            : zone,
        }

    except Exception as e:
        logger.error(f"{symbol} error: {e}")
        return None

# ─────────────────────────────────────────
# Scan
# ─────────────────────────────────────────
def scan():
    logger.info("Starting scan")
    symbols = get_all_usdt_symbols()
    matches = []
    start   = time.time()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_symbol, s) for s in symbols]
        for future in as_completed(futures):
            result = future.result()
            if result:
                logger.info(
                    f"H.BULL: {result['symbol']} | "
                    f"Price HL: {result['prev_price_low']} → {result['curr_price_low']} "
                    f"(+{result['price_rose_by']}%) | "
                    f"RSI LL: {result['prev_rsi']} → {result['curr_rsi']} "
                    f"(-{result['rsi_dropped_by']}) | "
                    f"{result['bars_between']}h apart | "
                    f"Signal {result['signal_hours_ago']}h ago | "
                    f"Live RSI={result['live_rsi']} | {result['zone']}"
                )
                matches.append(result)

    print(f"\nScan done in {round(time.time() - start, 2)}s")

    if matches:
        # freshest first
        matches.sort(key=lambda x: x["signal_hours_ago"])

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n=== HIDDEN BULLISH === {ts}")
        print("-" * 85)
        for m in matches:
            print(
                f"{m['symbol']:<12} | "
                f"Price HL: {m['prev_price_low']} → {m['curr_price_low']} "
                f"(+{m['price_rose_by']}%) | "
                f"RSI LL: {m['prev_rsi']} → {m['curr_rsi']} "
                f"(-{m['rsi_dropped_by']}) | "
                f"{m['bars_between']}h apart | "
                f"Signal {m['signal_hours_ago']}h ago | "
                f"Live RSI={m['live_rsi']} | {m['zone']}"
            )
        print("-" * 85)
        print(f"Total: {len(matches)}\n")

        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== HIDDEN BULLISH === {ts}\n")
            for m in matches:
                f.write(
                    f"{m['symbol']:<12} | "
                    f"Price HL: {m['prev_price_low']} → {m['curr_price_low']} "
                    f"(+{m['price_rose_by']}%) | "
                    f"RSI LL: {m['prev_rsi']} → {m['curr_rsi']} "
                    f"(-{m['rsi_dropped_by']}) | "
                    f"{m['bars_between']}h apart | "
                    f"Signal {m['signal_hours_ago']}h ago | "
                    f"Live RSI={m['live_rsi']} | {m['zone']}\n"
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
        logger.info("Sleeping 5 minutes...\n")
        time.sleep(300)