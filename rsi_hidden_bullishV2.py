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

# ─────────────────────────────────────────
# Config — mirrors TradingView inputs
# ─────────────────────────────────────────
LB_LEFT        = 5    # pivotlow left bars
LB_RIGHT       = 5    # pivotlow right bars
RANGE_LOWER    = 5    # min bars between pivots
RANGE_UPPER    = 60   # max bars between pivots

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
# Pivot Low finder
# Exact match to ta.pivotlow(osc, lbL, lbR)
# A pivot low at index i means:
#   rsi[i] is lower than lbL bars to the left
#   rsi[i] is lower than lbR bars to the right
# ─────────────────────────────────────────
def find_pivot_lows(rsi_vals: list, price_lows: list,
                    lb_left=5, lb_right=5):
    """
    Returns list of pivot low dicts:
    { idx, rsi_val, price_val, bars_ago }
    idx is position in the list.
    bars_ago = how many bars from the END of list
               (same as lbR offset in Pine — the pivot is lbR bars ago)
    """
    pivots = []
    n = len(rsi_vals)

    for i in range(lb_left, n - lb_right):
        rsi_val   = rsi_vals[i]
        price_val = price_lows[i]

        # RSI pivot low check
        left_ok  = all(rsi_val < rsi_vals[i - j]   for j in range(1, lb_left + 1))
        right_ok = all(rsi_val < rsi_vals[i + j]   for j in range(1, lb_right + 1))

        if left_ok and right_ok:
            bars_ago = n - 1 - i   # how many bars from the end
            pivots.append({
                "idx"      : i,
                "rsi_val"  : round(rsi_val, 2),
                "price_val": round(price_val, 8),
                "bars_ago" : bars_ago
            })

    return pivots

# ─────────────────────────────────────────
# In range check
# Mirrors Pine: rangeLower <= barssince(prev pivot) <= rangeUpper
# ─────────────────────────────────────────
def in_range(bars_between, range_lower=5, range_upper=60):
    return range_lower <= bars_between <= range_upper

# ─────────────────────────────────────────
# HIDDEN BULLISH — exact TradingView logic
#
# plFound  = current pivot low exists
# oscLL    = current pivot RSI < previous pivot RSI  (RSI lower low)
# priceHL  = current pivot price > previous pivot price (price higher low)
# inRange  = bars between pivots in [5, 60]
# ─────────────────────────────────────────
def detect_hidden_bullish_tv(rsi_vals: list, price_lows: list,
                              lb_left=5, lb_right=5,
                              range_lower=5, range_upper=60):
    """
    Scans all pivot lows and checks for hidden bullish divergence.
    Returns list of all detected signals (most recent last).
    """
    pivots  = find_pivot_lows(rsi_vals, price_lows, lb_left, lb_right)
    signals = []

    for k in range(1, len(pivots)):
        curr = pivots[k]    # current pivot (more recent)
        prev = pivots[k-1]  # previous pivot (older)

        bars_between = curr["idx"] - prev["idx"]

        # ── inRange check ──
        if not in_range(bars_between, range_lower, range_upper):
            continue

        # ── Hidden Bullish conditions (exact TradingView) ──
        osc_ll   = curr["rsi_val"]   < prev["rsi_val"]    # RSI lower low
        price_hl = curr["price_val"] > prev["price_val"]  # Price higher low

        if osc_ll and price_hl:
            signals.append({
                "curr_idx"       : curr["idx"],
                "prev_idx"       : prev["idx"],
                "curr_bars_ago"  : curr["bars_ago"],
                "prev_bars_ago"  : prev["bars_ago"],
                "curr_rsi"       : curr["rsi_val"],
                "prev_rsi"       : prev["rsi_val"],
                "curr_price_low" : curr["price_val"],
                "prev_price_low" : prev["price_val"],
                "bars_between"   : bars_between,
                "rsi_diff"       : round(prev["rsi_val"] - curr["rsi_val"], 2),
                "price_diff_pct" : round(
                    (curr["price_val"] - prev["price_val"])
                    / prev["price_val"] * 100, 2
                ),
            })

    return signals

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

        # RSI on closed candles only (exclude live forming candle)
        rsi        = RSIIndicator(df["close"], window=14).rsi()
        rsi_vals   = rsi.tolist()
        price_lows = df["low"].tolist()

        # detect hidden bullish — exact TradingView logic
        signals = detect_hidden_bullish_tv(
            rsi_vals, price_lows,
            lb_left=LB_LEFT,
            lb_right=LB_RIGHT,
            range_lower=RANGE_LOWER,
            range_upper=RANGE_UPPER
        )

        if not signals:
            return None

        # get the most recent signal
        latest = signals[-1]

        # ── freshness filter ──
        # curr_bars_ago = how many bars ago the signal pivot formed
        # lbR=5 means the pivot is confirmed 5 bars after it formed
        # so minimum bars_ago is always lbR (5)
        # we want signals where pivot formed recently (within 24h)
        if latest["curr_bars_ago"] > 24 + LB_RIGHT:
            return None   # signal too old

        # live RSI
        live_rsi, current_price = get_live_rsi(symbol, klines)
        if live_rsi is None:
            return None

        # cooldown
        if symbol in last_alert_rsi:
            if abs(live_rsi - last_alert_rsi[symbol]) < RSI_CHANGE_THRESHOLD:
                return None
        last_alert_rsi[symbol] = live_rsi

        hours_ago = latest["curr_bars_ago"]   # 1h candles = hours

        if live_rsi <= 40:
            zone = "30-40 oversold"
        elif live_rsi <= 50:
            zone = "40-50 mid recovery"
        else:
            zone = "50-60 mid zone"

        return {
            "symbol"          : symbol,
            "prev_price_low"  : latest["prev_price_low"],
            "curr_price_low"  : latest["curr_price_low"],
            "price_diff_pct"  : latest["price_diff_pct"],
            "prev_rsi"        : latest["prev_rsi"],
            "curr_rsi"        : latest["curr_rsi"],
            "rsi_diff"        : latest["rsi_diff"],
            "bars_between"    : latest["bars_between"],
            "hours_ago"       : hours_ago,
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
                    f"Price: {result['prev_price_low']} → {result['curr_price_low']} "
                    f"(+{result['price_diff_pct']}%) | "
                    f"RSI: {result['prev_rsi']} → {result['curr_rsi']} "
                    f"(-{result['rsi_diff']}) | "
                    f"{result['bars_between']}h apart | "
                    f"Signal {result['hours_ago']}h ago | "
                    f"Live RSI={result['live_rsi']} | {result['zone']}"
                )
                matches.append(result)

    print(f"\nScan done in {round(time.time() - start, 2)}s")

    if matches:
        # sort by freshest signal first
        matches.sort(key=lambda x: x["hours_ago"])

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n=== HIDDEN BULLISH (TradingView Logic) === {ts}")
        print("-" * 85)
        for m in matches:
            print(
                f"{m['symbol']:<12} | "
                f"Price: {m['prev_price_low']} → {m['curr_price_low']} "
                f"(+{m['price_diff_pct']}%) | "
                f"RSI: {m['prev_rsi']} → {m['curr_rsi']} "
                f"(-{m['rsi_diff']}) | "
                f"{m['bars_between']}h apart | "
                f"Signal {m['hours_ago']}h ago | "
                f"Live RSI={m['live_rsi']} | {m['zone']}"
            )
        print("-" * 85)
        print(f"Total: {len(matches)}\n")

        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== HIDDEN BEARCHISH === {ts}\n")
            for m in matches:
                f.write(
                    f"{m['symbol']:<12} | "
                    f"Price: {m['prev_price_low']} → {m['curr_price_low']} "
                    f"(+{m['price_diff_pct']}%) | "
                    f"RSI: {m['prev_rsi']} → {m['curr_rsi']} "
                    f"(-{m['rsi_diff']}) | "
                    f"{m['bars_between']}h apart | "
                    f"Signal {m['hours_ago']}h ago | "
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