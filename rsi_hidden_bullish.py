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
# Swing low finder
# point i is a swing low if it's strictly lower
# than `left` candles to the left AND `right` candles to the right
# ─────────────────────────────────────────
def find_swing_lows(values: list, left=5, right=5):
    lows = []
    for i in range(left, len(values) - right):
        if all(values[i] < values[i - j] for j in range(1, left + 1)) and \
           all(values[i] < values[i + j] for j in range(1, right + 1)):
            lows.append(i)
    return lows

# ─────────────────────────────────────────
# Live RSI — inject ticker price into forming candle
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

    s        = pd.Series(closes)
    live_rsi = RSIIndicator(s, window=period).rsi()
    return round(live_rsi.iloc[-1], 2), round(current_price, 8)

# ─────────────────────────────────────────
# REAL Hidden Bullish Divergence
#
#   PRICE → Lower Low  (p2 < p1)
#   RSI   → Higher Low (r2 > r1)
#
# Live RSI must be recovering into 30–55 zone
# ─────────────────────────────────────────
# ─────────────────────────────────────────
# Find ONE confirmed historical swing low
# (well in the past, fully confirmed by bars on both sides)
# ─────────────────────────────────────────
def find_last_confirmed_low(price_list, rsi_list, left=5, right=5,
                             search_from=-80, search_to=-10):
    """
    Find the MOST RECENT confirmed swing low.
    - search_from / search_to controls the window
    - We pick the LAST candidate (closest to now) not the first
    - right=5 means the low must have 5 confirmed bars after it
      so search_to=-5 minimum to allow confirmation
    """
    p_slice = price_list[search_from:search_to]
    r_slice = rsi_list[search_from:search_to]

    total_len   = len(price_list)
    slice_start = total_len + search_from   # actual index in full array

    candidates = []
    for i in range(left, len(p_slice) - right):
        left_ok  = all(p_slice[i] < p_slice[i - j] for j in range(1, left + 1))
        right_ok = all(p_slice[i] < p_slice[i + j] for j in range(1, right + 1))

        if left_ok and right_ok:
            actual_idx  = slice_start + i
            hours_ago   = total_len - 1 - actual_idx
            candidates.append((i, p_slice[i], r_slice[i], hours_ago))

    if not candidates:
        return None

    # ✅ return the LAST = most recent confirmed swing low
    return candidates[-1]   # (slice_idx, price, rsi, hours_ago)


# ─────────────────────────────────────────
# CURRENT Hidden Bullish Detection
#
# Low1 = confirmed historical swing low
# Low2 = RIGHT NOW (live candle or last closed candle)
#
# Price NOW < Price Low1  → lower low  ✅
# RSI   NOW > RSI   Low1  → higher low ✅
# ─────────────────────────────────────────
def detect_current_hidden_bullish(df, rsi_series, live_rsi,
                                   left=5, right=5,
                                   min_price_drop_pct=0.3,
                                   min_rsi_diff=2.0,
                                   rsi_zone=(28, 58),
                                   max_hours_ago=72):   # ← only accept Low1 within 72h
    price_list = df["low"].tolist()
    rsi_list   = rsi_series.tolist()

    historical = find_last_confirmed_low(
        price_list, rsi_list,
        left=left,
        right=right,
        search_from=-80,    # look back max 80 candles = 80h
        search_to=-5        # must be at least 5 candles old to be confirmed
    )

    if historical is None:
        return False, None

    _, p1, r1, hours_ago = historical   # ← now returns hours_ago directly

    # ✅ reject if Low1 is too old — not the latest signal
    if hours_ago > max_hours_ago:
        return False, None

    # current low = minimum of last 2 candles
    current_low = min(
        df["low"].iloc[-1],   # live forming candle
        df["low"].iloc[-2],   # last closed candle
    )

    r2 = live_rsi

    # price must be making lower low vs Low1
    price_drop_pct = ((p1 - current_low) / p1) * 100
    if price_drop_pct < min_price_drop_pct:
        return False, None

    # RSI must be making higher low vs Low1
    rsi_diff = r2 - r1
    if rsi_diff < min_rsi_diff:
        return False, None

    # RSI zone filter
    if not (rsi_zone[0] <= r1 <= rsi_zone[1]):
        return False, None
    if not (rsi_zone[0] <= r2 <= rsi_zone[1]):
        return False, None

    return True, {
        "price_low1"     : round(p1, 8),
        "current_low"    : round(current_low, 8),
        "price_drop_pct" : round(price_drop_pct, 2),
        "rsi_at_low1"    : round(r1, 2),
        "live_rsi"       : round(r2, 2),
        "rsi_held_by"    : round(rsi_diff, 2),
        "low1_hours_ago" : hours_ago,
    }
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

        if df["volume"].tail(10).mean() < 50000:
            return None

        # RSI on closed candles
        rsi = RSIIndicator(df["close"], window=14).rsi()

        # get live RSI first (this is our "current Low2")
        live_rsi, current_price = get_live_rsi(symbol, klines)
        if live_rsi is None:
            return None

        # quick pre-filter: live RSI must already be in recovery zone
        # no point doing heavy detection if RSI is at 70+
        if not (28 <= live_rsi <= 58):
            return None

        # detect current hidden bullish
        found, details = detect_current_hidden_bullish(
            df, rsi, live_rsi,
            left=5,
            right=8,
            min_price_drop_pct=0.5,
            min_rsi_diff=2.0,
            rsi_zone=(28, 58)
        )

        if not found:
            return None

        # cooldown
        if symbol in last_alert_rsi:
            if abs(live_rsi - last_alert_rsi[symbol]) < RSI_CHANGE_THRESHOLD:
                return None
        last_alert_rsi[symbol] = live_rsi

        # zone label
        if live_rsi <= 40:
            zone = "30-40 oversold"
        elif live_rsi <= 50:
            zone = "40-50 mid recovery"
        else:
            zone = "50-58 mid zone"

        return {
            "symbol"         : symbol,
            "price_low1"     : details["price_low1"],
            "current_low"    : details["current_low"],
            "price_drop_pct" : details["price_drop_pct"],
            "rsi_at_low1"    : details["rsi_at_low1"],
            "live_rsi"       : details["live_rsi"],
            "rsi_held_by"    : details["rsi_held_by"],
            "low1_hours_ago" : details["low1_hours_ago"],
            "current_price"  : current_price,
            "zone"           : zone,
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
                    f"HIDDEN BULL: {result['symbol']} | "
                    f"Price Low1={result['price_low1']} → Now={result['current_low']} "
                    f"(-{result['price_drop_pct']}%) | "
                    f"RSI Low1={result['rsi_at_low1']} → Live={result['live_rsi']} "
                    f"(+{result['rsi_held_by']}) | "
                    f"Low1 was {result['low1_hours_ago']}h ago | {result['zone']}"
                )
                matches.append(result)

    print(f"\nScan done in {round(time.time() - start, 2)}s")

    # In scan(), replace the print and file write section with this:

    if matches:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n=== HIDDEN BULLISH === {ts}")
        print("-" * 80)
        for m in matches:
            print(
                f"{m['symbol']:<12} | "
                f"Price Low1={m['price_low1']} → Now={m['current_low']} "
                f"(-{m['price_drop_pct']}%) | "
                f"RSI Low1={m['rsi_at_low1']} → Live={m['live_rsi']} "
                f"(+{m['rsi_held_by']}) | "
                f"Low1={m['low1_hours_ago']}h ago | "        # ← new
                f"Price={m['current_price']} | {m['zone']}"
            )
        print("-" * 80)
        print(f"Total: {len(matches)}\n")

        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== HIDDEN BULLISH === {ts}\n")
            for m in matches:
                f.write(
                    f"{m['symbol']:<12} | "
                    f"Price Low1={m['price_low1']} → Now={m['current_low']} "
                    f"(-{m['price_drop_pct']}%) | "
                    f"RSI Low1={m['rsi_at_low1']} → Live={m['live_rsi']} "
                    f"(+{m['rsi_held_by']}) | "
                    f"Low1={m['low1_hours_ago']}h ago | "        # ← new
                    f"Price={m['current_price']} | {m['zone']}\n"
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