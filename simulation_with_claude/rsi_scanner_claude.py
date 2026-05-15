import requests
import pandas as pd
import time
import logging
from ta.momentum import RSIIndicator
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
import threading
import time
from datetime import datetime, timezone


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

 #─────────────────────────────────────────
# IN-MEMORY DATABASE
# ─────────────────────────────────────────
class TradeDB:
    def __init__(self):
        self.open   = {}   # symbol → trade dict
        self.closed = []   # list of closed trades
        self.lock   = threading.Lock()
        self.stats  = {"wins": 0, "losses": 0, "total_pnl_pct": 0.0}

    def add(self, trade):
        with self.lock:
            self.open[trade["symbol"]] = trade

    def get_open(self, symbol):
        return self.open.get(symbol)

    def close(self, symbol, exit_price, reason):
        with self.lock:
            t = self.open.pop(symbol, None)
            if not t:
                return
            pnl_pct = round((exit_price - t["entry"]) / t["entry"] * 100, 2)
            closed = {**t, "exit": exit_price, "pnl_pct": pnl_pct, "reason": reason,
                      "closed_at": datetime.now(timezone.utc).strftime("%H:%M UTC")}
            self.closed.append(closed)
            self.stats["total_pnl_pct"] = round(self.stats["total_pnl_pct"] + pnl_pct, 2)
            if pnl_pct >= 0:
                self.stats["wins"] += 1
            else:
                self.stats["losses"] += 1
            self._log_closed(closed)

    def _log_closed(self, t):
        emoji = "✅" if t["pnl_pct"] >= 0 else "❌"
        logger.info(
            f"{emoji} CLOSED {t['symbol']} | "
            f"Entry={t['entry']} Exit={t['exit']} | "
            f"PnL={t['pnl_pct']}% | "
            f"1h%={t.get('one_hour_pct', 'N/A')} | "
            f"Reason={t['reason']}"
        )
        with open("trades.txt", "a") as f:
            f.write(
                f"SELL | {t['symbol']} | Entry={t['entry']} | Exit={t['exit']} | "
                f"PnL={t['pnl_pct']}% | 1h%={t.get('one_hour_pct','N/A')} | "
                f"Reason={t['reason']} | {t['closed_at']}\n"
            )

    def print_summary(self):
        logger.info("─" * 50)
        logger.info(f"  OPEN TRADES   : {len(self.open)}")
        logger.info(f"  CLOSED TRADES : {len(self.closed)}")
        logger.info(f"  WINS / LOSSES : {self.stats['wins']} / {self.stats['losses']}")
        logger.info(f"  TOTAL PnL     : {self.stats['total_pnl_pct']}%")
        logger.info("─" * 50)

db = TradeDB()   # ← single global instance

# ─────────────────────────────────────────
# TRADE CONFIG
# ─────────────────────────────────────────
TRADE_AMOUNT_USDT = 10
TAKE_PROFIT_PCT   = 0.02   # 2%
STOP_LOSS_PCT     = 0.01   # 1%
MAX_HOLD_MINUTES  = 60
MAX_OPEN_TRADES   = 3

# ─────────────────────────────────────────
# BUY — stores trade in memory DB
# ─────────────────────────────────────────
def open_trade(symbol, live_rsi):
    with db.lock:
        if symbol in db.open:
            return
        if len(db.open) >= MAX_OPEN_TRADES:
            logger.info(f"Max trades open, skipping {symbol}")
            return

    price  = get_price(symbol)
    target = round(price * (1 + TAKE_PROFIT_PCT), 8)
    stop   = round(price * (1 - STOP_LOSS_PCT), 8)

    trade = {
        "symbol"       : symbol,
        "entry"        : price,
        "target"       : target,
        "stop"         : stop,
        "live_rsi"     : live_rsi,
        "open_time"    : time.time(),
        "open_ts"      : datetime.now(timezone.utc).strftime("%H:%M UTC"),

        # 1h tracking
        "hour_start_price" : price,   # price at entry
        "one_hour_pct"     : None,    # filled after 1h
        "hour_recorded"    : False,
    }

    db.add(trade)

    logger.info(
        f"📈 BUY  {symbol} | Entry={price} | "
        f"Target={target} (+{TAKE_PROFIT_PCT*100}%) | "
        f"Stop={stop} (-{STOP_LOSS_PCT*100}%) | "
        f"Live RSI={live_rsi}"
    )
    with open("trades.txt", "a") as f:
        f.write(
            f"BUY  | {symbol} | Entry={price} | Target={target} | "
            f"Stop={stop} | RSI={live_rsi} | {trade['open_ts']}\n"
        )

    t = threading.Thread(target=monitor_trade, args=(symbol,), daemon=True)
    t.start()

# -------------------------------
# Get current price
# -------------------------------
def get_price(symbol):
    r = session.get(
        f"{B_API}/api/v3/ticker/price?symbol={symbol}", timeout=5
    ).json()
    return float(r["price"])


# ─────────────────────────────────────────
# MONITOR — checks price + 1h % every 20s
# ─────────────────────────────────────────
def monitor_trade(symbol):
    while True:
        try:
            trade = db.get_open(symbol)
            if not trade:
                break   # already closed

            price   = get_price(symbol)
            elapsed = (time.time() - trade["open_time"]) / 60   # minutes

            # ── 1h % recording ──────────────────────
            if not trade["hour_recorded"] and elapsed >= 60:
                one_hour_pct = round((price - trade["hour_start_price"])
                                     / trade["hour_start_price"] * 100, 2)
                trade["one_hour_pct"]   = one_hour_pct
                trade["hour_recorded"]  = True
                logger.info(
                    f"⏱ 1H MARK {symbol} | "
                    f"Entry={trade['entry']} → Now={price} | "
                    f"1h Change={one_hour_pct}%"
                )

            # ── exit conditions ──────────────────────
            if price >= trade["target"]:
                db.close(symbol, price, "TAKE_PROFIT")
                break

            if price <= trade["stop"]:
                db.close(symbol, price, "STOP_LOSS")
                break

            if elapsed >= MAX_HOLD_MINUTES:
                db.close(symbol, price, "TIMEOUT")
                break

            logger.debug(
                f"👀 {symbol} | Price={price} | "
                f"PnL={round((price-trade['entry'])/trade['entry']*100,2)}% | "
                f"1h%={trade['one_hour_pct']} | "
                f"Elapsed={round(elapsed,1)}min"
            )

        except Exception as e:
            logger.error(f"Monitor error {symbol}: {e}")

        time.sleep(20)
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
# Track last alerted RSI value per symbol (not time)
last_alert_rsi = {}
RSI_CHANGE_THRESHOLD = 3  # only re-alert if live RSI moved 3+ points since last alert

def process_symbol(symbol):
    try:
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

        if df["volume"].tail(10).mean() < 50000:
            return None

        rsi        = RSIIndicator(df["close"], window=14).rsi()
        rsi_closed = rsi.iloc[-2]
        rsi_prev   = rsi.iloc[-3]

        if not (rsi_prev < rsi_closed < 53):
            return None

        if (rsi_closed - rsi_prev) < 1:
            return None

        live_rsi = get_live_rsi(symbol, klines)
        if live_rsi is None:
            return None

        if not (53 < live_rsi <= 60):
            return None

        # ✅ RSI-based cooldown instead of time-based
        if symbol in last_alert_rsi:
            last_rsi = last_alert_rsi[symbol]
            if abs(live_rsi - last_rsi) < RSI_CHANGE_THRESHOLD:
                return None   # RSI hasn't moved enough, skip

        last_alert_rsi[symbol] = live_rsi  # update with current live RSI

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

    for m in matches:
     open_trade(m["symbol"])  

# -------------------------------
# MAIN LOOP
# -------------------------------
if __name__ == "__main__":
    logger.info("Scanner started With Claude Logic")
    while True:
        scan()
        logger.info("Sleeping 2.5 minutes...\n")
        time.sleep(150)