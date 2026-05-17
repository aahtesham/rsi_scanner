import os
import sys
import threading
import time
import logging

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from time_util import scan_timestamp, trade_time_stamp

import requests
import pandas as pd
from ta.momentum import RSIIndicator
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter


# ─────────────────────────────────────────
# IN-MEMORY DATABASE
# ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
# -------------------------------
# CONFIGURATION
# -------------------------------
TRADE_AMOUNT_USDT = 10       # paper notional per trade (logged + est. USDT PnL); exits still %-based on entry
TAKE_PROFIT_PCT   = 0.011    # 1.1% — more reachable than 2% on short holds
STOP_LOSS_PCT     = 0.018    # 1.8% — wider than 1% to reduce noise stops on volatile alts
MAX_HOLD_MINUTES  = 90       # more time to reach TP before TIMEOUT
MAX_OPEN_TRADES   = 5        # don't open more than 5 at once
MONITOR_LOG_INTERVAL_S = 60  # INFO heartbeat while open (debug every 20s is hidden at INFO)
MAX_CLOSED_IN_MEMORY = 500   # trim closed-trade list to limit RAM
MAX_ALERT_RSI_KEYS = 350     # prune cooldown map when it grows (symbols not currently open)

# Track last alerted RSI value per symbol (not time)
last_alert_rsi = {}
last_alert_lock = threading.Lock()
RSI_CHANGE_THRESHOLD = 3

B_API = "https://api.binance.com"
session = requests.Session()
adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
session.mount("https://", adapter)
session.mount("http://", adapter)


def _prune_last_alert_rsi_if_needed() -> None:
    """Drop cooldown entries for symbols that are not open when the map grows large."""
    with last_alert_lock:
        if len(last_alert_rsi) <= MAX_ALERT_RSI_KEYS:
            return
        with db.lock:
            open_syms = set(db.open.keys())
        for k in list(last_alert_rsi.keys()):
            if k not in open_syms:
                last_alert_rsi.pop(k, None)
            if len(last_alert_rsi) <= MAX_ALERT_RSI_KEYS:
                break


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
        with self.lock:
            return self.open.get(symbol)

    def close(self, symbol, exit_price, reason):
        closed = None
        with self.lock:
            t = self.open.pop(symbol, None)
            if not t:
                return
            pnl_pct = round((exit_price - t["entry"]) / t["entry"] * 100, 2)
            notional = float(t.get("notional_usdt", TRADE_AMOUNT_USDT))
            pnl_usdt_est = round(notional * (pnl_pct / 100.0), 4)
            closed = {
                **t,
                "exit": exit_price,
                "pnl_pct": pnl_pct,
                "pnl_usdt_est": pnl_usdt_est,
                "reason": reason,
                "closed_at": trade_time_stamp(),
            }
            self.closed.append(closed)
            if len(self.closed) > MAX_CLOSED_IN_MEMORY:
                del self.closed[: len(self.closed) - MAX_CLOSED_IN_MEMORY]
            self.stats["total_pnl_pct"] = round(self.stats["total_pnl_pct"] + pnl_pct, 2)
            if pnl_pct >= 0:
                self.stats["wins"] += 1
            else:
                self.stats["losses"] += 1
        if closed is not None:
            self._log_closed(closed)

    def _log_closed(self, t):
        emoji = "✅" if t["pnl_pct"] >= 0 else "❌"
        logger.info(
            f"{emoji} CLOSED {t['symbol']} | "
            f"Entry={t['entry']} Exit={t['exit']} | "
            f"PnL={t['pnl_pct']}% (~{t.get('pnl_usdt_est', 'n/a')} USDT @ {t.get('notional_usdt', TRADE_AMOUNT_USDT)} notion) | "
            f"1h%={t.get('one_hour_pct', 'N/A')} | "
            f"Reason={t['reason']}"
        )
        with open("trades.txt", "a") as f:
            f.write(
                f"SELL_TRIGGER | {t['symbol']} | reason={t['reason']} | "
                f"entry={t['entry']} | exit={t['exit']} | pnl={t['pnl_pct']}% | "
                f"pnl_usdt_est={t.get('pnl_usdt_est')} | {t['closed_at']}\n"
            )
            f.write(
                f"SELL | {t['symbol']} | Entry={t['entry']} | Exit={t['exit']} | "
                f"PnL={t['pnl_pct']}% | pnl_usdt_est={t.get('pnl_usdt_est')} | "
                f"1h%={t.get('one_hour_pct','N/A')} | "
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

# ─────────────────────────────────────────
# BUY — stores trade in memory DB
# ─────────────────────────────────────────
def open_trade(symbol, live_rsi) -> bool:
    """Open a position. Returns True if a new trade was opened (caller may record cooldown)."""
    with db.lock:
        if symbol in db.open:
            return False
        if len(db.open) >= MAX_OPEN_TRADES:
            logger.info(f"Max trades open, skipping {symbol}")
            return False

    price = get_price(symbol)
    if price is None:
        logger.error(f"open_trade price unavailable {symbol}")
        return False

    target = round(price * (1 + TAKE_PROFIT_PCT), 8)
    stop   = round(price * (1 - STOP_LOSS_PCT), 8)
    qty_base = round(TRADE_AMOUNT_USDT / price, 8) if price else 0.0

    trade = {
        "symbol"       : symbol,
        "entry"        : price,
        "target"       : target,
        "stop"         : stop,
        "live_rsi"     : live_rsi,
        "notional_usdt": float(TRADE_AMOUNT_USDT),
        "qty_base"     : qty_base,
        "open_time"    : time.time(),
        "open_ts"      : trade_time_stamp(),

        # 1h tracking
        "hour_start_price" : price,   # price at entry
        "one_hour_pct"     : None,    # filled after 1h
        "hour_recorded"    : False,
        "last_status_log_ts": 0.0,
    }

    db.add(trade)

    logger.info(
        f"📈 BUY  {symbol} | Entry={price} | "
        f"Target={target} (+{TAKE_PROFIT_PCT*100}%) | "
        f"Stop={stop} (-{STOP_LOSS_PCT*100}%) | "
        f"Notional≈{TRADE_AMOUNT_USDT} USDT qty≈{qty_base} | "
        f"Live RSI={live_rsi}"
    )
    with open("trades.txt", "a") as f:
        f.write(
            f"BUY  | {symbol} | Entry={price} | Target={target} | "
            f"Stop={stop} | notional_usdt={TRADE_AMOUNT_USDT} | qty_base≈{qty_base} | "
            f"RSI={live_rsi} | {trade['open_ts']}\n"
        )

    t = threading.Thread(target=monitor_trade, args=(symbol,), daemon=True)
    t.start()
    return True


# ─────────────────────────────────────────
# MONITOR — checks price + 1h % every 20s
# ─────────────────────────────────────────
def monitor_trade(symbol):
    while True:
        try:
            trade = db.get_open(symbol)
            if not trade:
                break   # already closed

            price = get_price(symbol)
            if price is None:
                logger.warning(f"Monitor {symbol}: price unavailable, retry in 20s")
                time.sleep(20)
                continue

            elapsed = (time.time() - trade["open_time"]) / 60   # minutes

            now_ts = time.time()
            if now_ts - float(trade.get("last_status_log_ts", 0)) >= MONITOR_LOG_INTERVAL_S:
                trade["last_status_log_ts"] = now_ts
                entry = float(trade["entry"])
                tgt = float(trade["target"])
                stp = float(trade["stop"])
                pnl_pct = round((price - entry) / entry * 100, 3)
                need_rise_pct = round((tgt - price) / entry * 100, 4)
                headroom_sl_pct = round((price - stp) / entry * 100, 4)
                logger.info(
                    f"📊 OPEN {symbol} | px={price} entry={entry} | "
                    f"PnL={pnl_pct}% | need_to_TP≈{need_rise_pct}% (from entry) | "
                    f"headroom_to_SL≈{headroom_sl_pct}% | "
                    f"elapsed={round(elapsed, 1)}m / {MAX_HOLD_MINUTES}m | "
                    f"target={tgt} stop={stp}"
                )

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
                logger.info(
                    f"🔔 SELL_TRIGGER TAKE_PROFIT {symbol} | price={price} >= target={trade['target']}"
                )
                db.close(symbol, price, "TAKE_PROFIT")
                break

            if price <= trade["stop"]:
                logger.info(
                    f"🔔 SELL_TRIGGER STOP_LOSS {symbol} | price={price} <= stop={trade['stop']}"
                )
                db.close(symbol, price, "STOP_LOSS")
                break

            if elapsed >= MAX_HOLD_MINUTES:
                logger.info(
                    f"🔔 SELL_TRIGGER TIMEOUT {symbol} | elapsed={round(elapsed, 1)}m | "
                    f"price={price} entry={trade['entry']} target={trade['target']}"
                )
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
# Get process_symbol
# -------------------------------
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

        # RSI-based cooldown (read only here; write after successful open_trade in scan())
        with last_alert_lock:
            if symbol in last_alert_rsi:
                last_rsi = last_alert_rsi[symbol]
                if abs(live_rsi - last_rsi) < RSI_CHANGE_THRESHOLD:
                    return None

        vol_mean = float(df["volume"].tail(10).mean())
        # Rank candidates: steeper 1h RSI rise + stronger continuation (live vs last close) + liquidity
        rsi_step = float(rsi_closed - rsi_prev)
        live_push = float(live_rsi - rsi_closed)
        match_score = rsi_step * 2.0 + live_push + min(vol_mean / 500_000.0, 5.0)

        return {
            "symbol"    : symbol,
            "rsi_prev"  : round(rsi_prev, 2),
            "rsi_closed": round(rsi_closed, 2),
            "live_rsi"  : round(live_rsi, 2),
            "match_score": round(match_score, 4),
            "vol_mean"  : round(vol_mean, 2),
        }

    except Exception as e:
        logger.error(f"{symbol} error: {e}")
        return None

# -------------------------------
# Get current price
# -------------------------------
def get_price(symbol):
    try:
        r = session.get(
            f"{B_API}/api/v3/ticker/price?symbol={symbol}", timeout=5
        ).json()
        return float(r["price"])
    except Exception as e:
        logger.debug(f"get_price failed {symbol}: {e}")
        return None


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
                    f"Live: {result['live_rsi']} | "
                    f"score={result.get('match_score')} vol10={result.get('vol_mean')}"
                )
                matches.append(result)

    print(f"\nScan done in {round(time.time() - start, 2)}s")

    matches.sort(key=lambda m: m.get("match_score", 0.0), reverse=True)

    if matches:
        print(f"\n=== MATCHES (best first by score) === {scan_timestamp()}")
        print("-" * 55)
        for m in matches:
            print(
                f"{m['symbol']:<12} | "
                f"1h RSI: {m['rsi_prev']} → {m['rsi_closed']} | "
                f"Live RSI: {m['live_rsi']} | "
                f"score={m.get('match_score')} | vol10={m.get('vol_mean')}"
            )
        print("-" * 55)
        print(f"Total: {len(matches)}\n")

        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== MATCHES === {scan_timestamp()}\n")
            for m in matches:
                f.write(
                    f"{m['symbol']:<12} | "
                    f"1h RSI: {m['rsi_prev']} → {m['rsi_closed']} | "
                    f"Live RSI: {m['live_rsi']} | score={m.get('match_score')}\n"
                )
            f.write(f"Total: {len(matches)}\n\n")
    else:
        print("No matches found.\n")

    # Best candidates first; only record RSI cooldown after a successful open.
    for m in matches:
        if open_trade(m["symbol"], m["live_rsi"]):
            with last_alert_lock:
                last_alert_rsi[m["symbol"]] = m["live_rsi"]

    _prune_last_alert_rsi_if_needed()

    # print DB summary after each scan
    db.print_summary()


# -------------------------------
# -------------------------------
# MAIN LOOP
# -------------------------------
if __name__ == "__main__":
    logger.info("Scanner started With Claude Logic")
    while True:
        scan()
        logger.info("Sleeping 2.5 minutes...\n")
        time.sleep(150)
