import threading
import time
from datetime import datetime, timezone

# ─────────────────────────────────────────
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