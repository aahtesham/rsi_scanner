import os
from pathlib import Path

import requests
import pandas as pd
import time
import logging
from ta.momentum import RSIIndicator
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter

from time_util import scan_timestamp
from datetime import datetime
from notify_telegram import format_matches_message, send_telegram, telegram_configured

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

NOTIFY_TELEGRAM = os.environ.get("NOTIFY_TELEGRAM", "1").strip().lower() in ("1", "true", "yes", "on")
SCAN_SLEEP_S = int(os.environ.get("SCAN_SLEEP_S", "150"))


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_PROJECT_DIR = Path(__file__).resolve().parent
_load_dotenv(_PROJECT_DIR / ".env")


# -------------------------------
# Get current price
# -------------------------------
def get_price(symbol):
    r = session.get(
        f"{B_API}/api/v3/ticker/price?symbol={symbol}", timeout=5
    ).json()
    return float(r["price"])


# -------------------------------
# Get all symbols
# -------------------------------
def get_all_usdt_symbols():
    url = f"{B_API}/api/v3/exchangeInfo"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RSI-Scanner/1.0)"}

    # Try exchangeInfo with a few retries
    for attempt in range(3):
        try:
            r = session.get(url, timeout=10, headers=headers)
            r.raise_for_status()
            return [
                s["symbol"]
                for s in r.json().get("symbols", [])
                if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
            ]
        except requests.exceptions.HTTPError as e:
            status = None
            try:
                status = e.response.status_code
            except Exception:
                pass
            logger.warning(f"exchangeInfo HTTP error (attempt {attempt+1}): {e} status={status}")
            # 451 indicates legal restriction — stop retrying and fallback
            if status == 451:
                logger.error("Received 451 from Binance (Unavailable For Legal Reasons). Falling back to ticker/price endpoint.")
                break
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"exchangeInfo error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)

    # Fallback: use ticker/price endpoint and filter for USDT pairs
    try:
        r = session.get(f"{B_API}/api/v3/ticker/price", timeout=10, headers=headers)
        r.raise_for_status()
        return [p["symbol"] for p in r.json() if p.get("symbol", "").endswith("USDT")]
    except Exception as e:
        logger.error(f"Failed to fetch symbols from Binance (fallback): {e}")
        # If Binance is blocked (451) or unavailable, try CoinGecko as a public fallback
        try:
            cg_url = (
                "https://api.coingecko.com/api/v3/coins/markets"
                "?vs_currency=usd&order=market_cap_desc&per_page=250&page=1&sparkline=false"
            )
            cg = session.get(cg_url, timeout=10, headers=headers)
            cg.raise_for_status()
            coins = cg.json()
            symbols = []
            for c in coins:
                s = c.get("symbol", "").upper()
                if not s:
                    continue
                symbols.append(s + "USDT")
            logger.info(f"CoinGecko fallback produced {len(symbols)} candidate symbols")
            return symbols
        except Exception as e2:
            logger.error(f"CoinGecko fallback failed: {e2}")
            return []

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
       
        df["close"]      = df["close"].astype(float)
        df["volume"]     = df["volume"].astype(float)
        df["taker_base"] = df["taker_base"].astype(float) 

        if df["volume"].tail(10).mean() < 50000:
            return None

        rsi        = RSIIndicator(df["close"], window=14).rsi()
        rsi_closed = rsi.iloc[-2]
        rsi_prev   = rsi.iloc[-3]

        buy_ratio = df["taker_base"].iloc[-2] / df["volume"].iloc[-2]
        if buy_ratio < 0.52:
            return None


        if not (rsi_prev < rsi_closed < 53):
            return None

        if (rsi_closed - rsi_prev) < 1:
            return None

        manual_live_rsi, auto_live_rsi, current_price = get_live_rsi(symbol, klines)
        if auto_live_rsi is None:
            return None

        if not (53 < auto_live_rsi <= 60):
            return None

        # ✅ RSI-based cooldown instead of time-based
        if symbol in last_alert_rsi:
            last_rsi = last_alert_rsi[symbol]
            if abs(auto_live_rsi - last_rsi) < RSI_CHANGE_THRESHOLD:
                return None   # RSI hasn't moved enough, skip

        last_alert_rsi[symbol] = auto_live_rsi  # update with current live RSI

        return {
            "symbol"    : symbol,
            "rsi_prev"  : round(rsi_prev, 2),
            "rsi_closed": round(rsi_closed, 2),
            "manual_live_rsi"    : round(manual_live_rsi, 2),
            "live_rsi"  : round(auto_live_rsi, 2),
            "current_price" : round(current_price, 8)
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
                    f"Live: {result['live_rsi']} | "
                    f"Price: {result['current_price']}"
                )
                matches.append(result)

    print(f"\nScan done in {round(time.time() - start, 2)}s")

    if matches:
        print(f"\n=== MATCHES === {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
        print("-" * 55)
        for m in matches:
            print(
                f"{m['symbol']:<12} | "
                f"1h RSI: {m['rsi_prev']} → {m['rsi_closed']} | "
                f"Manual Live RSI: {m['manual_live_rsi']} | "
                f"Live RSI: {m['live_rsi']} | "
                f"Price: {m['current_price']}"
            )
        print("-" * 55)
        print(f"Total: {len(matches)}\n")

        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== MATCHES === {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n")
            for m in matches:
                f.write(
                    f"{m['symbol']:<12} | "
                    f"1h RSI: {m['rsi_prev']} → {m['rsi_closed']} | "
                    f"Manual Live RSI: {m['manual_live_rsi']} | "
                    f"Live RSI: {m['live_rsi']} | "
                    f"Price: {m['current_price']}\n"
                )
            f.write(f"Total: {len(matches)}\n\n")
    
    
        if NOTIFY_TELEGRAM and telegram_configured():
                ok = send_telegram(
                    format_matches_message(matches, f"Early Bird is matches — {scan_timestamp()}")
                )
                if ok:
                    logger.info("Telegram sent (%s matches)", len(matches))
                else:
                    logger.warning("Telegram send failed (see warnings above)")
        elif NOTIFY_TELEGRAM:
                logger.warning(
                    "Telegram skipped: create %s with TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID",
                    _PROJECT_DIR / ".env",
                )
    else:
        print("No matches found.\n")
    
# -------------------------------
# MAIN LOOP
# -------------------------------
if __name__ == "__main__":
    logger.info("Scanner started With Claude Logic")
    while True:
        scan()
        logger.info("Sleeping 2.5 minutes...\n")
        time.sleep(350)