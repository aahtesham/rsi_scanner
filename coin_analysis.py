import requests
import pandas as pd
import numpy as np
import time
import logging
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
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

last_alert_rsi   = {}
last_alert_slope = {}

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
# Fetch candles for multiple timeframes
# ─────────────────────────────────────────
def get_klines(symbol, interval="1h", limit=200):
    r = session.get(
        f"{B_API}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=10
    )
    if r.status_code != 200:
        return None
    return r.json()

# ─────────────────────────────────────────
# Build dataframe
# ─────────────────────────────────────────
def build_df(klines):
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "qav", "trades",
        "taker_base", "taker_quote", "ignore"
    ])
    df["open"]   = df["open"].astype(float)
    df["high"]   = df["high"].astype(float)
    df["low"]    = df["low"].astype(float)
    df["close"]  = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["taker_base"] = df["taker_base"].astype(float)
    return df

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
# BTC dominance proxy
# Is the market risk-on? (BTC rising = risk on)
# ─────────────────────────────────────────
def get_btc_trend():
    try:
        klines = get_klines("BTCUSDT", interval="4h", limit=50)
        if not klines:
            return "unknown"
        df    = build_df(klines)
        ema20 = EMAIndicator(df["close"], window=20).ema_indicator()
        ema50 = EMAIndicator(df["close"], window=50).ema_indicator()
        if ema20.iloc[-2] > ema50.iloc[-2]:
            return "risk_on"    # BTC in uptrend = altcoins likely to follow
        return "risk_off"
    except Exception:
        return "unknown"

# ─────────────────────────────────────────
# DIMENSION 1: Trend Structure
# Is price in a healthy uptrend?
# ─────────────────────────────────────────
def analyze_trend(df):
    result = {"score": 0, "details": []}

    ema20 = EMAIndicator(df["close"], window=20).ema_indicator()
    ema50 = EMAIndicator(df["close"], window=50).ema_indicator()
    ema200 = EMAIndicator(df["close"], window=200).ema_indicator()

    price = df["close"].iloc[-2]

    # price above all EMAs = strong uptrend
    if price > ema20.iloc[-2]:
        result["score"] += 1
        result["details"].append("above EMA20")
    if price > ema50.iloc[-2]:
        result["score"] += 1
        result["details"].append("above EMA50")
    if price > ema200.iloc[-2]:
        result["score"] += 2
        result["details"].append("above EMA200")

    # EMA stack: 20 > 50 > 200 = perfect bull structure
    if ema20.iloc[-2] > ema50.iloc[-2] > ema200.iloc[-2]:
        result["score"] += 2
        result["details"].append("EMA stack bullish")

    result["ema20"]  = round(ema20.iloc[-2], 6)
    result["ema50"]  = round(ema50.iloc[-2], 6)
    result["ema200"] = round(ema200.iloc[-2], 6)

    return result   # max score: 6

# ─────────────────────────────────────────
# DIMENSION 2: Momentum Confluence
# Multiple momentum indicators agreeing
# ─────────────────────────────────────────
def analyze_momentum(df, live_rsi):
    result = {"score": 0, "details": []}

    rsi = RSIIndicator(df["close"], window=14).rsi()
    rsi_closed = rsi.iloc[-2]
    rsi_prev   = rsi.iloc[-3]
    rsi_prev4  = rsi.iloc[-6]

    short_slope  = rsi_closed - rsi_prev
    long_slope   = rsi_closed - rsi_prev4
    acceleration = short_slope - (rsi.iloc[-3] - rsi.iloc[-4])

    # RSI rising below 50 and live crossing above
    if rsi_prev < rsi_closed < 50 and live_rsi > 50:
        result["score"] += 2
        result["details"].append("RSI crossing 50")

    if short_slope >= 1:
        result["score"] += 1
        result["details"].append(f"short slope +{round(short_slope,1)}")

    if long_slope >= 3:
        result["score"] += 1
        result["details"].append(f"long slope +{round(long_slope,1)}")

    if acceleration > 0:
        result["score"] += 1
        result["details"].append("RSI accelerating")

    # MACD cross
    macd     = MACD(df["close"])
    macd_val = macd.macd().iloc[-2]
    signal   = macd.macd_signal().iloc[-2]
    hist     = macd.macd_diff().iloc[-2]
    hist_prev = macd.macd_diff().iloc[-3]

    if macd_val > signal and hist > hist_prev:
        result["score"] += 2
        result["details"].append("MACD bullish cross")
    elif hist > 0:
        result["score"] += 1
        result["details"].append("MACD positive")

    result["rsi_closed"]   = round(rsi_closed, 2)
    result["rsi_prev"]     = round(rsi_prev, 2)
    result["short_slope"]  = round(short_slope, 2)
    result["long_slope"]   = round(long_slope, 2)
    result["acceleration"] = round(acceleration, 2)

    return result   # max score: 8

# ─────────────────────────────────────────
# DIMENSION 3: Volume Structure
# Institutional accumulation vs retail hype
# ─────────────────────────────────────────
def analyze_volume(df):
    result = {"score": 0, "details": []}

    avg_vol  = df["volume"].tail(20).mean()
    last_vol = df["volume"].iloc[-2]
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0

    # volume spike on rising candle = accumulation
    last_close = df["close"].iloc[-2]
    last_open  = df["open"].iloc[-2]
    green_candle = last_close > last_open

    if vol_ratio >= 2.0 and green_candle:
        result["score"] += 3
        result["details"].append(f"vol spike {round(vol_ratio,1)}x on green")
    elif vol_ratio >= 1.5 and green_candle:
        result["score"] += 2
        result["details"].append(f"vol surge {round(vol_ratio,1)}x")
    elif vol_ratio >= 1.2:
        result["score"] += 1
        result["details"].append(f"vol above avg {round(vol_ratio,1)}x")

    # taker buy ratio — buyers hitting the ask = aggressive buying
    taker_buy  = df["taker_base"].iloc[-2]
    total_vol  = df["volume"].iloc[-2]
    buy_ratio  = taker_buy / total_vol if total_vol > 0 else 0.5

    if buy_ratio >= 0.65:
        result["score"] += 2
        result["details"].append(f"taker buy {round(buy_ratio*100)}%")
    elif buy_ratio >= 0.55:
        result["score"] += 1
        result["details"].append(f"buy pressure {round(buy_ratio*100)}%")

    # rising volume over last 3 candles
    v1 = df["volume"].iloc[-4]
    v2 = df["volume"].iloc[-3]
    v3 = df["volume"].iloc[-2]
    if v3 > v2 > v1:
        result["score"] += 1
        result["details"].append("volume expanding")

    result["vol_ratio"]  = round(vol_ratio, 2)
    result["buy_ratio"]  = round(buy_ratio, 2)
    result["avg_vol"]    = round(avg_vol, 0)

    return result   # max score: 6

# ─────────────────────────────────────────
# DIMENSION 4: Price Structure
# Support levels, Bollinger squeeze
# ─────────────────────────────────────────
def analyze_price_structure(df):
    result = {"score": 0, "details": []}

    price = df["close"].iloc[-2]

    # Bollinger Bands
    bb    = BollingerBands(df["close"], window=20, window_dev=2)
    bb_lo = bb.bollinger_lband().iloc[-2]
    bb_hi = bb.bollinger_hband().iloc[-2]
    bb_ma = bb.bollinger_mavg().iloc[-2]
    bb_width = (bb_hi - bb_lo) / bb_ma

    # price bouncing off lower band
    if price <= bb_lo * 1.02:
        result["score"] += 2
        result["details"].append("near lower BB")

    # BB squeeze — big move coming
    bb_width_prev = (
        (bb.bollinger_hband().iloc[-10] - bb.bollinger_lband().iloc[-10])
        / bb.bollinger_mavg().iloc[-10]
    )
    if bb_width < bb_width_prev * 0.7:
        result["score"] += 2
        result["details"].append("BB squeeze")

    # ATR — volatility is manageable
    atr     = AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    atr_val = atr.average_true_range().iloc[-2]
    atr_pct = atr_val / price * 100

    if atr_pct <= 3:
        result["score"] += 1
        result["details"].append(f"ATR {round(atr_pct,1)}% controlled")

    # higher lows structure (last 5 candle lows rising)
    lows = df["low"].iloc[-6:-1].tolist()
    if all(lows[i] <= lows[i+1] for i in range(len(lows)-1)):
        result["score"] += 2
        result["details"].append("higher lows structure")

    result["bb_width"] = round(bb_width * 100, 2)
    result["atr_pct"]  = round(atr_pct, 2)

    return result   # max score: 7

# ─────────────────────────────────────────
# DIMENSION 5: Relative Strength vs BTC
# Coin stronger than BTC = leadership
# ─────────────────────────────────────────
def analyze_relative_strength(symbol, df):
    result = {"score": 0, "details": [], "rs": None}

    try:
        btc_klines = get_klines("BTCUSDT", interval="1h", limit=50)
        if not btc_klines:
            return result

        btc_df   = build_df(btc_klines)
        # % change over last 24 candles
        coin_chg = (df["close"].iloc[-2] - df["close"].iloc[-26]) / df["close"].iloc[-26] * 100
        btc_chg  = (btc_df["close"].iloc[-2] - btc_df["close"].iloc[-26]) / btc_df["close"].iloc[-26] * 100

        rs = coin_chg - btc_chg   # relative strength vs BTC

        if rs >= 5:
            result["score"] += 3
            result["details"].append(f"outperforming BTC by +{round(rs,1)}%")
        elif rs >= 2:
            result["score"] += 2
            result["details"].append(f"beating BTC +{round(rs,1)}%")
        elif rs >= 0:
            result["score"] += 1
            result["details"].append("in line with BTC")

        result["rs"]        = round(rs, 2)
        result["coin_chg"]  = round(coin_chg, 2)
        result["btc_chg"]   = round(btc_chg, 2)

    except Exception:
        pass

    return result   # max score: 3

# ─────────────────────────────────────────
# Risk/Reward Calculator
# Professional entry discipline
# ─────────────────────────────────────────
def calculate_rr(df, current_price):
    # support = recent swing low (last 20 candles)
    support  = df["low"].tail(20).min()
    # resistance = recent swing high (last 20 candles)
    resistance = df["high"].tail(20).max()

    stop_loss  = support * 0.995        # just below support
    target     = current_price + (current_price - stop_loss) * 2  # 2:1 RR

    risk_pct   = (current_price - stop_loss) / current_price * 100
    reward_pct = (target - current_price) / current_price * 100
    rr_ratio   = reward_pct / risk_pct if risk_pct > 0 else 0

    return {
        "stop_loss"  : round(stop_loss, 8),
        "target"     : round(target, 8),
        "support"    : round(support, 8),
        "resistance" : round(resistance, 8),
        "risk_pct"   : round(risk_pct, 2),
        "reward_pct" : round(reward_pct, 2),
        "rr_ratio"   : round(rr_ratio, 2),
    }

# ─────────────────────────────────────────
# Final Grade
# ─────────────────────────────────────────
def grade(total_score, max_score=30):
    pct = total_score / max_score * 100
    if pct >= 80:   return "A  — Strong buy"
    if pct >= 65:   return "B  — Good setup"
    if pct >= 50:   return "C  — Watchlist"
    if pct >= 35:   return "D  — Weak signal"
    return          "F  — Skip"

# ─────────────────────────────────────────
# Process symbol — full institutional analysis
# ─────────────────────────────────────────
def process_symbol(symbol, btc_trend):
    try:
        # skip if market is risk-off
        if btc_trend == "risk_off":
            return None

        klines = get_klines(symbol, interval="1h", limit=200)
        if not klines or len(klines) < 150:
            return None

        df = build_df(klines)

        # base volume filter
        if df["volume"].tail(10).mean() < 50000:
            return None

        # live RSI quick pre-filter (cheap check first)
        live_rsi, current_price = get_live_rsi(symbol, klines)
        if live_rsi is None:
            return None

        if not (45 <= live_rsi <= 65):
            return None

        # ── run all 5 dimensions ──
        trend   = analyze_trend(df)
        momentum = analyze_momentum(df, live_rsi)
        volume  = analyze_volume(df)
        structure = analyze_price_structure(df)
        rs      = analyze_relative_strength(symbol, df)

        # minimum momentum gate — must have RSI signal
        if momentum["score"] < 3:
            return None

        # minimum trend gate — must be somewhat in uptrend
        if trend["score"] < 2:
            return None

        total_score = (
            trend["score"] +
            momentum["score"] +
            volume["score"] +
            structure["score"] +
            rs["score"]
        )

        signal_grade = grade(total_score)

        # skip weak grades
        if signal_grade.startswith("F") or signal_grade.startswith("D"):
            return None

        # risk/reward
        rr = calculate_rr(df, current_price)

        # skip if risk/reward is bad
        if rr["rr_ratio"] < 1.5:
            return None

        # cooldown
        if symbol in last_alert_rsi:
            if abs(live_rsi - last_alert_rsi[symbol]) < 3:
                return None
        last_alert_rsi[symbol] = live_rsi

        return {
            "symbol"       : symbol,
            "grade"        : signal_grade,
            "total_score"  : total_score,
            "trend_score"  : trend["score"],
            "momentum_score": momentum["score"],
            "volume_score" : volume["score"],
            "structure_score": structure["score"],
            "rs_score"     : rs["score"],
            "trend_details": ", ".join(trend["details"]),
            "mom_details"  : ", ".join(momentum["details"]),
            "vol_details"  : ", ".join(volume["details"]),
            "str_details"  : ", ".join(structure["details"]),
            "rs_details"   : ", ".join(rs["details"]),
            "live_rsi"     : live_rsi,
            "rsi_closed"   : momentum["rsi_closed"],
            "vol_ratio"    : volume["vol_ratio"],
            "buy_ratio"    : volume["buy_ratio"],
            "rs_vs_btc"    : rs.get("rs"),
            "current_price": current_price,
            "stop_loss"    : rr["stop_loss"],
            "target"       : rr["target"],
            "risk_pct"     : rr["risk_pct"],
            "reward_pct"   : rr["reward_pct"],
            "rr_ratio"     : rr["rr_ratio"],
            "btc_trend"    : btc_trend,
        }

    except Exception as e:
        logger.error(f"{symbol} error: {e}")
        return None

# ─────────────────────────────────────────
# Scan
# ─────────────────────────────────────────
def scan():
    logger.info("Starting institutional scan")

    # check market context first
    btc_trend = get_btc_trend()
    logger.info(f"Market context: BTC trend = {btc_trend}")

    if btc_trend == "risk_off":
        print("Market is RISK OFF — no trades. Waiting.\n")
        return

    symbols = get_all_usdt_symbols()
    matches = []
    start   = time.time()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_symbol, s, btc_trend) for s in symbols]
        for future in as_completed(futures):
            result = future.result()
            if result:
                matches.append(result)

    # sort by total score descending
    matches.sort(key=lambda x: x["total_score"], reverse=True)

    print(f"\nScan done in {round(time.time() - start, 2)}s")

    if matches:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n{'='*90}")
        print(f"  INSTITUTIONAL RSI SCAN — {ts}  |  BTC: {btc_trend.upper()}")
        print(f"{'='*90}")

        for m in matches:
            print(f"""
{m['grade']}  |  {m['symbol']}  |  Score: {m['total_score']}/30  |  Live RSI: {m['live_rsi']}
  Trend     [{m['trend_score']}/6]  : {m['trend_details']}
  Momentum  [{m['momentum_score']}/8]  : {m['mom_details']}
  Volume    [{m['volume_score']}/6]  : {m['vol_details']}
  Structure [{m['structure_score']}/7]  : {m['str_details']}
  Rel.Str   [{m['rs_score']}/3]  : {m['rs_details']}
  Entry: {m['current_price']}  |  Stop: {m['stop_loss']} (-{m['risk_pct']}%)  |  Target: {m['target']} (+{m['reward_pct']}%)  |  RR: {m['rr_ratio']}:1
""")

        print(f"{'='*90}")
        print(f"Total signals: {len(matches)}\n")

        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*90}\n")
            f.write(f"INSTITUTIONAL SCAN — {ts}  |  BTC: {btc_trend}\n")
            f.write(f"{'='*90}\n")
            for m in matches:
                f.write(
                    f"{m['grade']} | {m['symbol']} | Score:{m['total_score']}/30 | "
                    f"RSI:{m['live_rsi']} | Vol:{m['vol_ratio']}x | "
                    f"BuyRatio:{m['buy_ratio']} | RS:{m['rs_vs_btc']}% | "
                    f"Entry:{m['current_price']} Stop:{m['stop_loss']} "
                    f"Target:{m['target']} RR:{m['rr_ratio']}:1\n"
                    f"  → {m['trend_details']} | {m['mom_details']} | "
                    f"{m['vol_details']} | {m['str_details']}\n\n"
                )
    else:
        print("No qualifying signals.\n")

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Institutional scanner started")
    while True:
        scan()
        logger.info("Sleeping 15 minutes...\n")
        time.sleep(900)