"""
RSI scanner V3 — logic lives in ``RSIScannerV3`` only; ``rsi_scannerv2.py`` unchanged.

Run:
  python rsi_scannerv3.py

Strategy modes (``RSI_STRATEGY_MODE``):
  cross_above (default) — RSI **above** ``RSI_TARGET`` (default 55), optional **fresh cross**
    (``RSI_REQUIRE_FRESH_CROSS``: prior close **at/below** target, current **above**), optional
    ``min(lookback) < target``, optional **last bar rising**, optional **RSI_MOMENTUM_STEPS** (strict
    rise over the last N closes), then optional TV Smoothed HA **lime** (green candle).
  strict_rise — previous behaviour: strict rising tail + RSI_LO band (see env below).

Env (optional):
  BINANCE_API, BINANCE_FALLBACK_BASE_URLS (comma list; ``off`` = none), BINANCE_MIRROR_RETRY_DELAY_S,
  BINANCE_USER_AGENT, REQUEST_READ_TIMEOUT_S (30),
  INTERVAL — **only ``1h`` or ``30m``** for RSI. **Default ``1h``** (preferred). Set ``INTERVAL=30m`` only
    when you explicitly want 30‑minute RSI. Invalid values (e.g. ``15m``, ``5m``) → warning and **1h**.
  CANDLE_LIMIT,
  RSI_PERIOD (14) — Wilder RSI length (TradingView / v2 default 14),
  RSI_STRATEGY_MODE (cross_above | strict_rise),
  RSI_TARGET (55), RSI_LOOKBACK_BARS (6), RSI_REQUIRE_BELOW_IN_LOOKBACK (1),
  RSI_REQUIRE_LAST_RISING (1), RSI_REQUIRE_FRESH_CROSS (0) — prior RSI<=target & now>target,
  RSI_MOMENTUM_STEPS (0) — require RSI strictly rising over last N steps (1 step = now>prev),
  RSI_HI (optional cap on latest RSI),
  For strict_rise only: RSI_LO, RSI_RISE_BARS, RSI_REQUIRE_ALL_BARS_ABOVE_LO, RISE_EPSILON,
  SCAN_SLEEP_S (600) — seconds between full scan rounds (10 minutes default),
  REQUEST_DELAY_S, DROP_OPEN_CANDLE, RUN_ONCE, MAX_SYMBOLS, LOG_LEVEL,
  REQUIRE_SMOOTHED_HA_GREEN,
  HA_PIPELINE (tv_smoothed | legacy), HA_TV_LEN1 (10), HA_TV_LEN2 (10), HA_SMOOTH_SPAN (legacy only, 5),
  RSI_BAR_UP_LEVEL (70), RSI_BAR_DOWN_LEVEL (30),
  REQUIRE_RSI_BAR_COLOR (off | green | red | not_red | not_green) — optional extra gate on matches,
  TAKE_PROFIT_PCT (1) — suggested sell = buy_reference × (1 + pct/100); buy_reference = last closed close.

Illustrative levels only (not execution advice). Fees / spread / slippage not included.

Note: RSI is computed on a **single** kline interval per run (``1h`` **preferred** default, or ``30m``
if you set ``INTERVAL=30m``). Invalid ``INTERVAL`` values fall back to ``1h``. The outer loop can still
sleep ``SCAN_SLEEP_S`` between full scans.

HTTP 451 / 403: Binance blocks some regions. Set ``BINANCE_API=https://api.binance.us`` (US) or
another spot REST host. Optional ``BINANCE_FALLBACK_BASE_URLS`` (comma list; ``off`` to disable).

RSI uses **Wilder / RMA smoothing** (same family as TradingView’s built-in RSI). TV can still
differ slightly if your chart uses non-default OHLC (Heikin Ashi candles, different session, etc.).

Smoothed HA (``HA_PIPELINE``):
  * ``tv_smoothed`` (default) — matches the common “Smoothed Heiken Ashi Candles” flow: EMA(len1) on each
    OHLC, build HA from those series, then EMA(len2) on ha open/high/low/close; **lime** candle =
    ``o2 <= c2`` (same as Pine ``col = o2 > c2 ? red : lime``).
  * ``legacy`` — older path: classic HA on raw OHLC, then EMA only on ha_open / ha_close.

RSI “chart bars” (Glaz-style, optional filter ``REQUIRE_RSI_BAR_COLOR``): **green** tint = RSI above
``RSI_BAR_UP_LEVEL`` (70), **red** = RSI below ``RSI_BAR_DOWN_LEVEL`` (30), else **neutral** — same
idea as ``barcolor(isup() ? green : isdown() ? red : na)`` (we always report the zone; filter is optional).
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

_BINANCE_GEO_WARNED = False

# Binance RSI timeframe: only these intervals are supported (no 15m / 5m RSI in this scanner).
_ALLOWED_RSI_INTERVALS = frozenset({"1h", "30m"})


def _interval_from_env() -> str:
    raw = (os.environ.get("INTERVAL") or "1h").strip().lower()
    if raw in _ALLOWED_RSI_INTERVALS:
        return raw
    warnings.warn(
        f"INTERVAL={raw!r} is not supported; only 1h and 30m are allowed for RSI. Using 1h (preferred).",
        UserWarning,
        stacklevel=2,
    )
    return "1h"


def _parse_optional_float_env(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw.lower() in ("", "none", "off"):
        return None
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_fallback_base_urls(raw: Optional[str]) -> Tuple[str, ...]:
    if raw is None:
        raw = "https://api1.binance.com,https://api2.binance.com,https://api3.binance.com"
    raw = raw.strip()
    if raw.lower() in ("", "off", "none", "0"):
        return ()
    return tuple(x.strip().rstrip("/") for x in raw.split(",") if x.strip())


def _dedupe_bases(primary: str, fallbacks: Sequence[str]) -> List[str]:
    out: List[str] = []
    for u in [primary.rstrip("/")] + [x.rstrip("/") for x in fallbacks]:
        if u and u not in out:
            out.append(u)
    return out


def wilder_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI aligned with the usual TradingView / Wilder definition (RMA on gains/losses).
    """
    c = close.astype(float).to_numpy()
    n = len(c)
    rsi = np.full(n, np.nan, dtype=float)
    if period <= 0 or n < period + 1:
        return pd.Series(rsi, index=close.index)

    changes = np.diff(c)
    gains = np.maximum(changes, 0.0)
    losses = np.maximum(-changes, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    def rsi_from_avgs(ag: float, al: float) -> float:
        if al == 0.0:
            return 100.0 if ag > 0.0 else 50.0
        rs = ag / al
        return 100.0 - (100.0 / (1.0 + rs))

    rsi[period] = rsi_from_avgs(avg_gain, avg_loss)
    for close_idx in range(period + 1, n):
        ch_idx = close_idx - 1
        avg_gain = (avg_gain * (period - 1) + gains[ch_idx]) / period
        avg_loss = (avg_loss * (period - 1) + losses[ch_idx]) / period
        rsi[close_idx] = rsi_from_avgs(avg_gain, avg_loss)

    return pd.Series(rsi, index=close.index)


def rsi_chart_bar_color(rsi_val: float, up_level: float, down_level: float) -> str:
    """Glaz-style RSI chart bars: green above up_level, red below down_level, else neutral."""
    if rsi_val > up_level:
        return "green"
    if rsi_val < down_level:
        return "red"
    return "neutral"


def rsi_bar_filter_allows(
    rsi_val: float, requirement: str, up_level: float, down_level: float
) -> bool:
    tag = rsi_chart_bar_color(rsi_val, up_level, down_level)
    if requirement == "green":
        return tag == "green"
    if requirement == "red":
        return tag == "red"
    if requirement == "not_red":
        return tag != "red"
    if requirement == "not_green":
        return tag != "green"
    return True


@dataclass
class RSIScannerV3Config:
    binance_api: str = "https://api.binance.com"
    binance_fallback_bases: Tuple[str, ...] = ()
    binance_mirror_retry_delay_s: float = 0.35
    request_read_timeout_s: float = 30.0
    interval: str = "1h"
    candle_limit: int = 120
    rsi_period: int = 14
    # cross_above (default)
    rsi_strategy_mode: str = "cross_above"
    rsi_target_level: float = 55.0
    rsi_lookback_bars: int = 6
    rsi_require_below_target_in_lookback: bool = True
    rsi_require_last_rising: bool = True
    rsi_require_fresh_cross: bool = False  # cross_above: prior bar at/below target, now above
    rsi_momentum_steps: int = 0  # cross_above: >0 = require RSI rising over last N steps (strict)
    rsi_hi: Optional[float] = None  # optional exclusive cap on latest RSI
    # strict_rise only
    rsi_lo: float = 53.0
    rsi_rise_bars: int = 6
    rsi_require_all_bars_above_lo: bool = True
    rise_epsilon: float = 0.0
    # loop / IO
    scan_sleep_s: float = 600.0
    request_delay_s: float = 0.06
    drop_open_candle: bool = True
    max_symbols: int = 0
    require_smoothed_ha_green: bool = True
    # HA: TradingView “Smoothed Heiken Ashi” (tv_smoothed) vs older legacy path
    ha_pipeline: str = "tv_smoothed"
    ha_tv_len1: int = 10
    ha_tv_len2: int = 10
    ha_smooth_span: int = 5  # legacy only: EMA span on ha_open / ha_close
    # Glaz-style RSI bar tint zones (overlay indicator); optional scan filter
    rsi_bar_up_level: float = 70.0
    rsi_bar_down_level: float = 30.0
    require_rsi_bar_color: Optional[str] = None  # off | green | red | not_red | not_green
    # illustrative buy/sell from last closed candle (not trading advice)
    take_profit_pct: float = 1.0

    @classmethod
    def from_env(cls) -> "RSIScannerV3Config":
        mode = os.environ.get("RSI_STRATEGY_MODE", "cross_above").strip().lower()
        if mode not in ("cross_above", "strict_rise"):
            mode = "cross_above"
        ha_pipe = os.environ.get("HA_PIPELINE", "tv_smoothed").strip().lower()
        if ha_pipe not in ("tv_smoothed", "legacy"):
            ha_pipe = "tv_smoothed"
        rsi_bar_filter = os.environ.get("REQUIRE_RSI_BAR_COLOR", "").strip().lower()
        if rsi_bar_filter in ("", "0", "off", "none", "false"):
            rsi_bar_filter_val: Optional[str] = None
        elif rsi_bar_filter in ("green", "red", "not_red", "not_green"):
            rsi_bar_filter_val = rsi_bar_filter
        else:
            rsi_bar_filter_val = None
        return cls(
            binance_api=os.environ.get("BINANCE_API", "https://api.binance.com").rstrip("/"),
            binance_fallback_bases=_parse_fallback_base_urls(os.environ.get("BINANCE_FALLBACK_BASE_URLS")),
            binance_mirror_retry_delay_s=float(os.environ.get("BINANCE_MIRROR_RETRY_DELAY_S", "0.35")),
            request_read_timeout_s=float(os.environ.get("REQUEST_READ_TIMEOUT_S", "30")),
            interval=_interval_from_env(),
            candle_limit=int(os.environ.get("CANDLE_LIMIT", "120")),
            rsi_period=max(2, int(os.environ.get("RSI_PERIOD", "14"))),
            rsi_strategy_mode=mode,
            rsi_target_level=float(os.environ.get("RSI_TARGET", "55")),
            rsi_lookback_bars=max(3, int(os.environ.get("RSI_LOOKBACK_BARS", "6"))),
            rsi_require_below_target_in_lookback=_env_bool("RSI_REQUIRE_BELOW_IN_LOOKBACK", True),
            rsi_require_last_rising=_env_bool("RSI_REQUIRE_LAST_RISING", True),
            rsi_require_fresh_cross=_env_bool("RSI_REQUIRE_FRESH_CROSS", False),
            rsi_momentum_steps=max(0, min(30, int(os.environ.get("RSI_MOMENTUM_STEPS", "0")))),
            rsi_hi=_parse_optional_float_env("RSI_HI", None),
            rsi_lo=float(os.environ.get("RSI_LO", "53")),
            rsi_rise_bars=max(2, int(os.environ.get("RSI_RISE_BARS", "6"))),
            rsi_require_all_bars_above_lo=_env_bool("RSI_REQUIRE_ALL_BARS_ABOVE_LO", True),
            rise_epsilon=float(os.environ.get("RISE_EPSILON", "0")),
            scan_sleep_s=float(os.environ.get("SCAN_SLEEP_S", "600")),
            request_delay_s=float(os.environ.get("REQUEST_DELAY_S", "0.06")),
            drop_open_candle=_env_bool("DROP_OPEN_CANDLE", True),
            max_symbols=int(os.environ.get("MAX_SYMBOLS", "0")),
            require_smoothed_ha_green=_env_bool("REQUIRE_SMOOTHED_HA_GREEN", True),
            ha_pipeline=ha_pipe,
            ha_tv_len1=max(1, int(os.environ.get("HA_TV_LEN1", "10"))),
            ha_tv_len2=max(1, int(os.environ.get("HA_TV_LEN2", "10"))),
            ha_smooth_span=max(2, int(os.environ.get("HA_SMOOTH_SPAN", "5"))),
            rsi_bar_up_level=float(os.environ.get("RSI_BAR_UP_LEVEL", "70")),
            rsi_bar_down_level=float(os.environ.get("RSI_BAR_DOWN_LEVEL", "30")),
            require_rsi_bar_color=rsi_bar_filter_val,
            take_profit_pct=max(0.01, float(os.environ.get("TAKE_PROFIT_PCT", "1"))),
        )


def _ha_scan_log_suffix(c: RSIScannerV3Config) -> str:
    if not c.require_smoothed_ha_green:
        return "ha=off"
    if c.ha_pipeline == "legacy":
        return f"ha=legacy_ewm_oc@{c.ha_smooth_span}"
    return f"ha=tv_smoothed@{c.ha_tv_len1}/{c.ha_tv_len2}"


class RSIScannerV3:
    """Binance spot USDT: RSI + optional Smoothed HA (see config)."""

    KLINES_COLUMNS = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "qav",
        "trades",
        "taker_base",
        "taker_quote",
        "ignore",
    ]

    def __init__(self, config: Optional[RSIScannerV3Config] = None) -> None:
        self.cfg = config or RSIScannerV3Config.from_env()
        self._log = logging.getLogger(self.__class__.__name__)
        self._base_urls = _dedupe_bases(self.cfg.binance_api, self.cfg.binance_fallback_bases)
        self._request_headers = {
            "User-Agent": os.environ.get(
                "BINANCE_USER_AGENT",
                "Mozilla/5.0 (compatible; RSIScannerV3/1.0; +https://www.binance.com/en/support)",
            ),
            "Accept": "application/json",
        }
        if len(self._base_urls) > 1:
            self._log.info("Binance REST bases (try in order): %s", self._base_urls)

    def _http_get_json(self, path: str, params: Dict[str, Any]) -> Optional[Any]:
        global _BINANCE_GEO_WARNED
        timeout = (10.0, self.cfg.request_read_timeout_s)
        saw_block = False
        for base in self._base_urls:
            url = f"{base}{path}"
            try:
                r = requests.get(url, params=params, headers=self._request_headers, timeout=timeout)
            except requests.Timeout as e:
                self._log.warning("Binance timeout %s %s: %s", path, params.get("symbol", ""), e)
                return None
            except requests.RequestException as e:
                self._log.warning("Binance request error %s: %s", url, e)
                return None
            if r.status_code in (451, 403):
                saw_block = True
                self._log.debug("HTTP %s from %s (%s)", r.status_code, base, params.get("symbol", path))
                if self.cfg.binance_mirror_retry_delay_s > 0:
                    time.sleep(self.cfg.binance_mirror_retry_delay_s)
                continue
            if r.status_code != 200:
                self._log.warning("HTTP %s for %s", r.status_code, params.get("symbol", path))
                return None
            try:
                return r.json()
            except ValueError:
                self._log.warning("Invalid JSON from %s", url)
                return None
        if saw_block and not _BINANCE_GEO_WARNED:
            _BINANCE_GEO_WARNED = True
            self._log.warning(
                "Binance returned HTTP 451/403 from every host tried — regional / compliance block. "
                "Set BINANCE_API=https://api.binance.us (US) or another Binance spot REST base."
            )
        return None

    def get_all_usdt_symbols(self) -> List[str]:
        self._log.info("Fetching USDT spot symbols")
        data = self._http_get_json("/api/v3/exchangeInfo", {"permissions": "SPOT"})
        if not data:
            raise RuntimeError("exchangeInfo failed from all Binance bases — check network / BINANCE_API")
        out: List[str] = []
        for s in data.get("symbols", []):
            if s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
                continue
            if s.get("isSpotTradingAllowed") is False:
                continue
            sym = s.get("symbol")
            if isinstance(sym, str):
                out.append(sym)
        out.sort()
        if self.cfg.max_symbols > 0:
            out = out[: self.cfg.max_symbols]
        self._log.info("Fetched %s symbols", len(out))
        return out

    def get_klines(self, symbol: str) -> Optional[List[Any]]:
        params = {"symbol": symbol, "interval": self.cfg.interval, "limit": self.cfg.candle_limit}
        return self._http_get_json("/api/v3/klines", params)

    def calculate_rsi(self, close: pd.Series) -> pd.Series:
        return wilder_rsi_series(close, self.cfg.rsi_period)

    @staticmethod
    def _fmt_price(x: float) -> str:
        if x >= 1000:
            return f"{x:.2f}"
        if x >= 1:
            s = f"{x:.6f}".rstrip("0").rstrip(".")
            return s if s else "0"
        s = f"{x:.8f}".rstrip("0").rstrip(".")
        return s if s else "0"

    @staticmethod
    def _fmt_ha_sigma(hsc: float, hso: float) -> str:
        """Smoothed HA line for printing; includes Δ so tiny greens are not hidden as 0.0033>0.0033."""
        d = float(hsc) - float(hso)
        mag = max(abs(hsc), abs(hso))
        if mag >= 500:
            return f"HAσ {hsc:.2f}>{hso:.2f} (Δ={d:+.5g})"
        if mag >= 0.05:
            return f"HAσ {hsc:.6f}>{hso:.6f} (Δ={d:+.6g})"
        return f"HAσ {hsc:.10f}>{hso:.10f} (Δ={d:+.2e})"

    @staticmethod
    def _price_levels(last_close: float, take_profit_pct: float) -> Tuple[float, float, float]:
        """buy_reference, sell_target, profit_abs (quote per 1 base unit, before fees)."""
        buy = float(last_close)
        mult = 1.0 + take_profit_pct / 100.0
        sell = buy * mult
        profit_abs = sell - buy
        return buy, sell, profit_abs

    @staticmethod
    def _strictly_rising_tail(values: Sequence[float], eps: float) -> bool:
        if len(values) < 2:
            return False
        for i in range(1, len(values)):
            if not (values[i] + eps > values[i - 1]):
                return False
        return True

    @staticmethod
    def heikin_ashi_raw(df: pd.DataFrame) -> pd.DataFrame:
        o = df["open"].astype(float).to_numpy()
        h = df["high"].astype(float).to_numpy()
        low = df["low"].astype(float).to_numpy()
        c = df["close"].astype(float).to_numpy()
        n = len(df)
        ha_close = np.empty(n, dtype=float)
        ha_open = np.empty(n, dtype=float)
        ha_high = np.empty(n, dtype=float)
        ha_low = np.empty(n, dtype=float)
        for i in range(n):
            ha_close[i] = (o[i] + h[i] + low[i] + c[i]) / 4.0
            if i == 0:
                ha_open[i] = (o[i] + c[i]) / 2.0
            else:
                ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
            ha_high[i] = max(h[i], ha_open[i], ha_close[i])
            ha_low[i] = min(low[i], ha_open[i], ha_close[i])
        return pd.DataFrame(
            {"ha_open": ha_open, "ha_high": ha_high, "ha_low": ha_low, "ha_close": ha_close},
            index=df.index,
        )

    def smoothed_heikin_ashi_color(self, df: pd.DataFrame) -> Tuple[bool, float, float, bool]:
        """
        HA gate for matches. Returns (pass_green, display_open, display_close, raw_stage1_green).

        tv_smoothed: display_open/close are o2/c2 (second EMA layer); pass = TV lime (o2 <= c2).
        legacy: display = EMA(ha_open), EMA(ha_close); pass = close > open (strict).
        """
        if self.cfg.ha_pipeline == "legacy":
            ha = self.heikin_ashi_raw(df)
            span = self.cfg.ha_smooth_span
            s_open = ha["ha_open"].ewm(span=span, adjust=False).mean()
            s_close = ha["ha_close"].ewm(span=span, adjust=False).mean()
            raw_green = bool(ha["ha_close"].iloc[-1] > ha["ha_open"].iloc[-1])
            sc_last = s_close.iloc[-1]
            so_last = s_open.iloc[-1]
            if pd.isna(sc_last) or pd.isna(so_last):
                return False, float("nan"), float("nan"), raw_green
            sc = float(sc_last)
            so = float(so_last)
            green = sc > so
            return green, sc, so, raw_green
        return self._tv_smoothed_heikin_ashi_color(df, self.cfg.ha_tv_len1, self.cfg.ha_tv_len2)

    @staticmethod
    def _tv_smoothed_heikin_ashi_color(
        df: pd.DataFrame, len1: int, len2: int
    ) -> Tuple[bool, float, float, bool]:
        """TradingView pipeline: EMA(len1) on OHLC -> HA -> EMA(len2) on HA OHLC; lime when o2 <= c2."""
        o = df["open"].astype(float)
        h = df["high"].astype(float)
        low = df["low"].astype(float)
        c = df["close"].astype(float)
        eo = o.ewm(span=len1, adjust=False).mean()
        eh = h.ewm(span=len1, adjust=False).mean()
        el = low.ewm(span=len1, adjust=False).mean()
        ec = c.ewm(span=len1, adjust=False).mean()

        eo_np = eo.to_numpy(dtype=np.float64, copy=False)
        eh_np = eh.to_numpy(dtype=np.float64, copy=False)
        el_np = el.to_numpy(dtype=np.float64, copy=False)
        ec_np = ec.to_numpy(dtype=np.float64, copy=False)
        n = len(df)
        ha_close = np.empty(n, dtype=np.float64)
        ha_open = np.empty(n, dtype=np.float64)
        ha_high = np.empty(n, dtype=np.float64)
        ha_low = np.empty(n, dtype=np.float64)
        for i in range(n):
            ha_close[i] = (eo_np[i] + eh_np[i] + el_np[i] + ec_np[i]) * 0.25
            if i == 0:
                ha_open[i] = (eo_np[i] + ec_np[i]) * 0.5
            else:
                ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) * 0.5
            ha_high[i] = max(eh_np[i], ha_open[i], ha_close[i])
            ha_low[i] = min(el_np[i], ha_open[i], ha_close[i])

        idx = df.index
        ha_open_s = pd.Series(ha_open, index=idx)
        ha_high_s = pd.Series(ha_high, index=idx)
        ha_low_s = pd.Series(ha_low, index=idx)
        ha_close_s = pd.Series(ha_close, index=idx)

        o2 = ha_open_s.ewm(span=len2, adjust=False).mean()
        c2 = ha_close_s.ewm(span=len2, adjust=False).mean()

        o2_last = float(o2.iloc[-1])
        c2_last = float(c2.iloc[-1])
        if o2_last != o2_last or c2_last != c2_last:
            return False, float("nan"), float("nan"), False

        raw_green = bool(ha_close[-1] > ha_open[-1])
        lime = o2_last <= c2_last
        return lime, c2_last, o2_last, raw_green

    def _min_rows_needed(self) -> int:
        p = self.cfg.rsi_period
        if self.cfg.rsi_strategy_mode == "strict_rise":
            n = self.cfg.rsi_rise_bars
            min_rsi = p + n + 2
        else:
            n = self.cfg.rsi_lookback_bars
            n = max(n, self.cfg.rsi_momentum_steps + 1, 3)
            min_rsi = p + n + 2
        if not self.cfg.require_smoothed_ha_green:
            return min_rsi
        if self.cfg.ha_pipeline == "legacy":
            ha_extra = 30 + self.cfg.ha_smooth_span * 2
        else:
            ha_extra = 40 + self.cfg.ha_tv_len1 * 3 + self.cfg.ha_tv_len2 * 3
        return max(min_rsi, ha_extra)

    @staticmethod
    def _rsi_tail_strictly_rising_end(rsi_tail: Sequence[float], steps: int) -> bool:
        """Last (steps+1) RSI closes strictly increase (steps>=1 → now>prev>…). steps<=0: True."""
        if steps <= 0:
            return True
        if len(rsi_tail) < steps + 1:
            return False
        seg = rsi_tail[-(steps + 1) :]
        for i in range(1, len(seg)):
            if seg[i] <= seg[i - 1]:
                return False
        return True

    def scan_one(self, symbol: str) -> Optional[dict]:
        min_rows = self._min_rows_needed()
        klines = self.get_klines(symbol)
        if not klines or len(klines) < min_rows:
            return None

        df = pd.DataFrame(klines, columns=self.KLINES_COLUMNS)
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype(float)

        if self.cfg.drop_open_candle and len(df) > 1:
            df = df.iloc[:-1].copy()

        rsi = self.calculate_rsi(df["close"])

        ok = False
        rsi_now = 0.0
        rsi_tail: List[float] = []

        if self.cfg.rsi_strategy_mode == "strict_rise":
            n_rise = self.cfg.rsi_rise_bars
            tail = rsi.iloc[-n_rise:]
            if tail.isna().any() or len(tail) < n_rise:
                return None
            rsi_tail = [float(x) for x in tail.tolist()]
            rsi_now = rsi_tail[-1]
            rising = self._strictly_rising_tail(rsi_tail, self.cfg.rise_epsilon)
            floor_ok = rsi_now >= self.cfg.rsi_lo
            if self.cfg.rsi_require_all_bars_above_lo and min(rsi_tail) < self.cfg.rsi_lo - 1e-12:
                floor_ok = False
            cap_ok = self.cfg.rsi_hi is None or rsi_now < self.cfg.rsi_hi
            ok = rising and floor_ok and cap_ok
        else:
            L = max(self.cfg.rsi_lookback_bars, self.cfg.rsi_momentum_steps + 1, 3)
            tail = rsi.iloc[-L:]
            if tail.isna().any() or len(tail) < L:
                return None
            rsi_tail = [float(x) for x in tail.tolist()]
            rsi_now = rsi_tail[-1]
            rsi_prev = rsi_tail[-2]
            tgt = self.cfg.rsi_target_level
            ok = False
            if rsi_now <= tgt:
                ok = False
            elif self.cfg.rsi_require_last_rising and rsi_now <= rsi_prev:
                ok = False
            elif self.cfg.rsi_require_fresh_cross and rsi_prev > tgt:
                ok = False
            elif self.cfg.rsi_require_below_target_in_lookback and min(rsi_tail) >= tgt - 1e-12:
                ok = False
            elif self.cfg.rsi_hi is not None and rsi_now >= self.cfg.rsi_hi:
                ok = False
            elif not self._rsi_tail_strictly_rising_end(rsi_tail, self.cfg.rsi_momentum_steps):
                ok = False
            else:
                ok = True

        if not ok:
            return None

        rsi_chart_bar = rsi_chart_bar_color(
            rsi_now, self.cfg.rsi_bar_up_level, self.cfg.rsi_bar_down_level
        )
        if self.cfg.require_rsi_bar_color and not rsi_bar_filter_allows(
            rsi_now,
            self.cfg.require_rsi_bar_color,
            self.cfg.rsi_bar_up_level,
            self.cfg.rsi_bar_down_level,
        ):
            return None

        ha_green: Optional[bool] = None
        ha_smooth_close: Optional[float] = None
        ha_smooth_open: Optional[float] = None
        ha_raw_green: Optional[bool] = None

        if self.cfg.require_smoothed_ha_green:
            ha_green, ha_smooth_close, ha_smooth_open, ha_raw_green = self.smoothed_heikin_ashi_color(df)
            if not ha_green or (ha_smooth_close != ha_smooth_close):
                return None

        ha_label = (
            "green"
            if ha_green is True
            else ("off" if not self.cfg.require_smoothed_ha_green else "red")
        )
        self._log.info(
            "MATCH: %s RSI=%.2f RSIbars=%s HA=%s",
            symbol,
            rsi_now,
            rsi_chart_bar,
            ha_label,
        )

        rsi_fresh_cross_val: Optional[bool] = None
        if self.cfg.rsi_strategy_mode == "cross_above" and len(rsi_tail) >= 2:
            tgt_fc = self.cfg.rsi_target_level
            rsi_fresh_cross_val = bool(rsi_tail[-2] <= tgt_fc and rsi_tail[-1] > tgt_fc)

        last_close = float(df["close"].iloc[-1])
        buy_ref, sell_tgt, profit_abs = self._price_levels(last_close, self.cfg.take_profit_pct)

        return {
            "symbol": symbol,
            "rsi": round(rsi_now, 2),
            "rsi_tail": [round(x, 2) for x in rsi_tail],
            "last_close": last_close,
            "buy_reference": buy_ref,
            "sell_target": sell_tgt,
            "take_profit_pct": self.cfg.take_profit_pct,
            "profit_per_unit": profit_abs,
            "rsi_chart_bar": rsi_chart_bar,
            "ha_pipeline": self.cfg.ha_pipeline,
            "ha_smooth_green": ha_green if self.cfg.require_smoothed_ha_green else None,
            "ha_smooth_close": ha_smooth_close,
            "ha_smooth_open": ha_smooth_open,
            "ha_raw_green": ha_raw_green,
            "rsi_fresh_cross": rsi_fresh_cross_val,
            "rsi_momentum_steps": self.cfg.rsi_momentum_steps
            if self.cfg.rsi_strategy_mode == "cross_above"
            else None,
        }

    def scan(self) -> List[dict]:
        c = self.cfg
        if c.rsi_strategy_mode == "strict_rise":
            hi_s = f"RSI<{c.rsi_hi:g}" if c.rsi_hi is not None else "RSI_hi=(no cap)"
            self._log.info(
                "Scan mode=strict_rise interval=%s limit=%s RSI_lo=%.2f %s rise_bars=%s "
                "all_tail_above_lo=%s eps=%.3f drop_open=%s ha=%s rsi_bar_filter=%s",
                c.interval,
                c.candle_limit,
                c.rsi_lo,
                hi_s,
                c.rsi_rise_bars,
                c.rsi_require_all_bars_above_lo,
                c.rise_epsilon,
                c.drop_open_candle,
                _ha_scan_log_suffix(c),
                c.require_rsi_bar_color or "off",
            )
        else:
            hi_s = f"RSI<{c.rsi_hi:g}" if c.rsi_hi is not None else "RSI_hi=(no cap)"
            self._log.info(
                "Scan mode=cross_above interval=%s limit=%s target=%.2f lookback=%s "
                "below_in_lb=%s last_rising=%s fresh_cross=%s mom_steps=%s %s drop_open=%s %s rsi_bar_filter=%s sleep=%ss",
                c.interval,
                c.candle_limit,
                c.rsi_target_level,
                c.rsi_lookback_bars,
                c.rsi_require_below_target_in_lookback,
                c.rsi_require_last_rising,
                c.rsi_require_fresh_cross,
                c.rsi_momentum_steps,
                hi_s,
                c.drop_open_candle,
                _ha_scan_log_suffix(c),
                c.require_rsi_bar_color or "off",
                c.scan_sleep_s,
            )

        symbols = self.get_all_usdt_symbols()
        matches: List[dict] = []
        for idx, symbol in enumerate(symbols, start=1):
            self._log.debug("[%s/%s] %s", idx, len(symbols), symbol)
            try:
                hit = self.scan_one(symbol)
                if hit:
                    matches.append(hit)
            except Exception as e:
                self._log.error("Unexpected error processing %s: %s", symbol, e, exc_info=True)
            finally:
                if self.cfg.request_delay_s > 0:
                    time.sleep(self.cfg.request_delay_s)
        return matches

    def print_matches(self, matches: List[dict]) -> None:
        if matches:
            c = self.cfg
            if c.require_smoothed_ha_green:
                if c.ha_pipeline == "legacy":
                    ha_note = f" + legacy HA EMA on ha_oc (span={c.ha_smooth_span})"
                else:
                    ha_note = (
                        f" + TV Smoothed HA (EMA {c.ha_tv_len1} OHLC → HA → EMA {c.ha_tv_len2}; "
                        f"lime=c2>=o2)"
                    )
            else:
                ha_note = ""
            rsi_bar_note = ""
            if c.require_rsi_bar_color:
                rsi_bar_note = (
                    f" + RSIbars filter={c.require_rsi_bar_color} "
                    f"(green RSI>{c.rsi_bar_up_level:g}, red RSI<{c.rsi_bar_down_level:g})"
                )
            if c.rsi_strategy_mode == "strict_rise":
                hi = c.rsi_hi
                band = f"RSI>={c.rsi_lo:g} & RSI<{hi:g}" if hi is not None else f"RSI>={c.rsi_lo:g}"
                tail_note = (
                    f" (each of last {c.rsi_rise_bars} >= {c.rsi_lo:g})"
                    if c.rsi_require_all_bars_above_lo
                    else ""
                )
                title = f"{band} + last {c.rsi_rise_bars} RSI strictly rising{tail_note}{ha_note}{rsi_bar_note}"
            else:
                hi = c.rsi_hi
                cap = f" & RSI<{hi:g}" if hi is not None else ""
                below = "min lookback was < target" if c.rsi_require_below_target_in_lookback else "no below requirement"
                rise = " + last RSI > prior" if c.rsi_require_last_rising else ""
                fresh_s = " + fresh cross (prev≤target & now>target)" if c.rsi_require_fresh_cross else ""
                mom_s = (
                    f" + RSI momentum {c.rsi_momentum_steps} step(s) strictly up"
                    if c.rsi_momentum_steps >= 1
                    else ""
                )
                eff_lb = max(c.rsi_lookback_bars, c.rsi_momentum_steps + 1, 3)
                title = (
                    f"RSI > {c.rsi_target_level:g}{cap} ({below}){rise}{fresh_s}{mom_s} "
                    f"| window={eff_lb} closes{ha_note}{rsi_bar_note}"
                )
            print(f"\n=== {title} ===")
            print(f"Scan Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
            print(
                f"Data: Binance spot klines | interval={c.interval} | RSI({c.rsi_period}) on regular **close** | "
                f"DROP_OPEN_CANDLE={c.drop_open_candle} | HA_PIPELINE={c.ha_pipeline}"
            )
            if c.require_smoothed_ha_green and c.ha_pipeline == "legacy":
                ha_expl = "HA pass = EMA(ha_close) > EMA(ha_open) (legacy)."
            elif c.require_smoothed_ha_green:
                ha_expl = (
                    "HA pass = TV double-smoothed Heikin-Ashi: **lime** when c2>=o2 (same as Pine o2>c2 ? red : lime)."
                )
            else:
                ha_expl = "HA filter off."
            print(
                f"RSI Chart Bars (Glaz-style zones on each row): **green** RSI>{c.rsi_bar_up_level:g}, "
                f"**red** RSI<{c.rsi_bar_down_level:g}, else neutral. {ha_expl}"
            )
            print(
                "RSI matches TradingView only with the **same interval**, regular candles, "
                "and the **same last closed bar**."
            )
            if c.rsi_strategy_mode == "cross_above" and c.rsi_momentum_steps > 0:
                print(
                    f"RSI window length = max(lookback={c.rsi_lookback_bars}, momentum_steps+1) "
                    f"for momentum + below-target checks."
                )
            print(
                f"(Illustrative) buy @ last close → sell +{c.take_profit_pct:g}% (fees/spread not included)"
            )
            print("-" * 45)
            for m in matches:
                hsc = m.get("ha_smooth_close")
                hso = m.get("ha_smooth_open")
                ha_bit = ""
                if hsc is not None and hso is not None and hsc == hsc and hso == hso:
                    pfx = "TV " if m.get("ha_pipeline") == "tv_smoothed" else ""
                    ha_bit = f" | {pfx}{self._fmt_ha_sigma(float(hsc), float(hso))}"
                tail = m.get("rsi_tail")
                tail_s = f" | tail={tail}" if tail else ""
                br = m.get("buy_reference")
                st = m.get("sell_target")
                px = ""
                if br is not None and st is not None:
                    px = (
                        f" | buy≈{self._fmt_price(float(br))} "
                        f"sell≈{self._fmt_price(float(st))} (+{c.take_profit_pct:g}%)"
                    )
                bars = m.get("rsi_chart_bar", "neutral")
                xfc = m.get("rsi_fresh_cross")
                xfc_bit = ""
                if xfc is not None:
                    xfc_bit = " | 55fresh" if xfc else " | 55noFresh"
                mv = m.get("rsi_momentum_steps")
                mom_bit = f" | RSImom:{mv}↑" if mv is not None and mv > 0 else ""
                print(
                    f"{m['symbol']:<10} | RSI: {m['rsi']} | RSIbars:{bars}{xfc_bit}{mom_bit}{px}{ha_bit}{tail_s}"
                )
            print("-" * 45)
            print(f"Total Matches: {len(matches)}\n")
        else:
            print("No matching tokens found.\n")


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    _setup_logging()
    run_once = os.environ.get("RUN_ONCE", "0").strip().lower() in ("1", "true", "yes", "on")
    scanner = RSIScannerV3()
    sleep_s = scanner.cfg.scan_sleep_s
    logging.getLogger(__name__).info(
        "RSIScannerV3 started (RUN_ONCE=%s, SCAN_SLEEP_S=%s)", run_once, sleep_s
    )
    while True:
        logging.getLogger(__name__).info("Running RSI scanner (v3)")
        scanner.print_matches(scanner.scan())
        if run_once:
            break
        logging.getLogger(__name__).info("Sleeping %.0fs\n", sleep_s)
        time.sleep(sleep_s)
