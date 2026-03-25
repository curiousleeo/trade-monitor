"""
features_1h.py
──────────────
Builds a 46-feature market observation matrix using 1h candles as the
primary execution timeframe, with 4h and 1d as higher-timeframe context.

Input:  15m OHLCV parquet (with taker_buy_base + trades columns)
Output: DataFrame at 1h cadence with 47 market features

Feature vector layout (46 values):
  ── Block A: Z-scored OHLCV (15 features) ──────────────────────────────────
   0  1h_open_z       rolling z-score, 100-period lookback on native 1h TF
   1  1h_high_z
   2  1h_low_z
   3  1h_close_z
   4  1h_vol_z
   5  4h_open_z       computed on native 4h TF, forward-filled to 1h (no lookahead)
   6  4h_high_z
   7  4h_low_z
   8  4h_close_z
   9  4h_vol_z
  10  1d_open_z       computed on native 1d TF, forward-filled to 1h
  11  1d_high_z
  12  1d_low_z
  13  1d_close_z
  14  1d_vol_z

  ── Block B: Technical indicators (8 features) ─────────────────────────────
  15  1h_rsi          (rsi14 - 50) / 50 → [-1, 1]
  16  1h_macd_hist    z-scored over 100 periods
  17  1h_ema_spread   EMA(9) - EMA(21) / close, z-scored
  18  1h_bb_pct       Bollinger %B [0,1]
  19  4h_rsi
  20  4h_macd_hist
  21  4h_ema_spread
  22  4h_bb_pct

  ── Block C: Candle shape (9 features) ──────────────────────────────────────
  23  1h_body_ratio   abs(close-open) / (high-low+ε)   [0,1]
  24  1h_upper_wick   (high - max(o,c)) / (high-low+ε) [0,1]
  25  1h_lower_wick   (min(o,c) - low) / (high-low+ε)  [0,1]
  26  4h_body_ratio
  27  4h_upper_wick
  28  4h_lower_wick
  29  1d_body_ratio
  30  1d_upper_wick
  31  1d_lower_wick

  ── Block D: Regime & context (4 features) ──────────────────────────────────
  32  vol_regime      ATR(14)/ATR(50) on 4h — expansion vs chop
  33  rel_volume      volume/SMA(vol,20) on 1h, clipped [0,3], /3 → [0,1]
  34  funding_rate    raw rate clipped [-0.001,0.001], /0.001 → [-1,1]
  35  cross_pair_mom  other pair's 1h log-return, z-scored

  ── Block E: Time encoding (4 features) ─────────────────────────────────────
  36  tod_sin         sin(2π * hour_of_day / 24)
  37  tod_cos         cos(2π * hour_of_day / 24)
  38  dow_sin         sin(2π * day_of_week / 7)   — Mon=0 … Sun=6
  39  dow_cos         cos(2π * day_of_week / 7)

  ── Block F: Order flow & timeframe confluence (6 features) ─────────────────
  40  vol_delta       (taker_buy - taker_sell) / total_vol → [-1, 1]
  41  trade_intensity trades/SMA(trades,20), clipped [0,3], /3 → [0,1]
  42  trend_align_1h_4h  sign(1h_ema_raw) * sign(4h_ema_raw) → {-1,0,+1}
  43  trend_align_4h_1d  sign(4h_ema_raw) * sign(1d_ema_raw) → {-1,0,+1}
  44  momentum_align  mean of sign(1h_macd, 4h_macd, 1d_macd) → [-1,1]
  45  htf_bias        (4h_close_z + 1d_close_z) / 2

Extra column (not a feature, used by env for 4h trend gate):
  _4h_trend_up      +1 if 4h EMA9 > EMA21, -1 if below, 0 if flat

Lookahead prevention:
  4h and 1d features use shift(1) on native TF before aligning to 1h,
  ensuring only COMPLETED candles are visible — never the forming one.
"""

import numpy as np
import pandas as pd
import ta

N_FEATURES = 47

FEATURE_COLS = [
    # Block A
    "1h_open_z", "1h_high_z", "1h_low_z", "1h_close_z", "1h_vol_z",
    "4h_open_z", "4h_high_z", "4h_low_z", "4h_close_z", "4h_vol_z",
    "1d_open_z", "1d_high_z", "1d_low_z", "1d_close_z", "1d_vol_z",
    # Block B
    "1h_rsi", "1h_macd_hist", "1h_ema_spread", "1h_bb_pct",
    "4h_rsi", "4h_macd_hist", "4h_ema_spread", "4h_bb_pct",
    # Block C
    "1h_body_ratio", "1h_upper_wick", "1h_lower_wick",
    "4h_body_ratio", "4h_upper_wick", "4h_lower_wick",
    "1d_body_ratio", "1d_upper_wick", "1d_lower_wick",
    # Block D
    "vol_regime", "rel_volume", "funding_rate", "cross_pair_mom",
    # Block E
    "tod_sin", "tod_cos", "dow_sin", "dow_cos",
    # Block F
    "vol_delta", "trade_intensity",
    "trend_align_1h_4h", "trend_align_4h_1d", "momentum_align", "htf_bias",
    # Block G: Win probability (filled by WinPredictor per walk-forward window)
    "p_win",
]

assert len(FEATURE_COLS) == N_FEATURES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _zscore(series: pd.Series, lookback: int = 100) -> pd.Series:
    mu    = series.rolling(lookback, min_periods=lookback // 2).mean()
    sigma = series.rolling(lookback, min_periods=lookback // 2).std()
    return ((series - mu) / (sigma + 1e-8)).clip(-5, 5)


def _candle_shape(o, h, l, c):
    rng        = (h - l).clip(lower=1e-8)
    body_ratio = (c - o).abs() / rng
    upper_wick = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng
    lower_wick = (pd.concat([o, c], axis=1).min(axis=1) - l) / rng
    return body_ratio.clip(0, 1), upper_wick.clip(0, 1), lower_wick.clip(0, 1)


def _resample_to_1h(df_15m: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 15m OHLCV (with optional taker_buy_base + trades) to 1h.
    Summing taker_buy_base/trades gives accurate hourly order flow.
    """
    df = df_15m.copy().set_index("time")
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    if "taker_buy_base" in df.columns:
        agg["taker_buy_base"] = "sum"
    if "trades" in df.columns:
        agg["trades"] = "sum"
    df_1h = df.resample("1h", closed="left", label="left").agg(agg).dropna()
    return df_1h.reset_index()


def _resample_no_lookahead(df_1h: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Resample 1h OHLCV to a higher timeframe (4h or 1d) WITHOUT lookahead.
    Z-scores computed on native TF, then shift(1) + forward-fill to 1h.
    """
    _alias = {"open": "open", "high": "high", "low": "low",
              "close": "close", "volume": "vol"}

    df = df_1h.copy().set_index("time")
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    df_tf = df.resample(tf, closed="left", label="left").agg(agg).dropna()

    for col in ["open", "high", "low", "close", "volume"]:
        alias = _alias[col]
        df_tf[f"{alias}_z"] = _zscore(df_tf[col])

    df_tf = df_tf.shift(1)
    aligned = df_tf.reindex(df.index, method="ffill")
    return aligned.reset_index()


def _indicators_on_tf(df_tf: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Block B + Block C features for a given TF DataFrame aligned to 1h."""
    c, o, h, l = df_tf["close"], df_tf["open"], df_tf["high"], df_tf["low"]
    out = pd.DataFrame(index=df_tf.index)

    # RSI
    rsi = ta.momentum.RSIIndicator(c, window=14).rsi()
    out[f"{prefix}_rsi"] = ((rsi - 50) / 50).clip(-1, 1)

    # MACD histogram
    out[f"{prefix}_macd_hist"] = _zscore(ta.trend.MACD(c).macd_diff())

    # EMA spread
    ema9  = ta.trend.EMAIndicator(c, window=9).ema_indicator()
    ema21 = ta.trend.EMAIndicator(c, window=21).ema_indicator()
    out[f"{prefix}_ema_spread"] = _zscore((ema9 - ema21) / (c + 1e-8))

    # Bollinger %B
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    out[f"{prefix}_bb_pct"] = bb.bollinger_pband().clip(0, 1)

    # Candle shape
    body, uwk, lwk = _candle_shape(o, h, l, c)
    out[f"{prefix}_body_ratio"] = body
    out[f"{prefix}_upper_wick"] = uwk
    out[f"{prefix}_lower_wick"] = lwk

    return out


# ── Main feature builder ──────────────────────────────────────────────────────

def add_features_1h(
    df_15m: pd.DataFrame,
    funding_series: pd.Series = None,
    other_close_series: pd.Series = None,
) -> pd.DataFrame:
    """
    Build 46-feature observation matrix at 1h cadence from 15m OHLCV.

    Args:
        df_15m:             15m OHLCV (with taker_buy_base + trades if available)
        funding_series:     Optional. Funding rates indexed by UTC timestamp.
        other_close_series: Optional. Other pair's 1h close prices.

    Returns:
        DataFrame at 1h cadence with FEATURE_COLS + _4h_trend_up column.
    """
    # ── Step 1: Resample 15m → 1h ─────────────────────────────────────────────
    df = _resample_to_1h(df_15m)

    # ── Step 2: Higher TF alignments (no lookahead) ───────────────────────────
    df_4h = _resample_no_lookahead(df, "4h")
    df_1d = _resample_no_lookahead(df, "1d")

    # ── Block A: Z-scored OHLCV ───────────────────────────────────────────────
    for col, alias in [("open","open"),("high","high"),("low","low"),
                       ("close","close"),("volume","vol")]:
        df[f"1h_{alias}_z"] = _zscore(df[col]).values

    for tf_label, df_tf in [("4h", df_4h), ("1d", df_1d)]:
        for alias in ["open", "high", "low", "close", "vol"]:
            df[f"{tf_label}_{alias}_z"] = df_tf[f"{alias}_z"].reset_index(drop=True).values

    # ── Block B + C: Indicators & candle shape ────────────────────────────────
    ind_1h = _indicators_on_tf(df, "1h")
    ind_4h = _indicators_on_tf(df_4h, "4h")
    ind_1d = _indicators_on_tf(df_1d, "1d")

    for col in ind_1h.columns:
        df[col] = ind_1h[col].values
    for col in ind_4h.columns:
        df[col] = ind_4h[col].values
    # 1d: shape only (daily indicators too slow-moving for entry decisions)
    for col in ["1d_body_ratio", "1d_upper_wick", "1d_lower_wick"]:
        df[col] = ind_1d[col].values

    # ── Block D: Regime & context ─────────────────────────────────────────────
    atr14 = ta.volatility.AverageTrueRange(
        df_4h["high"], df_4h["low"], df_4h["close"], window=14
    ).average_true_range()
    atr50 = ta.volatility.AverageTrueRange(
        df_4h["high"], df_4h["low"], df_4h["close"], window=50
    ).average_true_range()
    df["vol_regime"] = ((atr14 / (atr50 + 1e-8)).clip(0, 4) / 4).values

    vol_sma = df["volume"].rolling(20, min_periods=5).mean()
    df["rel_volume"] = (df["volume"] / (vol_sma + 1e-8)).clip(0, 3) / 3

    if funding_series is not None:
        aligned = funding_series.reindex(df["time"]).ffill().bfill().fillna(0.0)
        df["funding_rate"] = (aligned.values / 0.001).clip(-1, 1)
    else:
        df["funding_rate"] = 0.0

    if other_close_series is not None:
        other = other_close_series.reindex(df["time"]).ffill().bfill()
        log_ret = np.log(other / other.shift(1).replace(0, np.nan))
        df["cross_pair_mom"] = _zscore(log_ret).values
    else:
        df["cross_pair_mom"] = 0.0

    # also keep raw ATR on 1h for the env's emergency stop
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    # ── Block E: Time encoding ────────────────────────────────────────────────
    t = pd.to_datetime(df["time"], utc=True) if not pd.api.types.is_datetime64_any_dtype(df["time"]) else df["time"]

    frac_day = (t.dt.hour + t.dt.minute / 60.0) / 24.0
    df["tod_sin"] = np.sin(2 * np.pi * frac_day)
    df["tod_cos"] = np.cos(2 * np.pi * frac_day)

    frac_week = t.dt.dayofweek / 7.0   # Mon=0 … Sun=6
    df["dow_sin"] = np.sin(2 * np.pi * frac_week)
    df["dow_cos"] = np.cos(2 * np.pi * frac_week)

    # ── Block F: Order flow & confluence ─────────────────────────────────────
    # F1: Volume delta
    if "taker_buy_base" in df.columns:
        taker_buy  = df["taker_buy_base"].clip(lower=0)
        taker_sell = (df["volume"] - taker_buy).clip(lower=0)
        df["vol_delta"] = ((taker_buy - taker_sell) / (taker_buy + taker_sell + 1e-8)).clip(-1, 1)
    else:
        df["vol_delta"] = 0.0

    # F2: Trade intensity
    if "trades" in df.columns:
        trade_sma = df["trades"].rolling(20, min_periods=5).mean()
        df["trade_intensity"] = (df["trades"] / (trade_sma + 1e-8)).clip(0, 3) / 3
    else:
        df["trade_intensity"] = 0.0

    # F3-F4: Trend alignment (raw EMA spreads, pre-z-score)
    ema9_1h  = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    ema21_1h = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    raw_1h   = ema9_1h - ema21_1h

    ema9_4h  = ta.trend.EMAIndicator(df_4h["close"], window=9).ema_indicator()
    ema21_4h = ta.trend.EMAIndicator(df_4h["close"], window=21).ema_indicator()
    raw_4h   = (ema9_4h - ema21_4h).values

    ema9_1d  = ta.trend.EMAIndicator(df_1d["close"], window=9).ema_indicator()
    ema21_1d = ta.trend.EMAIndicator(df_1d["close"], window=21).ema_indicator()
    raw_1d   = (ema9_1d - ema21_1d).values

    df["trend_align_1h_4h"] = (np.sign(raw_1h.values) * np.sign(raw_4h)).clip(-1, 1)
    df["trend_align_4h_1d"] = (np.sign(raw_4h) * np.sign(raw_1d)).clip(-1, 1)

    # F5: Momentum alignment (1h, 4h, 1d MACD signs)
    macd_1h = ta.trend.MACD(df["close"]).macd_diff()
    macd_4h = ta.trend.MACD(df_4h["close"]).macd_diff()
    macd_1d = ta.trend.MACD(df_1d["close"]).macd_diff()
    df["momentum_align"] = (
        (np.sign(macd_1h.values) + np.sign(macd_4h.values) + np.sign(macd_1d.values)) / 3
    ).clip(-1, 1)

    # F6: HTF bias — combined 4h + 1d positioning
    df["htf_bias"] = ((df["4h_close_z"] + df["1d_close_z"]) / 2).clip(-5, 5)

    # ── 4h trend gate column (used by env, not a feature) ────────────────────
    # +1 = 4h bullish (EMA9 > EMA21), -1 = bearish, 0 = flat
    df["_4h_trend_up"] = np.sign(raw_4h).astype(np.float32)

    # ── Block G: Win probability (default 0.5 — overwritten per window in walk-forward) ──
    df["p_win"] = 0.5

    # ── Final cleanup ─────────────────────────────────────────────────────────
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(-5, 5)

    return df


def get_obs(df: pd.DataFrame, idx: int) -> np.ndarray:
    """Return the 46 market features at row idx as float32."""
    return df.iloc[idx][FEATURE_COLS].to_numpy(dtype=np.float32)
