"""
features.py
───────────
Turns raw OHLCV candles into a normalised feature matrix
that the RL agent observes at each step.

All features are bounded to [-1, 1] or [0, 1] so the network
doesn't have to deal with wildly different scales.

Feature vector (15 values):
  0  ema_20_vs_price   price position relative to EMA-20
  1  ema_50_vs_price   price position relative to EMA-50
  2  ema_200_vs_price  price position relative to EMA-200 (trend regime)
  3  ema_cross         EMA-20 vs EMA-50 (momentum)
  4  rsi_14            RSI normalised to [-1, 1]  (0.5 → 0, overbought → +1)
  5  macd_hist         MACD histogram, z-scored then clipped
  6  bb_pct            Bollinger %B: where price sits in the band [0, 1]
  7  atr_pct           ATR / price — volatility regime [0, 1]
  8  volume_z          volume z-score vs 20-period window, clipped [-3, 3] → /3
  9  candle_body       (close - open) / ATR — body size & direction
  10 upper_wick        upper wick / ATR
  11 lower_wick        lower wick / ATR
  12 tod_sin           time-of-day sine encoding (daily cycle)
  13 tod_cos           time-of-day cosine encoding
  14 dow               day-of-week normalised [0, 1]
"""

import numpy as np
import pandas as pd
import ta


N_FEATURES = 15


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all indicator columns to the dataframe.
    Returns a new dataframe with NaN rows dropped.
    Input must have columns: time, open, high, low, close, volume
    """
    df = df.copy()

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    open_ = df["open"]
    vol   = df["volume"]

    # ── EMAs ─────────────────────────────────────────────────────────────────
    ema20  = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    ema50  = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    ema200 = ta.trend.EMAIndicator(close, window=200).ema_indicator()

    df["ema20_vs_price"]  = (close - ema20)  / close
    df["ema50_vs_price"]  = (close - ema50)  / close
    df["ema200_vs_price"] = (close - ema200) / close
    df["ema_cross"]       = (ema20 - ema50)  / close

    # ── RSI → [-1, 1] ────────────────────────────────────────────────────────
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["rsi"] = (rsi - 50) / 50   # 0→-1, 50→0, 100→+1

    # ── MACD histogram → z-score clipped ─────────────────────────────────────
    macd_hist = ta.trend.MACD(close).macd_diff()
    macd_std  = macd_hist.rolling(100).std().replace(0, np.nan)
    df["macd_hist"] = (macd_hist / macd_std).clip(-3, 3) / 3

    # ── Bollinger %B ──────────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pct"] = bb.bollinger_pband().clip(0, 1)

    # ── ATR % ─────────────────────────────────────────────────────────────────
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr"] = atr   # keep raw for position sizing later
    df["atr_pct"] = (atr / close).clip(0, 0.1) / 0.1   # normalise to [0,1]

    # ── Volume z-score ────────────────────────────────────────────────────────
    vol_mean = vol.rolling(20).mean()
    vol_std  = vol.rolling(20).std().replace(0, np.nan)
    df["volume_z"] = ((vol - vol_mean) / vol_std).clip(-3, 3) / 3

    # ── Candle anatomy / ATR ──────────────────────────────────────────────────
    body        = close - open_
    upper_wick  = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_wick  = pd.concat([close, open_], axis=1).min(axis=1) - low
    atr_safe    = atr.replace(0, np.nan)

    df["candle_body"]  = (body       / atr_safe).clip(-3, 3) / 3
    df["upper_wick"]   = (upper_wick / atr_safe).clip(0, 3)  / 3
    df["lower_wick"]   = (lower_wick / atr_safe).clip(0, 3)  / 3

    # ── Time encoding ─────────────────────────────────────────────────────────
    if pd.api.types.is_datetime64_any_dtype(df["time"]):
        t = df["time"]
    else:
        t = pd.to_datetime(df["time"], utc=True)

    seconds_in_day = 24 * 3600
    tod = t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second
    df["tod_sin"] = np.sin(2 * np.pi * tod / seconds_in_day)
    df["tod_cos"] = np.cos(2 * np.pi * tod / seconds_in_day)
    df["dow"]     = t.dt.dayofweek / 6   # Mon=0 → 0.0, Sun=6 → 1.0

    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    "ema20_vs_price", "ema50_vs_price", "ema200_vs_price", "ema_cross",
    "rsi", "macd_hist", "bb_pct", "atr_pct", "volume_z",
    "candle_body", "upper_wick", "lower_wick",
    "tod_sin", "tod_cos", "dow",
]


def get_obs(df: pd.DataFrame, idx: int, lookback: int = 1) -> np.ndarray:
    """
    Return the observation at index `idx`.
    With lookback=1 (default) returns a flat 15-element array.
    With lookback>1 returns a (lookback, 15) matrix for CNN/LSTM agents.
    """
    start = max(0, idx - lookback + 1)
    rows  = df.iloc[start : idx + 1][FEATURE_COLS].values
    if lookback > 1 and len(rows) < lookback:
        # pad with zeros at the front if near start of data
        pad  = np.zeros((lookback - len(rows), N_FEATURES))
        rows = np.vstack([pad, rows])
    return rows.astype(np.float32).flatten() if lookback == 1 else rows.astype(np.float32)
