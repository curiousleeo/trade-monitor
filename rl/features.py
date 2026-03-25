"""
features.py
───────────
Builds a 44-feature market observation matrix from 15m OHLCV data.
Position-state features (direction, entry_distance) are added by the environment,
giving a total observation dimension of 46.

Feature vector layout (44 values):
  ── Block A: Z-scored OHLCV (15 features) ──────────────────────────────────
   0  15m_open_z       rolling z-score, 100-period lookback
   1  15m_high_z
   2  15m_low_z
   3  15m_close_z
   4  15m_vol_z
   5  1h_open_z        computed on native 1h TF, forward-filled (no lookahead)
   6  1h_high_z
   7  1h_low_z
   8  1h_close_z
   9  1h_vol_z
  10  4h_open_z        same, on native 4h TF
  11  4h_high_z
  12  4h_low_z
  13  4h_close_z
  14  4h_vol_z

  ── Block B: Technical indicators (8 features) ─────────────────────────────
  15  15m_rsi          (rsi - 50) / 50 → [-1, 1]
  16  15m_macd_hist    z-scored over 100 periods
  17  15m_ema_spread   EMA(9) - EMA(21) / close, z-scored
  18  15m_bb_pct       Bollinger %B [0,1] — mean reversion on 15m
  19  1h_rsi
  20  1h_macd_hist
  21  1h_ema_spread
  22  1h_bb_pct        Bollinger %B [0,1] — diverges from z-score in vol expansions

  ── Block C: Candle shape (9 features) ──────────────────────────────────────
  23  15m_body_ratio   abs(close-open) / (high-low+ε)   [0,1]
  24  15m_upper_wick   (high - max(o,c)) / (high-low+ε) [0,1]
  25  15m_lower_wick   (min(o,c) - low) / (high-low+ε)  [0,1]
  26  1h_body_ratio
  27  1h_upper_wick
  28  1h_lower_wick
  29  4h_body_ratio
  30  4h_upper_wick
  31  4h_lower_wick

  ── Block D: Regime & context (4 features) ──────────────────────────────────
  32  vol_regime       ATR(14)/ATR(50) on 1h — >1.5 expansion, <0.7 chop
  33  rel_volume       volume/SMA(vol,20) on 15m, clipped [0,3], /3 → [0,1]
  34  funding_rate     raw rate clipped [-0.001,0.001], /0.001 → [-1,1]
                       (0.0 if no funding data provided)
  35  cross_pair_mom   other pair's 15m log-return, z-scored over 100 periods
                       (0.0 if no cross-pair data provided)

  ── Block E: Time encoding (2 features) ─────────────────────────────────────
  36  tod_sin          sin(2π * hour_of_day / 24)
  37  tod_cos          cos(2π * hour_of_day / 24)

  ── Block F: Order flow & timeframe confluence (6 features) ─────────────────
  38  vol_delta        (taker_buy_vol - taker_sell_vol) / total_vol → [-1, 1]
                       buy pressure vs sell pressure within the candle
                       (0.0 if taker_buy_base not in df)
  39  trade_intensity  trades/SMA(trades,20), clipped [0,3], /3 → [0,1]
                       proxy for order flow activity / liquidity
                       (0.0 if trades not in df)
  40  trend_align_15_1h  sign(15m_ema_spread_raw) * sign(1h_ema_spread_raw) → {-1,0,+1}
                         +1 = 15m and 1h trend agree, -1 = counter-trend (dead cat bounce signal)
  41  trend_align_1h_4h  sign(1h_ema_spread_raw) * sign(4h_ema_spread_raw) → {-1,0,+1}
                         +1 = 1h and 4h trend agree, -1 = 1h is fighting 4h structure
  42  momentum_align   mean of sign(15m_macd), sign(1h_macd), sign(4h_macd) → [-1,1]
                       +1 = all three TFs agree bullish, -1 = all three agree bearish
                       values in between = mixed / uncertain
  43  htf_bias         (1h_close_z + 4h_close_z) / 2, clipped [-5,5]
                       combined higher-TF positioning — are we high or low in the
                       higher-TF range? Used to filter counter-trend 15m entries.

Lookahead prevention:
  Higher-timeframe (1h, 4h) features are computed from the COMPLETED candle
  that closed BEFORE the current 15m step. Partial/forming candles are never
  used. See _resample_no_lookahead() for implementation.

NaN / Inf policy:
  Final step clips all values to [-5, 5] and fills any residual NaN with 0.
  This matches the env's obs clipping.
"""

import numpy as np
import pandas as pd
import ta


N_FEATURES = 44   # market features only; env adds 2 position-state features = 46 obs total

FEATURE_COLS = [
    # Block A: Z-scored OHLCV (15 features)
    "15m_open_z", "15m_high_z", "15m_low_z", "15m_close_z", "15m_vol_z",
    "1h_open_z",  "1h_high_z",  "1h_low_z",  "1h_close_z",  "1h_vol_z",
    "4h_open_z",  "4h_high_z",  "4h_low_z",  "4h_close_z",  "4h_vol_z",
    # Block B: Technical indicators (8 features — RSI, MACD, EMA spread, BB%B × 2 TFs)
    "15m_rsi", "15m_macd_hist", "15m_ema_spread", "15m_bb_pct",
    "1h_rsi",  "1h_macd_hist",  "1h_ema_spread",  "1h_bb_pct",
    # Block C: Candle shape (9 features)
    "15m_body_ratio", "15m_upper_wick", "15m_lower_wick",
    "1h_body_ratio",  "1h_upper_wick",  "1h_lower_wick",
    "4h_body_ratio",  "4h_upper_wick",  "4h_lower_wick",
    # Block D: Regime & context (4 features)
    "vol_regime", "rel_volume", "funding_rate", "cross_pair_mom",
    # Block E: Time encoding (2 features)
    "tod_sin", "tod_cos",
    # Block F: Order flow & timeframe confluence (6 features)
    "vol_delta", "trade_intensity",
    "trend_align_15_1h", "trend_align_1h_4h", "momentum_align", "htf_bias",
]

assert len(FEATURE_COLS) == N_FEATURES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _zscore(series: pd.Series, lookback: int = 100) -> pd.Series:
    """Rolling z-score. Requires at least lookback//2 values to avoid long NaN run."""
    mu    = series.rolling(lookback, min_periods=lookback // 2).mean()
    sigma = series.rolling(lookback, min_periods=lookback // 2).std()
    return ((series - mu) / (sigma + 1e-8)).clip(-5, 5)


def _candle_shape(o: pd.Series, h: pd.Series,
                  l: pd.Series, c: pd.Series) -> tuple:
    """
    Candle anatomy ratios, naturally bounded [0, 1].
    Returns (body_ratio, upper_wick, lower_wick).
    """
    rng         = (h - l).clip(lower=1e-8)
    body_ratio  = (c - o).abs() / rng
    upper_wick  = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng
    lower_wick  = (pd.concat([o, c], axis=1).min(axis=1) - l) / rng
    return body_ratio.clip(0, 1), upper_wick.clip(0, 1), lower_wick.clip(0, 1)


def _resample_no_lookahead(df_15m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Resample 15m OHLCV to a higher timeframe WITHOUT lookahead.

    At each 15m step t we see the most recently COMPLETED higher-TF candle.
    Z-scores are computed on the native TF cadence (100 TF-periods) BEFORE
    aligning to 15m — this ensures the z-score is truly constant within each
    TF period and only updates when a new candle closes.

    Steps:
      1. Resample (closed='left', label='left' — Binance open-time convention)
      2. Compute z-scores on native TF (100-period rolling)
      3. shift(1) → only show the COMPLETED candle, not the forming one
      4. reindex to 15m timestamps + forward-fill
    """
    _alias = {"open": "open", "high": "high", "low": "low",
              "close": "close", "volume": "vol"}

    df = df_15m.copy().set_index("time")
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    df_tf = df.resample(tf, closed="left", label="left").agg(agg).dropna()

    # Z-scores on native TF — only updates when a candle closes
    for col in ["open", "high", "low", "close", "volume"]:
        alias = _alias[col]
        df_tf[f"{alias}_z"] = _zscore(df_tf[col])

    df_tf = df_tf.shift(1)                               # completed candle only
    aligned = df_tf.reindex(df.index, method="ffill")    # forward-fill to 15m
    return aligned.reset_index()


def _indicators_on_tf(df_tf: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Compute Block-B and Block-C features for a given timeframe DataFrame.
    df_tf must have columns: time, open, high, low, close, volume (aligned to 15m index).
    Returns a DataFrame of computed columns, indexed same as df_tf.
    """
    c, o, h, l, v = (df_tf["close"], df_tf["open"],
                     df_tf["high"], df_tf["low"], df_tf["volume"])

    out = pd.DataFrame(index=df_tf.index)

    # RSI → [-1, 1]
    rsi = ta.momentum.RSIIndicator(c, window=14).rsi()
    out[f"{prefix}_rsi"] = ((rsi - 50) / 50).clip(-1, 1)

    # MACD histogram → z-scored
    macd_hist = ta.trend.MACD(c).macd_diff()
    out[f"{prefix}_macd_hist"] = _zscore(macd_hist)

    # EMA spread → z-scored
    ema9  = ta.trend.EMAIndicator(c, window=9).ema_indicator()
    ema21 = ta.trend.EMAIndicator(c, window=21).ema_indicator()
    spread = (ema9 - ema21) / (c + 1e-8)
    out[f"{prefix}_ema_spread"] = _zscore(spread)

    # Candle shape [0, 1]
    body, uwk, lwk = _candle_shape(o, h, l, c)
    out[f"{prefix}_body_ratio"] = body
    out[f"{prefix}_upper_wick"] = uwk
    out[f"{prefix}_lower_wick"] = lwk

    return out


# ── Main feature builder ──────────────────────────────────────────────────────

def add_features(
    df: pd.DataFrame,
    funding_series: pd.Series = None,
    other_close_series: pd.Series = None,
) -> pd.DataFrame:
    """
    Build the full 37-feature observation matrix from 15m OHLCV.

    Args:
        df:                  15m OHLCV DataFrame [time, open, high, low, close, volume]
        funding_series:      Optional. Funding rates indexed by UTC timestamp.
                             Use fetch_data.align_funding_to_15m() to prepare this.
        other_close_series:  Optional. Other pair's close prices indexed by UTC timestamp
                             (BTC agent gets ETH close, ETH agent gets BTC close).
                             Must cover the same time range as df.

    Returns:
        DataFrame with FEATURE_COLS added, NaN warmup rows dropped.
    """
    df = df.copy().reset_index(drop=True)

    # ── Resample to higher timeframes (no lookahead) ──────────────────────────
    df_1h = _resample_no_lookahead(df, "1h")
    df_4h = _resample_no_lookahead(df, "4h")

    # ── Block A: Z-scored OHLCV ───────────────────────────────────────────────
    # 15m: compute z-score directly on 15m series (updates every candle)
    for col, alias in [("open","open"),("high","high"),("low","low"),
                       ("close","close"),("volume","vol")]:
        df[f"15m_{alias}_z"] = _zscore(df[col]).values

    # 1h / 4h: z-scores pre-computed on native TF in _resample_no_lookahead,
    # then forward-filled — truly constant within each TF period, no lookahead
    for tf_label, df_tf in [("1h", df_1h), ("4h", df_4h)]:
        for alias in ["open", "high", "low", "close", "vol"]:
            col = f"{alias}_z"
            df[f"{tf_label}_{alias}_z"] = df_tf[col].reset_index(drop=True).values

    # ── Block B + C: Indicators & candle shape ────────────────────────────────
    ind_15m = _indicators_on_tf(df, "15m")
    ind_1h  = _indicators_on_tf(df_1h, "1h")
    ind_4h  = _indicators_on_tf(df_4h, "4h")

    for col in ind_15m.columns:
        df[col] = ind_15m[col].values
    for col in ind_1h.columns:
        df[col] = ind_1h[col].values
    # 4h: only shape features (indicators on 4h are too slow-moving to be useful)
    for col in ["4h_body_ratio", "4h_upper_wick", "4h_lower_wick"]:
        df[col] = ind_4h[col].values

    # ── Block D1: Volatility regime (ATR14/ATR50 on 1h-aligned data) ──────────
    atr14 = ta.volatility.AverageTrueRange(
        df_1h["high"], df_1h["low"], df_1h["close"], window=14
    ).average_true_range()
    atr50 = ta.volatility.AverageTrueRange(
        df_1h["high"], df_1h["low"], df_1h["close"], window=50
    ).average_true_range()
    vol_regime = (atr14 / (atr50 + 1e-8)).clip(0, 4) / 4   # normalise to [0,1]
    df["vol_regime"] = vol_regime.values

    # also keep raw ATR for the environment's emergency stop calculation
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    # ── Block D2: Relative volume ─────────────────────────────────────────────
    vol_sma = df["volume"].rolling(20, min_periods=5).mean()
    df["rel_volume"] = (df["volume"] / (vol_sma + 1e-8)).clip(0, 3) / 3

    # ── Block D3: Funding rate ────────────────────────────────────────────────
    if funding_series is not None:
        aligned = funding_series.reindex(df["time"]).ffill().bfill().fillna(0.0)
        df["funding_rate"] = (aligned.values / 0.001).clip(-1, 1)
    else:
        df["funding_rate"] = 0.0

    # ── Block D4: Cross-pair momentum ─────────────────────────────────────────
    if other_close_series is not None:
        # Align to our 15m index
        other = other_close_series.reindex(df["time"]).ffill().bfill()
        log_ret = np.log(other / other.shift(1).replace(0, np.nan))
        df["cross_pair_mom"] = _zscore(log_ret).values
    else:
        df["cross_pair_mom"] = 0.0

    # ── Block E: Time encoding ────────────────────────────────────────────────
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        t = pd.to_datetime(df["time"], utc=True)
    else:
        t = df["time"]

    frac_of_day = (t.dt.hour + t.dt.minute / 60.0) / 24.0
    df["tod_sin"] = np.sin(2 * np.pi * frac_of_day)
    df["tod_cos"] = np.cos(2 * np.pi * frac_of_day)

    # ── Block B addition: Bollinger %B on 15m and 1h ─────────────────────────
    bb_15m = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["15m_bb_pct"] = bb_15m.bollinger_pband().clip(0, 1)

    bb_1h = ta.volatility.BollingerBands(df_1h["close"].reset_index(drop=True),
                                          window=20, window_dev=2)
    df["1h_bb_pct"] = bb_1h.bollinger_pband().clip(0, 1).values

    # ── Block F: Order flow & timeframe confluence ────────────────────────────

    # F1: Volume delta — buy pressure vs sell pressure
    # taker_buy_base is included in klines data; if missing (old parquet), default 0
    if "taker_buy_base" in df.columns:
        taker_buy  = df["taker_buy_base"].clip(lower=0)
        taker_sell = (df["volume"] - taker_buy).clip(lower=0)
        total      = taker_buy + taker_sell + 1e-8
        df["vol_delta"] = ((taker_buy - taker_sell) / total).clip(-1, 1)
    else:
        df["vol_delta"] = 0.0

    # F2: Trade intensity — order flow activity proxy
    if "trades" in df.columns:
        trade_sma = df["trades"].rolling(20, min_periods=5).mean()
        df["trade_intensity"] = (df["trades"] / (trade_sma + 1e-8)).clip(0, 3) / 3
    else:
        df["trade_intensity"] = 0.0

    # F3 & F4: Trend alignment — raw (unscaled) EMA spreads computed on each TF
    # We need the raw EMA spread sign BEFORE z-scoring.
    # 15m raw spread already available from ind_15m computation path — recompute cheaply
    ema9_15m  = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    ema21_15m = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    raw_spread_15m = (ema9_15m - ema21_15m)

    ema9_1h  = ta.trend.EMAIndicator(df_1h["close"], window=9).ema_indicator()
    ema21_1h = ta.trend.EMAIndicator(df_1h["close"], window=21).ema_indicator()
    raw_spread_1h = (ema9_1h - ema21_1h).values  # aligned to 15m via df_1h

    ema9_4h  = ta.trend.EMAIndicator(df_4h["close"], window=9).ema_indicator()
    ema21_4h = ta.trend.EMAIndicator(df_4h["close"], window=21).ema_indicator()
    raw_spread_4h = (ema9_4h - ema21_4h).values  # aligned to 15m via df_4h

    sign_15m = np.sign(raw_spread_15m.values)
    sign_1h  = np.sign(raw_spread_1h)
    sign_4h  = np.sign(raw_spread_4h)

    # +1 = both TFs agree direction, -1 = counter-trend (e.g. dead cat bounce)
    df["trend_align_15_1h"] = (sign_15m * sign_1h).clip(-1, 1)
    df["trend_align_1h_4h"] = (sign_1h  * sign_4h).clip(-1, 1)

    # F5: Momentum alignment across all 3 TFs
    macd_15m = ta.trend.MACD(df["close"]).macd_diff()
    macd_1h  = ta.trend.MACD(df_1h["close"]).macd_diff()
    macd_4h  = ta.trend.MACD(df_4h["close"]).macd_diff()

    sign_macd_15m = np.sign(macd_15m.values)
    sign_macd_1h  = np.sign(macd_1h.values)
    sign_macd_4h  = np.sign(macd_4h.values)

    # Average of three signs: +1 = all bullish, -1 = all bearish, 0 = mixed
    df["momentum_align"] = ((sign_macd_15m + sign_macd_1h + sign_macd_4h) / 3).clip(-1, 1)

    # F6: HTF bias — combined higher-TF positioning
    # Are we high or low in the higher-TF price range?
    df["htf_bias"] = ((df["1h_close_z"] + df["4h_close_z"]) / 2).clip(-5, 5)

    # ── Final cleanup ─────────────────────────────────────────────────────────
    # Drop warmup rows where rolling indicators haven't filled yet
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # Clip and zero-fill any residual edge cases
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(-5, 5)

    return df


def get_obs(df: pd.DataFrame, idx: int) -> np.ndarray:
    """
    Return the 37-feature observation at row `idx` as a float32 array.
    The environment appends position-state features on top of this.
    """
    return df.iloc[idx][FEATURE_COLS].to_numpy(dtype=np.float32)
