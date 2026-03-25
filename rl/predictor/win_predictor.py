"""
win_predictor.py
────────────────
Supervised win-probability predictor for trade entries.

For each candle where the 4h trend gate allows a trade, simulates a forward
TP/SL outcome and labels it as win (1) or loss (0). Trains a LightGBM
classifier to predict P(win | features) from the 46 market features.

This predictor sits UNDER the RL agent — its output (p_win) becomes
feature 47 in the observation, giving the agent a pre-computed edge signal.

No lookahead: predictor is trained on training data only, then applied to
both train and test data within each walk-forward window.

TP/SL logic:
  - For long entries (4h bullish): TP = +4%, SL = -2%
  - For short entries (4h bearish): TP = -4% from entry, SL = +2% from entry
  - Uses future candle high/low to check whether TP or SL was touched first
  - Timeout after max_bars candles → label = 0 (no clean winner)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Optional

# The 46 base features (no p_win — can't use itself as input)
_BASE_FEATURE_COLS = [
    "1h_open_z", "1h_high_z", "1h_low_z", "1h_close_z", "1h_vol_z",
    "4h_open_z", "4h_high_z", "4h_low_z", "4h_close_z", "4h_vol_z",
    "1d_open_z", "1d_high_z", "1d_low_z", "1d_close_z", "1d_vol_z",
    "1h_rsi", "1h_macd_hist", "1h_ema_spread", "1h_bb_pct",
    "4h_rsi", "4h_macd_hist", "4h_ema_spread", "4h_bb_pct",
    "1h_body_ratio", "1h_upper_wick", "1h_lower_wick",
    "4h_body_ratio", "4h_upper_wick", "4h_lower_wick",
    "1d_body_ratio", "1d_upper_wick", "1d_lower_wick",
    "vol_regime", "rel_volume", "funding_rate", "cross_pair_mom",
    "tod_sin", "tod_cos", "dow_sin", "dow_cos",
    "vol_delta", "trade_intensity",
    "trend_align_1h_4h", "trend_align_4h_1d", "momentum_align", "htf_bias",
]


def label_trades(
    df: pd.DataFrame,
    sl_pct: float = 0.02,
    tp_pct: float = 0.04,
    max_bars: int = 48,
) -> np.ndarray:
    """
    Label each candle with the outcome of a hypothetical trend-following trade.

    Entry: at the close of the current candle (if 4h trend gate allows).
    Exit:  first candle where high/low touches TP or SL.

    Returns float32 array of shape (len(df),):
      1.0  = TP hit first (winner)
      0.0  = SL hit first or timeout (loser)
      0.5  = no trade allowed at this candle (4h flat), treated as neutral
    """
    labels  = np.full(len(df), 0.5, dtype=np.float32)
    closes  = df["close"].values
    highs   = df["high"].values
    lows    = df["low"].values

    if "_4h_trend_up" in df.columns:
        trends = df["_4h_trend_up"].values
    else:
        trends = np.ones(len(df), dtype=np.float32)

    n = len(df)
    for i in range(n - 1):
        direction = float(trends[i])
        if direction == 0.0:
            labels[i] = 0.5   # no trade
            continue

        entry    = closes[i]
        tp_price = entry * (1.0 + direction * tp_pct)
        sl_price = entry * (1.0 - direction * sl_pct)

        outcome = 0.0   # default: timeout = loss
        for j in range(i + 1, min(i + max_bars + 1, n)):
            h = highs[j]
            l = lows[j]
            if direction > 0:   # long
                if h >= tp_price:
                    outcome = 1.0
                    break
                if l <= sl_price:
                    outcome = 0.0
                    break
            else:               # short
                if l <= tp_price:
                    outcome = 1.0
                    break
                if h >= sl_price:
                    outcome = 0.0
                    break

        labels[i] = outcome

    return labels


class WinPredictor:
    """
    LightGBM classifier that predicts P(trade wins) given current features.

    Usage in walk-forward:
        predictor = WinPredictor()
        predictor.fit(train_df)                       # train on training data only
        train_df["p_win"] = predictor.predict(train_df)
        test_df["p_win"]  = predictor.predict(test_df)
    """

    def __init__(
        self,
        sl_pct:   float = 0.02,
        tp_pct:   float = 0.04,
        max_bars: int   = 48,
    ):
        self.sl_pct   = sl_pct
        self.tp_pct   = tp_pct
        self.max_bars = max_bars
        self._model: Optional[lgb.LGBMClassifier] = None
        self._fitted  = False

    def fit(self, train_df: pd.DataFrame) -> "WinPredictor":
        """Train on training data. Only uses 46 base features — no lookahead."""
        labels = label_trades(train_df, self.sl_pct, self.tp_pct, self.max_bars)

        # Only train on candles where a trade is allowed (trend != 0)
        if "_4h_trend_up" in train_df.columns:
            tradeable = train_df["_4h_trend_up"].values != 0
        else:
            tradeable = np.ones(len(train_df), dtype=bool)

        # Exclude last max_bars rows (no complete forward window for labeling)
        tradeable[-self.max_bars:] = False

        X = train_df[_BASE_FEATURE_COLS].values[tradeable].astype(np.float32)
        y = labels[tradeable]

        if len(X) < 50:
            # Not enough data — model stays unfitted, predict_proba returns 0.5
            return self

        pos_count = int(y.sum())
        neg_count = len(y) - pos_count
        scale_pos = neg_count / max(pos_count, 1)

        self._model = lgb.LGBMClassifier(
            n_estimators      = 200,
            max_depth         = 4,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            scale_pos_weight  = scale_pos,
            random_state      = 42,
            n_jobs            = -1,
            verbose           = -1,
        )
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return P(win) for every row in df as float32 array.
        Returns 0.5 for all rows if model is not fitted.
        """
        if not self._fitted or self._model is None:
            return np.full(len(df), 0.5, dtype=np.float32)

        X = df[_BASE_FEATURE_COLS].values.astype(np.float32)
        proba = self._model.predict_proba(X)[:, 1]
        return proba.astype(np.float32)
