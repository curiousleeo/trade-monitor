"""
environment.py
──────────────
OpenAI Gymnasium trading environment.

Actions (Discrete 4):
    0 = HOLD
    1 = LONG  (open long if flat, ignore if already long)
    2 = SHORT (open short if flat, ignore if already short)
    3 = CLOSE (close any open position)

State:
    Flat 15-feature vector from features.py (one candle)

Reward:
    Realised log-return when a trade closes, else 0.
    Penalty for holding too long without acting.
    Small transaction cost on every entry/exit.

Episode:
    Starts at a random candle, runs for `episode_len` steps.
    Terminates early if balance drops below 20% of start.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from features import FEATURE_COLS, N_FEATURES, get_obs


# ── Config ────────────────────────────────────────────────────────────────────

INITIAL_BALANCE  = 1_000.0
RISK_PER_TRADE   = 0.02          # 2% of balance risked per trade
SL_ATR_MULT      = 1.5           # stop loss = entry ± ATR × 1.5
TP_ATR_MULT      = 3.0           # take profit = entry ± ATR × 3.0
TAKER_FEE        = 0.001         # 0.1% per side (Binance taker)
HOLD_PENALTY     = -0.0001       # tiny per-step penalty for doing nothing
MIN_BALANCE_FRAC = 0.20          # terminate if balance < 20% of start


class TradingEnv(gym.Env):
    """Single-asset, single-position trading environment."""

    metadata = {"render_modes": []}

    def __init__(self, df, episode_len: int = 2_000, lookback: int = 1):
        super().__init__()
        self.df          = df.reset_index(drop=True)
        self.episode_len = episode_len
        self.lookback    = lookback

        obs_size = N_FEATURES * lookback + 3   # features + [position, unrealised_pnl, steps_in_trade]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)   # HOLD, LONG, SHORT, CLOSE

        self._reset_state()

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Start at a random point with enough room for a full episode
        max_start = len(self.df) - self.episode_len - 1
        self.cursor = self.np_random.integers(0, max(1, max_start))
        self._reset_state()
        obs  = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action)

        row         = self.df.iloc[self.cursor]
        price       = float(row["close"])
        atr         = float(row["atr"])
        reward      = 0.0
        terminated  = False

        # ── Check TP/SL on open position ─────────────────────────────────────
        if self.position != 0:
            self.steps_in_trade += 1
            high = float(self.df.iloc[self.cursor]["high"])
            low  = float(self.df.iloc[self.cursor]["low"])

            tp_hit = (self.position ==  1 and high >= self.tp_price) or \
                     (self.position == -1 and low  <= self.tp_price)
            sl_hit = (self.position ==  1 and low  <= self.sl_price) or \
                     (self.position == -1 and high >= self.sl_price)

            if tp_hit:
                reward     = self._close_trade(self.tp_price)
                action     = 3   # override to CLOSE
            elif sl_hit:
                reward     = self._close_trade(self.sl_price)
                action     = 3
            else:
                reward     = HOLD_PENALTY

        # ── Execute agent action ──────────────────────────────────────────────
        if action == 1 and self.position == 0:    # LONG
            self._open_trade(price, atr, direction=1)
        elif action == 2 and self.position == 0:  # SHORT
            self._open_trade(price, atr, direction=-1)
        elif action == 3 and self.position != 0:  # CLOSE (manual)
            reward += self._close_trade(price)

        # ── Advance ──────────────────────────────────────────────────────────
        self.cursor     += 1
        self.step_count += 1

        # ── Termination ──────────────────────────────────────────────────────
        truncated  = self.step_count >= self.episode_len
        terminated = (
            self.cursor >= len(self.df) - 1 or
            self.balance < INITIAL_BALANCE * MIN_BALANCE_FRAC
        )

        obs  = self._get_obs()
        info = {"balance": self.balance, "position": self.position}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass   # optionally implement later

    # ── Internals ─────────────────────────────────────────────────────────────

    def _reset_state(self):
        self.balance        = INITIAL_BALANCE
        self.position       = 0      # 0=flat, 1=long, -1=short
        self.entry_price    = 0.0
        self.entry_atr      = 0.0
        self.tp_price       = 0.0
        self.sl_price       = 0.0
        self.units          = 0.0
        self.step_count     = 0
        self.steps_in_trade = 0

    def _open_trade(self, price: float, atr: float, direction: int):
        sl_dist        = atr * SL_ATR_MULT
        risk_amount    = self.balance * RISK_PER_TRADE
        self.units     = risk_amount / sl_dist if sl_dist > 0 else 0
        self.units    *= (1 - TAKER_FEE)       # fee on entry

        self.position       = direction
        self.entry_price    = price
        self.entry_atr      = atr
        self.steps_in_trade = 0

        if direction == 1:   # LONG
            self.sl_price = price - sl_dist
            self.tp_price = price + atr * TP_ATR_MULT
        else:                # SHORT
            self.sl_price = price + sl_dist
            self.tp_price = price - atr * TP_ATR_MULT

    def _close_trade(self, exit_price: float) -> float:
        pnl             = self.position * (exit_price - self.entry_price) * self.units
        pnl            -= abs(pnl) * TAKER_FEE   # fee on exit
        self.balance   += pnl
        log_ret         = np.log(max(self.balance, 1e-6) / INITIAL_BALANCE)
        self.position   = 0
        self.units      = 0.0
        self.steps_in_trade = 0
        return float(log_ret)   # reward = cumulative log-return

    def _get_obs(self) -> np.ndarray:
        feat = get_obs(self.df, self.cursor, self.lookback)
        extra = np.array([
            float(self.position),
            self._unrealised_pnl_pct(),
            float(self.steps_in_trade) / 100.0,   # normalised
        ], dtype=np.float32)
        return np.concatenate([feat, extra])

    def _unrealised_pnl_pct(self) -> float:
        if self.position == 0:
            return 0.0
        price = float(self.df.iloc[self.cursor]["close"])
        raw   = self.position * (price - self.entry_price) * self.units
        return float(np.clip(raw / self.balance, -1, 1))
