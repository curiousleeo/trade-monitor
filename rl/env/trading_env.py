"""
trading_env.py
──────────────
Gymnasium trading environment for the RL agent.

Observation space (40 dims):
  [0:38]  Market features from features.py (38-dim)
  [38]    position_direction  — 0=flat, 1=long, -1=short  (normalised: /1)
  [39]    entry_distance      — (current_price - entry_price) / ATR at entry
                               0 when flat

Action space (Dict):
  direction: Discrete(4)   — 0=Hold, 1=Long, 2=Short, 3=Close
  sizing:    Box(0, 1)     — fraction of available equity to risk

Action semantics:
  Hold  — do nothing
  Long  — open long (or flip from short). Ignored if already long.
  Short — open short (or flip from long). Ignored if already short.
  Close — close current position. Ignored if flat.

  Sizing < 0.10 on Long/Short → overridden to Hold (agent not confident enough)

Position lifecycle:
  - Entry: charges taker fee + slippage immediately
  - Exit:  charges taker fee + slippage immediately
  - Flip (Long→Short):  one close + one open in the same step (double fees)
  - Max hold: 96 candles → force close at market
  - Emergency stop: if unrealised loss > min(3×ATR_entry, 5% equity) → force close

Reward:
  - On close: reward_on_close() from reward.py
  - On hold (flat): reward_on_hold() inactivity cost
  - On hold (in position): 0.0
  - Circuit breaker: episode terminates + CIRCUIT_PENALTY

NO unrealised PnL in reward — agent must learn to time exits, not just ride.
NO take-profit or stop-loss orders — agent manages exits via Close action.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.reward import (
    reward_on_close, reward_on_hold, check_circuit_breaker,
    TradeResult,
)
from env.fee_model import calc_trade_cost
try:
    from features_1h import get_obs, N_FEATURES
except ImportError:
    from features import get_obs, N_FEATURES


# ── Constants ──────────────────────────────────────────────────────────────────

INITIAL_EQUITY    = 10_000.0   # starting account size ($)
MAX_HOLD_CANDLES  = 500        # force-close after 500 candles (~3 weeks at 1h)
EMERGENCY_ATR_MULT = 3.0       # emergency stop: 3× ATR at entry
EMERGENCY_MAX_PCT  = 0.05      # emergency stop: 5% of equity at entry
MIN_SIZING         = 0.10      # sizing below this → Hold override
SL_PCT             = 0.02      # hard stop-loss: -2% from entry (risk management)
TP_PCT             = None      # no hard TP — agent decides when to exit (discretionary)


# ── Direction constants ────────────────────────────────────────────────────────

HOLD  = 0
LONG  = 1
SHORT = 2
CLOSE = 3


@dataclass
class PositionState:
    """Tracks the currently open position."""
    direction:       int    = 0      # 1=long, -1=short, 0=flat
    entry_price:     float = 0.0
    entry_idx:       int   = 0       # candle index at entry
    size:            float = 0.0     # position size in base units (e.g. BTC)
    size_pct:        float = 0.0     # agent's raw sizing output at entry
    atr_at_entry:    float = 0.0     # ATR(14) at entry candle
    equity_at_entry: float = 0.0     # account equity at entry (for emergency stop)
    mae:             float = 0.0     # max adverse excursion (dollars, always ≥ 0)
    # Market context at entry (for fee calculation)
    vol_regime_entry:  float = 0.0
    rel_volume_entry:  float = 0.0


class TradingEnv(gym.Env):
    """
    Single-asset RL trading environment.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched 15m DataFrame from features.add_features().
        Must have columns: FEATURE_COLS + ['open','high','low','close','atr']
    initial_equity : float
        Starting account balance in USD.
    stress_fees : bool
        If True, double slippage for stress-testing (validation check #8).
    episode_length : int or None
        Number of steps per episode. None = full dataset.
    start_idx : int
        Starting candle index (allows windowed training).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df,
        initial_equity: float = INITIAL_EQUITY,
        stress_fees:    bool  = False,
        episode_length: Optional[int] = None,
        start_idx:      int   = 0,
    ):
        super().__init__()

        self.df             = df.reset_index(drop=True)
        self.initial_equity = initial_equity
        self.stress_fees    = stress_fees
        self.episode_length = episode_length
        self.start_idx      = start_idx

        obs_dim = N_FEATURES + 2   # 38 market + direction + entry_distance

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Dict({
            "direction": spaces.Discrete(4),             # 0=Hold,1=Long,2=Short,3=Close
            "sizing":    spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

        # Episode state (initialised in reset)
        self._idx:          int           = 0
        self._end_idx:      int           = 0
        self._equity:       float         = initial_equity
        self._starting_eq:  float         = initial_equity
        self._pos:          PositionState = PositionState()
        self._candles_flat: int           = 0

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _price(self, idx: int) -> float:
        """Mid-price at candle `idx` (use close as execution price)."""
        return float(self.df.at[idx, "close"])

    def _atr(self, idx: int) -> float:
        return max(float(self.df.at[idx, "atr"]), 1e-8)

    def _vol_regime(self, idx: int) -> float:
        return float(self.df.at[idx, "vol_regime"])

    def _rel_volume(self, idx: int) -> float:
        return float(self.df.at[idx, "rel_volume"])

    def _obs(self) -> np.ndarray:
        market = get_obs(self.df, self._idx)   # (38,) float32

        if self._pos.direction != 0:
            direction_norm   = float(self._pos.direction)   # +1 or -1
            current_price    = self._price(self._idx)
            entry_dist = ((current_price - self._pos.entry_price)
                          * self._pos.direction           # positive = in profit
                          / max(self._pos.atr_at_entry, 1e-8))
            entry_dist = float(np.clip(entry_dist, -5.0, 5.0))
        else:
            direction_norm = 0.0
            entry_dist     = 0.0

        return np.append(market, [direction_norm, entry_dist]).astype(np.float32)

    def _unrealised_pnl(self) -> float:
        """Unrealised PnL in dollars (not used for reward, only for MAE tracking)."""
        if self._pos.direction == 0:
            return 0.0
        price = self._price(self._idx)
        return self._pos.direction * (price - self._pos.entry_price) * self._pos.size

    def _update_mae(self):
        """Update max adverse excursion for current position."""
        upnl = self._unrealised_pnl()
        if upnl < 0:
            adverse_dollars = abs(upnl)
            if adverse_dollars > self._pos.mae:
                self._pos.mae = adverse_dollars

    def _emergency_stop_triggered(self) -> bool:
        """True if unrealised loss exceeds the emergency stop threshold."""
        if self._pos.direction == 0:
            return False
        upnl = self._unrealised_pnl()
        if upnl >= 0:
            return False
        adverse = abs(upnl)
        threshold = min(
            EMERGENCY_ATR_MULT * self._pos.atr_at_entry * self._pos.size,
            EMERGENCY_MAX_PCT  * self._pos.equity_at_entry,
        )
        return adverse >= threshold

    def _open_position(self, direction: int, sizing: float) -> float:
        """
        Open a new position. Returns the cost (fee + slippage) charged.
        Updates self._equity and self._pos.
        """
        price      = self._price(self._idx)
        atr        = self._atr(self._idx)

        # Dollar risk per trade
        dollar_risk  = self._equity * sizing        # e.g. 2% of $10k = $200
        stop_dist    = EMERGENCY_ATR_MULT * atr     # ATR-based stop in price units
        if stop_dist < 1e-8:
            return 0.0

        # Position size in base units
        size_units   = dollar_risk / stop_dist
        notional     = size_units * price

        # Clamp: never put more than 95% of equity in a single position
        if notional > self._equity * 0.95:
            size_units = (self._equity * 0.95) / price
            notional   = size_units * price

        # Entry cost (fee + slippage) — exit will be charged on close
        cost_info = calc_trade_cost(
            entry_price     = price,
            exit_price      = price,   # placeholder, only entry cost used here
            position_size   = size_units,
            entry_vol_regime = self._vol_regime(self._idx),
            exit_vol_regime  = self._vol_regime(self._idx),
            entry_rel_volume = self._rel_volume(self._idx),
            exit_rel_volume  = self._rel_volume(self._idx),
            size_pct        = sizing,
            stress          = self.stress_fees,
        )
        entry_cost = cost_info["entry_fee"] + cost_info["entry_slippage"]

        self._equity -= entry_cost

        self._pos = PositionState(
            direction        = direction,
            entry_price      = price,
            entry_idx        = self._idx,
            size             = size_units,
            size_pct         = sizing,
            atr_at_entry     = atr,
            equity_at_entry  = self._equity,
            mae              = 0.0,
            vol_regime_entry = self._vol_regime(self._idx),
            rel_volume_entry = self._rel_volume(self._idx),
        )
        return entry_cost

    def _close_position(self, exit_price: float = None) -> float:
        """
        Close the current position. Returns net PnL (after all costs).
        exit_price: use this price instead of candle close (for TP/SL fills).
        Updates self._equity and resets self._pos.
        """
        price       = exit_price if exit_price is not None else self._price(self._idx)
        raw_pnl     = self._pos.direction * (price - self._pos.entry_price) * self._pos.size

        cost_info   = calc_trade_cost(
            entry_price      = self._pos.entry_price,
            exit_price       = price,
            position_size    = self._pos.size,
            entry_vol_regime = self._pos.vol_regime_entry,
            exit_vol_regime  = self._vol_regime(self._idx),
            entry_rel_volume = self._pos.rel_volume_entry,
            exit_rel_volume  = self._rel_volume(self._idx),
            size_pct         = self._pos.size_pct,
            stress           = self.stress_fees,
        )
        # Only exit costs here (entry was charged on open)
        exit_cost   = cost_info["exit_fee"] + cost_info["exit_slippage"]
        net_pnl     = raw_pnl - exit_cost

        self._equity += net_pnl
        self._equity  = max(self._equity, 0.0)   # floor at zero

        mae_saved   = self._pos.mae
        atr_saved   = self._pos.atr_at_entry

        self._pos   = PositionState()
        return net_pnl, mae_saved, atr_saved

    def _check_tp_sl(self) -> tuple:
        """
        Check if current candle's high/low touched the hard SL level.
        No hard TP — agent manages exits via Close action (discretionary profit-taking).
        Returns (triggered: bool, exit_price: float, reason: str).
        """
        if self._pos.direction == 0:
            return False, 0.0, ""

        entry = self._pos.entry_price
        h     = float(self.df.at[self._idx, "high"])
        l     = float(self.df.at[self._idx, "low"])

        if self._pos.direction == 1:   # long
            sl_price = entry * (1.0 - SL_PCT)
            if l <= sl_price:
                return True, sl_price, "sl"
        else:                          # short
            sl_price = entry * (1.0 + SL_PCT)
            if h >= sl_price:
                return True, sl_price, "sl"

        return False, 0.0, ""

    # ── Gym interface ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._idx          = self.start_idx
        end_raw            = len(self.df) - 1
        if self.episode_length is not None:
            self._end_idx  = min(self._idx + self.episode_length - 1, end_raw)
        else:
            self._end_idx  = end_raw

        self._equity       = self.initial_equity
        self._starting_eq  = self.initial_equity
        self._pos          = PositionState()
        self._candles_flat = 0

        return self._obs(), {}

    def step(self, action):
        direction_act = int(action["direction"])
        sizing        = float(np.clip(action["sizing"], 0.0, 1.0).flat[0])

        # Enforce minimum sizing threshold
        if direction_act in (LONG, SHORT) and sizing < MIN_SIZING:
            direction_act = HOLD

        # 4h trend gate — only trade WITH the 4h trend (top-down approach).
        # _4h_trend_up: +1 = 4h EMA9 > EMA21 (bullish), -1 = bearish, 0 = flat.
        # If the column is absent (old 15m data), gate is skipped.
        if "_4h_trend_up" in self.df.columns:
            trend = float(self.df.iloc[self._idx].get("_4h_trend_up", 0.0))
            if direction_act == LONG  and trend < 0:
                direction_act = HOLD   # 4h bearish → block long
            elif direction_act == SHORT and trend > 0:
                direction_act = HOLD   # 4h bullish → block short

        reward   = 0.0
        info     = {}

        # ── MAE update (before acting) ─────────────────────────────────────────
        self._update_mae()

        # ── Hard TP/SL check (primary exit — 4% TP, 2% SL) ───────────────────
        tp_sl_hit, tp_sl_price, tp_sl_reason = self._check_tp_sl()
        if tp_sl_hit:
            net_pnl, mae, atr_entry = self._close_position(exit_price=tp_sl_price)
            result  = TradeResult(
                net_pnl                = net_pnl,
                account_equity         = self._equity,
                max_adverse_excursion  = mae,
                atr_at_entry           = atr_entry,
            )
            reward  = reward_on_close(result)
            info[tp_sl_reason]     = True
            self._candles_flat     = 0
            direction_act = HOLD

        # ── Emergency stop (backup for gap-through / extreme moves) ───────────
        elif self._emergency_stop_triggered():
            net_pnl, mae, atr_entry = self._close_position()
            result  = TradeResult(
                net_pnl                = net_pnl,
                account_equity         = self._equity,
                max_adverse_excursion  = mae,
                atr_at_entry           = atr_entry,
            )
            reward  = reward_on_close(result)
            info["emergency_stop"] = True
            self._candles_flat     = 0
            direction_act = HOLD   # nothing else to do this step

        # ── Max hold check ─────────────────────────────────────────────────────
        elif (self._pos.direction != 0 and
              (self._idx - self._pos.entry_idx) >= MAX_HOLD_CANDLES):
            net_pnl, mae, atr_entry = self._close_position()
            result  = TradeResult(
                net_pnl                = net_pnl,
                account_equity         = self._equity,
                max_adverse_excursion  = mae,
                atr_at_entry           = atr_entry,
            )
            reward  = reward_on_close(result)
            info["max_hold_close"] = True
            self._candles_flat     = 0
            direction_act = HOLD

        # ── Agent action ───────────────────────────────────────────────────────
        if direction_act != HOLD:

            if direction_act == CLOSE:
                if self._pos.direction != 0:
                    net_pnl, mae, atr_entry = self._close_position()
                    result  = TradeResult(
                        net_pnl                = net_pnl,
                        account_equity         = self._equity,
                        max_adverse_excursion  = mae,
                        atr_at_entry           = atr_entry,
                    )
                    reward += reward_on_close(result)
                    self._candles_flat = 0

            elif direction_act == LONG:
                if self._pos.direction == -1:
                    # Flip: close short, then open long
                    net_pnl, mae, atr_entry = self._close_position()
                    result  = TradeResult(
                        net_pnl                = net_pnl,
                        account_equity         = self._equity,
                        max_adverse_excursion  = mae,
                        atr_at_entry           = atr_entry,
                    )
                    reward += reward_on_close(result)
                    self._open_position(1, sizing)
                    info["flip"] = "short→long"
                elif self._pos.direction == 0:
                    self._open_position(1, sizing)
                # if already long — ignore

            elif direction_act == SHORT:
                if self._pos.direction == 1:
                    # Flip: close long, then open short
                    net_pnl, mae, atr_entry = self._close_position()
                    result  = TradeResult(
                        net_pnl                = net_pnl,
                        account_equity         = self._equity,
                        max_adverse_excursion  = mae,
                        atr_at_entry           = atr_entry,
                    )
                    reward += reward_on_close(result)
                    self._open_position(-1, sizing)
                    info["flip"] = "long→short"
                elif self._pos.direction == 0:
                    self._open_position(-1, sizing)
                # if already short — ignore

        # ── Inactivity cost (flat position) ────────────────────────────────────
        if self._pos.direction == 0:
            self._candles_flat += 1
            reward += reward_on_hold(self._candles_flat)
        else:
            # Small bonus for holding a profitable position — teaches agent to
            # let winners run rather than closing at first sign of profit
            upnl = self._unrealised_pnl()
            if upnl > 0:
                from env.reward import HOLD_PROFIT_BONUS
                reward += HOLD_PROFIT_BONUS

        # ── Circuit breaker ────────────────────────────────────────────────────
        cb_triggered, cb_penalty = check_circuit_breaker(self._equity, self._starting_eq)
        if cb_triggered:
            reward    += cb_penalty
            terminated = True
            info["circuit_breaker"] = True
        else:
            terminated = False

        # ── Advance clock ──────────────────────────────────────────────────────
        self._idx += 1
        truncated  = self._idx > self._end_idx

        obs  = self._obs() if not (terminated or truncated) else np.zeros(
            N_FEATURES + 2, dtype=np.float32
        )

        info.update({
            "equity":     self._equity,
            "position":   self._pos.direction,
            "step":       self._idx,
        })

        return obs, float(reward), terminated, truncated, info

    def render(self):
        pos_str = {0: "FLAT", 1: "LONG", -1: "SHORT"}[self._pos.direction]
        print(f"[{self._idx:>6}]  equity={self._equity:>10,.2f}  pos={pos_str}")

    # ── Episode metrics (call after episode ends) ───────────────────────────────

    def episode_return(self) -> float:
        return (self._equity - self._starting_eq) / self._starting_eq

    def final_equity(self) -> float:
        return self._equity
