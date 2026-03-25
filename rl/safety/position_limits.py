"""
position_limits.py
──────────────────
Hard position limits enforced BEFORE any order reaches the exchange.
These are non-negotiable guardrails — the Trust Ladder can restrict
further but can never relax these hard caps.

Limits:
  - Max 2 open positions total (across all coins)
  - Max 1 position per coin
  - Max position size: leverage_cap × available_equity
  - Max daily loss: 3% of starting equity for the day
  - Max single trade risk: 2% of equity (matches env sizing)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import time


# ── Hard caps (never overridden) ──────────────────────────────────────────────

MAX_OPEN_POSITIONS    = 2       # across all coins simultaneously
MAX_POSITIONS_PER_COIN = 1
MAX_SINGLE_RISK_PCT   = 0.02   # 2% of equity per trade
MAX_DAILY_LOSS_PCT    = 0.03   # 3% daily drawdown → halt trading for the day
BASE_LEVERAGE_CAP     = 1.0    # spot-equivalent (1× = no leverage)


@dataclass
class OpenPosition:
    coin:         str
    direction:    int    # 1=long, -1=short
    entry_price:  float
    size:         float  # base units
    notional:     float  # dollars at entry
    entry_time:   float  # unix timestamp


@dataclass
class PositionLimitState:
    """Mutable state tracked by PositionLimits."""
    open_positions:   Dict[str, OpenPosition] = field(default_factory=dict)
    day_start_equity: float = 0.0
    day_start_ts:     float = field(default_factory=time.time)
    daily_loss:       float = 0.0    # cumulative realised loss today (negative = loss)
    halted_until_eod: bool  = False


class PositionLimits:
    """
    Stateful hard-limit checker.

    Usage:
        limits = PositionLimits(starting_equity=10_000)
        ok, reason = limits.check_entry("BTC", direction=1, sizing=0.02, equity=10_000)
        if ok:
            limits.record_open("BTC", ...)
        ...
        limits.record_close("BTC", pnl=-50)
    """

    def __init__(self, starting_equity: float, leverage_cap: float = BASE_LEVERAGE_CAP):
        self.leverage_cap = leverage_cap
        self._state = PositionLimitState(
            day_start_equity = starting_equity,
            day_start_ts     = time.time(),
        )

    # ── Day boundary reset ────────────────────────────────────────────────────

    def _maybe_reset_day(self, current_equity: float):
        """Reset daily loss counter at UTC midnight."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        day_start = datetime.datetime.combine(
            now.date(),
            datetime.time.min,
            tzinfo=datetime.timezone.utc,
        ).timestamp()

        if self._state.day_start_ts < day_start:
            self._state.day_start_equity = current_equity
            self._state.day_start_ts     = day_start
            self._state.daily_loss       = 0.0
            self._state.halted_until_eod = False

    # ── Entry check ───────────────────────────────────────────────────────────

    def check_entry(
        self,
        coin:    str,
        direction: int,   # 1=long, -1=short
        sizing:  float,   # fraction of equity to risk
        equity:  float,
    ) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        reason is empty string if allowed.
        """
        self._maybe_reset_day(equity)
        s = self._state

        # Daily halt
        if s.halted_until_eod:
            return False, "daily loss limit reached — halted until EOD"

        # Daily loss check
        daily_dd = s.daily_loss / max(s.day_start_equity, 1e-8)
        if daily_dd <= -MAX_DAILY_LOSS_PCT:
            s.halted_until_eod = True
            return False, f"daily loss {daily_dd*100:.2f}% exceeds -{MAX_DAILY_LOSS_PCT*100:.0f}%"

        # Already in this coin
        if coin in s.open_positions:
            return False, f"already have open position in {coin}"

        # Max concurrent positions
        if len(s.open_positions) >= MAX_OPEN_POSITIONS:
            return False, f"max {MAX_OPEN_POSITIONS} concurrent positions reached"

        # Single trade risk cap
        effective_sizing = min(sizing, MAX_SINGLE_RISK_PCT)
        if sizing > MAX_SINGLE_RISK_PCT:
            # We'll clamp it, not block it — return a warning via reason
            # The caller should use effective_sizing
            pass

        # Notional cap via leverage
        max_notional = equity * self.leverage_cap
        trade_notional = equity * effective_sizing
        if trade_notional > max_notional:
            return False, f"notional ${trade_notional:,.0f} exceeds leverage cap ${max_notional:,.0f}"

        return True, ""

    def clamped_sizing(self, sizing: float) -> float:
        """Return sizing clamped to MAX_SINGLE_RISK_PCT."""
        return min(sizing, MAX_SINGLE_RISK_PCT)

    # ── State updates ─────────────────────────────────────────────────────────

    def record_open(
        self,
        coin:        str,
        direction:   int,
        entry_price: float,
        size:        float,
        notional:    float,
    ):
        self._state.open_positions[coin] = OpenPosition(
            coin         = coin,
            direction    = direction,
            entry_price  = entry_price,
            size         = size,
            notional     = notional,
            entry_time   = time.time(),
        )

    def record_close(self, coin: str, pnl: float):
        """Call when a position is closed. pnl in dollars."""
        self._state.open_positions.pop(coin, None)
        if pnl < 0:
            self._state.daily_loss += pnl

    # ── Inspection ────────────────────────────────────────────────────────────

    @property
    def n_open(self) -> int:
        return len(self._state.open_positions)

    @property
    def is_halted(self) -> bool:
        return self._state.halted_until_eod

    @property
    def daily_loss_pct(self) -> float:
        return self._state.daily_loss / max(self._state.day_start_equity, 1e-8)

    def status(self) -> dict:
        return {
            "open_positions":   list(self._state.open_positions.keys()),
            "n_open":           self.n_open,
            "daily_loss_pct":   self.daily_loss_pct,
            "halted":           self.is_halted,
            "leverage_cap":     self.leverage_cap,
        }
