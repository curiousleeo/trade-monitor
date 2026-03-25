"""
trust_ladder.py
───────────────
The Trust Ladder governs how much leverage and autonomy the agent earns
over time based on its live performance.

Philosophy: the agent starts restricted and earns trust through results.
It can be demoted back down automatically if performance deteriorates.

Rungs (4 levels):
  0 — Restricted   0.5× leverage,  paper-only,   max 1 position
  1 — Probation    0.5× leverage,  live allowed, max 1 position
  2 — Standard     1.0× leverage,  live,         max 2 positions
  3 — Extended     2.0× leverage,  live,         max 2 positions
                   (only reachable after 90 days on Standard+)

Promotion criteria (must hold for 30-day rolling window):
  - Net profitable (return > 0)
  - Sharpe ≥ 0.5
  - Max drawdown < 5%
  - ≥ 20 closed trades

Demotion triggers (immediate, any of):
  - Single-day loss > 2% (one rung down)
  - Weekly loss > 4%     (two rungs down, floor at 0)
  - Max drawdown ≥ 8%    (back to Rung 0)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import time
import numpy as np


# ── Rung definitions ──────────────────────────────────────────────────────────

@dataclass
class Rung:
    id:             int
    name:           str
    leverage_cap:   float
    max_positions:  int
    paper_only:     bool


RUNGS = [
    Rung(0, "Restricted", leverage_cap=0.5, max_positions=1, paper_only=True),
    Rung(1, "Probation",  leverage_cap=0.5, max_positions=1, paper_only=False),
    Rung(2, "Standard",   leverage_cap=1.0, max_positions=2, paper_only=False),
    Rung(3, "Extended",   leverage_cap=2.0, max_positions=2, paper_only=False),
]

# Promotion thresholds (30-day rolling)
PROMO_MIN_TRADES    = 20
PROMO_MIN_SHARPE    = 0.5
PROMO_MAX_DD        = 0.05
PROMO_WINDOW_DAYS   = 30

# Demotion triggers
DEMOTE_DAILY_LOSS   = 0.02   # single day loss > 2% → -1 rung
DEMOTE_WEEKLY_LOSS  = 0.04   # week loss > 4%       → -2 rungs
DEMOTE_DD_RESET     = 0.08   # drawdown ≥ 8%        → rung 0

# Extended rung requires this many consecutive days on Standard+
EXTENDED_MIN_DAYS   = 90


@dataclass
class TradeRecord:
    """A closed trade's outcome."""
    timestamp:  float   # unix
    pnl:        float   # dollars
    equity:     float   # equity after close


@dataclass
class TrustState:
    rung:               int   = 1   # start on Probation
    days_on_standard:   int   = 0
    trade_history:      List[TradeRecord] = field(default_factory=list)
    equity_curve:       List[float]       = field(default_factory=list)
    last_promotion_ts:  float             = field(default_factory=time.time)
    last_check_ts:      float             = field(default_factory=time.time)


class TrustLadder:
    """
    Tracks live performance and manages rung promotions/demotions.

    Call after every closed trade:
        ladder.record_trade(pnl, equity)

    Call after every day end:
        ladder.end_of_day(day_pnl, day_start_equity)

    Query current permissions:
        ladder.leverage_cap
        ladder.max_positions
        ladder.is_paper_only
        ladder.current_rung
    """

    def __init__(self, starting_rung: int = 1):
        self._state = TrustState(rung=max(0, min(starting_rung, 3)))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_rung(self) -> Rung:
        return RUNGS[self._state.rung]

    @property
    def leverage_cap(self) -> float:
        return self.current_rung.leverage_cap

    @property
    def max_positions(self) -> int:
        return self.current_rung.max_positions

    @property
    def is_paper_only(self) -> bool:
        return self.current_rung.paper_only

    # ── Event handlers ────────────────────────────────────────────────────────

    def record_trade(self, pnl: float, equity: float):
        """Call each time a position closes."""
        s = self._state
        s.trade_history.append(TradeRecord(
            timestamp = time.time(),
            pnl       = pnl,
            equity    = equity,
        ))
        s.equity_curve.append(equity)

        # Trim history older than 90 days
        cutoff = time.time() - 90 * 86400
        s.trade_history = [t for t in s.trade_history if t.timestamp > cutoff]

    def end_of_day(self, day_pnl: float, day_start_equity: float,
                   week_pnl: float = 0.0):
        """
        Call at end of each trading day.
        day_pnl: net realised PnL for the day (dollars)
        day_start_equity: equity at day open
        week_pnl: net realised PnL for the rolling 7 days (optional)
        """
        s = self._state

        day_loss_pct  = day_pnl  / max(day_start_equity, 1e-8)
        week_loss_pct = week_pnl / max(day_start_equity, 1e-8)

        # ── Demotion checks ───────────────────────────────────────────────────
        dd = self._rolling_drawdown(days=30)

        if dd >= DEMOTE_DD_RESET:
            self._set_rung(0, f"drawdown {dd*100:.1f}% ≥ {DEMOTE_DD_RESET*100:.0f}%")
            return

        if week_loss_pct <= -DEMOTE_WEEKLY_LOSS:
            new_rung = max(0, s.rung - 2)
            self._set_rung(new_rung,
                           f"weekly loss {week_loss_pct*100:.1f}% ≥ -{DEMOTE_WEEKLY_LOSS*100:.0f}%")
            return

        if day_loss_pct <= -DEMOTE_DAILY_LOSS:
            new_rung = max(0, s.rung - 1)
            self._set_rung(new_rung,
                           f"daily loss {day_loss_pct*100:.1f}% ≥ -{DEMOTE_DAILY_LOSS*100:.0f}%")
            return

        # ── Promotion check (only runs once per day) ──────────────────────────
        if s.rung < 3:
            self._check_promotion()

        # Track days on Standard+ for Extended eligibility
        if s.rung >= 2:
            s.days_on_standard += 1

    def _check_promotion(self):
        """Try to promote one rung if 30-day metrics qualify."""
        s = self._state

        trades_30d = self._trades_in_window(days=PROMO_WINDOW_DAYS)
        if len(trades_30d) < PROMO_MIN_TRADES:
            return

        sharpe = self._rolling_sharpe(days=PROMO_WINDOW_DAYS)
        dd     = self._rolling_drawdown(days=PROMO_WINDOW_DAYS)
        net    = sum(t.pnl for t in trades_30d)

        qualifies = (net > 0 and sharpe >= PROMO_MIN_SHARPE and dd < PROMO_MAX_DD)

        if not qualifies:
            return

        # Rung 3 (Extended) requires 90 days on Standard+
        if s.rung == 2 and s.days_on_standard < EXTENDED_MIN_DAYS:
            return

        new_rung = min(s.rung + 1, 3)
        self._set_rung(new_rung,
                       f"30d: net={net:+.2f}, sharpe={sharpe:.2f}, dd={dd*100:.1f}%")

    def _set_rung(self, new_rung: int, reason: str):
        old = self._state.rung
        self._state.rung = new_rung
        direction = "promoted" if new_rung > old else "demoted"
        print(
            f"  [TrustLadder] {direction}: "
            f"{RUNGS[old].name} → {RUNGS[new_rung].name}  ({reason})",
            flush=True,
        )

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _trades_in_window(self, days: int) -> list:
        cutoff = time.time() - days * 86400
        return [t for t in self._state.trade_history if t.timestamp > cutoff]

    def _rolling_sharpe(self, days: int) -> float:
        trades = self._trades_in_window(days)
        if len(trades) < 5:
            return 0.0
        pnls = np.array([t.pnl for t in trades])
        if pnls.std() < 1e-10:
            return 0.0
        # Rough annualisation: assume ~8 trades/day → 8×365 = 2920/year
        return float((pnls.mean() / pnls.std()) * np.sqrt(2920))

    def _rolling_drawdown(self, days: int) -> float:
        trades = self._trades_in_window(days)
        if len(trades) < 2:
            return 0.0
        eq   = np.array([t.equity for t in trades])
        peak = np.maximum.accumulate(eq)
        dd   = (peak - eq) / (peak + 1e-8)
        return float(dd.max())

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        s = self._state
        return {
            "rung":            s.rung,
            "rung_name":       self.current_rung.name,
            "leverage_cap":    self.leverage_cap,
            "max_positions":   self.max_positions,
            "paper_only":      self.is_paper_only,
            "days_on_standard": s.days_on_standard,
            "trades_30d":      len(self._trades_in_window(30)),
            "sharpe_30d":      round(self._rolling_sharpe(30), 3),
            "drawdown_30d":    round(self._rolling_drawdown(30), 4),
        }
