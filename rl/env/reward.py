"""
reward.py
─────────
4-component reward function. Separated from the environment for easy tuning.

Philosophy: Reward the trade. Evaluate the portfolio.
  - Most steps return 0.0
  - Primary signal fires on position CLOSE
  - Portfolio metrics (Sharpe, drawdown) are validation metrics, NOT reward components

Components:
  1. Trade PnL         — fires on close. Net of fees and slippage. Normalised by equity.
  2. MAE adjustment    — multiplicative. Penalises trades that went far against you
                         even if they ultimately won. Teaches conviction, not luck.
  3. Inactivity cost   — small per-step penalty after >6h flat. Prevents hibernation.
  4. Circuit breaker   — episode termination signal at -5% equity. One-time penalty.

What is NOT in the reward (by design):
  - Sharpe bonus         → portfolio metric, not per-step
  - Win rate bonus       → creates early-close bias
  - Unrealised PnL       → teaches position-chasing, not edge
  - Consecutive-win bonus → creates correlation-seeking
  - Trade frequency bonus → inactivity cost handles the floor
"""

from dataclasses import dataclass


# ── Tuneable constants ─────────────────────────────────────────────────────────

REWARD_SCALE       = 100.0    # scale normalised PnL for usable gradient signal
INACTIVITY_GRACE   = 6        # candles flat before penalty kicks in (~6h at 1h TF)
INACTIVITY_RATE    = 0.005    # flat penalty per candle beyond grace period
HOLD_PROFIT_BONUS  = 0.0003   # small per-step reward while in a profitable position
                               # teaches agent to hold winners, not cut early
CIRCUIT_BREAKER    = 0.95     # terminate if equity < starting_equity × this
CIRCUIT_PENALTY    = -5.0     # one-time reward on circuit breaker trigger
MAE_HALF_CREDIT    = 0.5      # multiplier for "won but risked too much" trades


@dataclass
class TradeResult:
    """All inputs needed to compute the close-time reward."""
    net_pnl:            float   # after fees and slippage (in dollars)
    account_equity:     float   # at time of close
    max_adverse_excursion: float # worst unrealised loss during the trade (dollars)
    atr_at_entry:       float   # ATR(14) at time of entry (dollars per unit)


def reward_on_close(result: TradeResult) -> float:
    """
    Component 1 + 2: PnL reward with MAE risk adjustment.
    Called once when a position is closed.

    Returns a scaled reward signal.
    """
    equity = max(result.account_equity, 1e-6)

    # Component 1: normalised net PnL
    normalised_pnl = result.net_pnl / equity

    # Component 2: MAE risk adjustment
    # How many ATRs did the trade go against us at its worst?
    atr_safe = max(result.atr_at_entry, 1e-8)
    mae_atr  = result.max_adverse_excursion / atr_safe

    if normalised_pnl > 0 and mae_atr > 0:
        # Won the trade. But did we risk too much to get here?
        # risk_ratio < 1 means we "earned less than we risked in ATR terms"
        risk_ratio = normalised_pnl / (mae_atr * 0.01 + 1e-8)
        if risk_ratio < 1.0:
            reward = normalised_pnl * MAE_HALF_CREDIT   # half credit
        else:
            reward = normalised_pnl                     # full credit
    elif normalised_pnl <= 0:
        # Loss — no adjustment needed, loss is punishment enough
        reward = normalised_pnl
    else:
        # Trade never went against us (MAE = 0) — full credit, clean entry
        reward = normalised_pnl

    return float(reward * REWARD_SCALE)


def reward_on_hold(candles_flat: int) -> float:
    """
    Component 3: Inactivity cost for flat positions.
    Called every step when position == 0.

    Returns a small negative reward after grace period, 0.0 within grace period.
    This prevents the agent from hibernating but never forces bad trades
    (a single bad trade costs 100-500× more than 24h of inactivity penalty).
    """
    if candles_flat <= INACTIVITY_GRACE:
        return 0.0
    return -INACTIVITY_RATE


def check_circuit_breaker(
    account_equity: float,
    starting_equity: float,
) -> tuple[bool, float]:
    """
    Component 4: Circuit breaker.
    Called every step.

    Returns (triggered: bool, penalty: float).
    When triggered: episode should terminate, agent receives CIRCUIT_PENALTY.
    """
    if account_equity < starting_equity * CIRCUIT_BREAKER:
        return True, CIRCUIT_PENALTY
    return False, 0.0
