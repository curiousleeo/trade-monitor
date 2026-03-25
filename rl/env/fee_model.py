"""
fee_model.py
────────────
Fee and slippage model for Binance USDT-M Futures.

Design principles:
- Agent always pays TAKER (market orders only)
- Both entry AND exit incur fees
- Slippage adapts to market conditions (volatility, liquidity, position size)
- Stress-test mode doubles slippage to verify edge robustness (validation check #8)
"""

# ── Constants ─────────────────────────────────────────────────────────────────

TAKER_FEE  = 0.0005   # 0.05% — Binance USDT-M Futures default tier
MAKER_FEE  = 0.0002   # 0.02% — not used (agent never places limit orders)


def calc_slippage(
    volatility_regime:  float,
    relative_volume:    float,
    position_size_pct:  float,
    stress:             bool = False,
) -> float:
    """
    Dynamic slippage model. Returns slippage as a fraction (e.g. 0.0003 = 0.03%).

    Args:
        volatility_regime:  ATR(14)/ATR(50) on 1h. >1.5 = expansion, <0.7 = chop.
                            Comes from the observation vector (vol_regime feature).
        relative_volume:    volume / SMA(vol, 20) on 15m, normalised to [0,1] in
                            features.py. Un-normalise here: actual = rel_vol * 3.
        position_size_pct:  raw sizing output from agent, in (0, 1).
        stress:             If True, double the result. Used for validation check #8.

    Returns:
        Slippage fraction. Multiply by (position_notional) to get dollar cost.
    """
    base = 0.0003   # 0.03% baseline — BTC/ETH perpetuals are very liquid

    # High volatility → worse fills (wider spreads, faster-moving book)
    # vol_regime is already ATR14/ATR50 ratio; un-normalise from [0,1] back to [0,4]
    vol_ratio  = volatility_regime * 4.0
    vol_mult   = max(1.0, vol_ratio)

    # Low liquidity → worse fills
    # relative_volume in features.py is clipped(vol/sma20, 0,3)/3 → [0,1]
    # un-normalise: actual_rel_vol = rel_volume * 3
    actual_rel_vol = max(relative_volume * 3.0, 0.05)   # avoid div/0
    liq_mult = max(1.0, 1.0 / actual_rel_vol)

    # Larger position → more market impact
    size_mult  = 1.0 + position_size_pct * 0.5

    slippage = base * vol_mult * liq_mult * size_mult

    # Hard cap — if we'd exceed this the pre-trade gate should have blocked it
    slippage = min(slippage, 0.01)   # max 1%

    if stress:
        slippage *= 2.0

    return slippage


def calc_trade_cost(
    entry_price:        float,
    exit_price:         float,
    position_size:      float,   # in base units (e.g. BTC)
    entry_vol_regime:   float,
    exit_vol_regime:    float,
    entry_rel_volume:   float,
    exit_rel_volume:    float,
    size_pct:           float,   # agent's raw sizing output
    stress:             bool = False,
) -> dict:
    """
    Compute all execution costs for a round-trip trade.

    Returns a dict with individual cost components for logging.
    """
    notional_entry = abs(position_size * entry_price)
    notional_exit  = abs(position_size * exit_price)

    entry_fee      = notional_entry * TAKER_FEE
    exit_fee       = notional_exit  * TAKER_FEE

    entry_slip_pct = calc_slippage(entry_vol_regime, entry_rel_volume, size_pct, stress)
    exit_slip_pct  = calc_slippage(exit_vol_regime,  exit_rel_volume,  size_pct, stress)
    entry_slippage = notional_entry * entry_slip_pct
    exit_slippage  = notional_exit  * exit_slip_pct

    total = entry_fee + exit_fee + entry_slippage + exit_slippage

    return {
        "entry_fee":      entry_fee,
        "exit_fee":       exit_fee,
        "entry_slippage": entry_slippage,
        "exit_slippage":  exit_slippage,
        "total":          total,
    }
