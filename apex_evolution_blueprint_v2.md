# APEX AI TRADING SYSTEM — EVOLUTION BLUEPRINT v2.0
## From DQN to PPO Hybrid | Complete Build Specification

**Target**: Live capital trading on Binance perpetual futures
**Pairs**: BTC/USDT + ETH/USDT
**Approach**: Evolve existing codebase — keep what works, replace what doesn't
**Framework**: CleanRL PPO (modified for hybrid discrete-continuous action space)
**Capital**: System is capital-agnostic (percentage-based everything)

---

## WHAT EXISTS TODAY (Do Not Rebuild)

These files are working and tested. The evolution builds ON TOP of them.

| Existing File | Status | Evolution Plan |
|---|---|---|
| `fetch_data.py` | ✅ KEEP | Add funding rate download endpoint |
| `features.py` | 🔄 EXTEND | Merge existing 18 features into new 38-feature design |
| `environment.py` | 🔄 REWRITE | New env with hybrid action space, no TP/SL, new reward |
| `dqn.py` | ❌ REPLACE | Replace with PPO hybrid agent |
| `train.py` | 🔄 REWRITE | Walk-forward training pipeline |
| `evaluate.py` | 🔄 EXTEND | Add 8-check validation suite |
| Terminal (React) | ✅ KEEP | Will connect to new agent in a later phase |

---

## KNOWN PROBLEMS WITH CURRENT SYSTEM (Why We're Evolving)

These are the specific failure modes visible in the current backtest results:

1. **ETH +2,468% return with -61% max drawdown** — untradeable. Nobody holds through 61% DD.
2. **SOL -77% max drawdown** — the agent learned to gamble, not trade.
3. **BTC 500k steps worse than 200k** — classic overfitting. Longer training = memorization.
4. **Win rates 47-50% with huge returns** — means a few giant trades carry everything. One lucky hit, not consistent edge.
5. **DQN can't do position sizing** — every trade is identical size, regardless of conviction.
6. **Fixed 2% risk + ATR TP/SL** — the agent never learned risk management, the rules did it for it.
7. **No safety layer** — nothing prevents the agent from blowing up on live capital.
8. **Single train/test split** — one 80/20 split can't detect overfitting across regimes.

The evolution fixes ALL of these.

---

## SYSTEM 1: OBSERVATION SPACE (38 features)

The agent's entire reality. No raw prices. No account balance. No unrealized PnL.

### 1A. Market Context — Z-Scored OHLCV (15 features)

Rolling z-score with 100-period lookback per timeframe.

```python
def zscore(series, lookback=100):
    """Stationary normalization — works across any price regime."""
    rolling_mean = series.rolling(lookback).mean()
    rolling_std = series.rolling(lookback).std()
    return (series - rolling_mean) / (rolling_std + 1e-8)
```

| Timeframe | Features | Count |
|-----------|----------|-------|
| 15m | open_z, high_z, low_z, close_z, volume_z | 5 |
| 1h  | open_z, high_z, low_z, close_z, volume_z | 5 |
| 4h  | open_z, high_z, low_z, close_z, volume_z | 5 |

**Timeframe alignment rule**: 15m is the step clock. Higher timeframes update ONLY when their candle closes. Between closes, the agent sees the last COMPLETED candle. Never use forming/partial candles — that's lookahead bias.

### 1B. Technical Indicators (6 features)

Carried over from existing `features.py` with normalization changes.

| Timeframe | Feature | Normalization |
|-----------|---------|---------------|
| 15m | RSI(14) | (rsi - 50) / 50 → [-1, 1] |
| 15m | MACD histogram | z-scored over 100 periods |
| 15m | EMA(9) - EMA(21) spread | divided by close, then z-scored |
| 1h  | RSI(14) | same |
| 1h  | MACD histogram | same |
| 1h  | EMA(9) - EMA(21) spread | same |

### 1C. Candle Shape (9 features) — FROM EXISTING SYSTEM

These are the valuable features from your current `features.py` that our initial design missed. Candle shape carries information that z-scored OHLCV does not — specifically about conviction and rejection.

| Timeframe | Feature | Calculation | Range |
|-----------|---------|-------------|-------|
| 15m | Body ratio | abs(close - open) / (high - low + 1e-8) | [0, 1] |
| 15m | Upper wick ratio | (high - max(open, close)) / (high - low + 1e-8) | [0, 1] |
| 15m | Lower wick ratio | (min(open, close) - low) / (high - low + 1e-8) | [0, 1] |
| 1h  | Body ratio | same | [0, 1] |
| 1h  | Upper wick ratio | same | [0, 1] |
| 1h  | Lower wick ratio | same | [0, 1] |
| 4h  | Body ratio | same | [0, 1] |
| 4h  | Upper wick ratio | same | [0, 1] |
| 4h  | Lower wick ratio | same | [0, 1] |

No normalization needed — these are naturally bounded [0, 1].

### 1D. Regime & Context (4 features) — NEW

| Feature | Calculation | Purpose |
|---------|-------------|---------|
| Volatility regime | ATR(14) / ATR(50) on 1h | >1.5 = volatile expansion, <0.7 = compression/chop |
| Relative volume | volume / SMA(volume, 20) on 15m | >2.0 = breakout confirmation, <0.5 = dead market |
| Funding rate | Raw Binance funding rate, clipped [-0.001, 0.001], then / 0.001 → [-1, 1] | Crowding signal. Extreme positive = longs crowded. Extreme negative = shorts crowded |
| Cross-pair momentum | Other pair's 15m log-return, z-scored over 100 periods | BTC↔ETH lead/lag. ETH agent sees BTC momentum, BTC agent sees ETH momentum |

### 1E. Time Encoding (2 features) — FROM EXISTING SYSTEM

```python
hour = timestamp.hour + timestamp.minute / 60.0
time_sin = sin(2 * pi * hour / 24)
time_cos = cos(2 * pi * hour / 24)
```

Captures session effects (Asian/European/US) and funding rate settlement times (00:00, 08:00, 16:00 UTC on Binance).

### 1F. Position State (2 features) — NEW DESIGN

| Feature | Encoding | Notes |
|---------|----------|-------|
| Direction | -1 (short), 0 (flat), 1 (long) | Raw integer |
| Entry distance | (current_price - entry_price) / ATR(14) on 15m | 0 when flat. Captures how extended the trade is relative to current volatility |

**WHAT IS NOT IN THE OBSERVATION SPACE (by design):**
- ❌ Unrealized PnL — creates emotional feedback loop
- ❌ Account balance — agent reads market, not its bank account
- ❌ Drawdown percentage — same as PnL, creates fear-based trading
- ❌ Time-in-trade — the entry distance + market context already encode this implicitly

### 1G. NaN / Inf Handling

```python
obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
obs = np.clip(obs, -5.0, 5.0)
```

Applied as the final step before passing to the agent. Every feature must be bounded.

---

## SYSTEM 2: ACTION SPACE (Hybrid Discrete-Continuous)

### Architecture

The actor network has TWO output heads sharing a common backbone:

```
Input (38 features)
    → Linear(38, 256) → Tanh
    → Linear(256, 256) → Tanh
    ├── Direction Head → Linear(256, 4) → Categorical distribution
    └── Sizing Head → Linear(256, 2) → Softplus → Beta(alpha, beta) distribution
```

### Direction Head — Discrete (4 actions)

| Action | Code | Behavior |
|--------|------|----------|
| Hold | 0 | Do nothing. Sizing head output ignored. |
| Long | 1 | Open long (or close short + open long if currently short). |
| Short | 2 | Open short (or close long + open short if currently long). |
| Close | 3 | Close current position. Sizing head output ignored. No-op if flat. |

### Sizing Head — Continuous via Beta Distribution

```python
# Network outputs two raw values
raw_alpha, raw_beta = sizing_head(shared_features)  # shape: (2,)

# Softplus ensures positive, +1 ensures alpha,beta >= 1 (unimodal distribution)
alpha = softplus(raw_alpha) + 1.0
beta = softplus(raw_beta) + 1.0

# Sample position size from Beta distribution
raw_size = Beta(alpha, beta).sample()  # naturally in (0, 1)

# Minimum conviction gate
if raw_size < 0.1:
    action = Hold  # override — conviction too low to trade

# Effective position size (applied by environment, NOT agent)
effective_size = raw_size * max_position_pct * leverage_multiplier
# max_position_pct = 1.0 (100% of available equity)
# leverage_multiplier = from Trust Ladder (0.5x to 2.0x, invisible to agent)
```

**Why Beta distribution:**
- Naturally bounded (0, 1) — no clipping artifacts
- Two learnable parameters express the full range of conviction
- alpha = beta = 1 → uniform (maximum uncertainty)
- alpha >> beta → skewed high (strong conviction)
- alpha << beta → skewed low (weak conviction)
- Gradients flow cleanly through reparameterization

### Log Probability for PPO

```python
total_log_prob = direction_dist.log_prob(direction) + sizing_dist.log_prob(raw_size)
total_entropy = direction_dist.entropy() + sizing_dist.entropy()
```

Both heads contribute to the policy gradient. Both are updated by PPO.

### Position Rules

- One position at a time per pair
- Long while short → atomically: close short (pay fees) + open long (pay fees) in same step
- Short while long → same, reversed
- Close while flat → no-op, no penalty
- Hold while in position → position continues, no action taken

---

## SYSTEM 3: REWARD FUNCTION

**Philosophy: Reward the trade. Evaluate the portfolio.**

Reward is event-driven. Most steps return 0.0. Signal fires primarily on position close.

### Component 1: Trade PnL (fires on close)

```python
# Gross P&L
gross_pnl = (exit_price - entry_price) * direction * position_size

# Fees (Binance USDⓈ-M Futures, taker)
entry_fee = abs(position_size * entry_price) * TAKER_FEE   # 0.05%
exit_fee = abs(position_size * exit_price) * TAKER_FEE     # 0.05%

# Dynamic slippage (see System 6)
entry_slippage = calc_slippage(entry_volatility, entry_rel_volume, raw_size)
exit_slippage = calc_slippage(exit_volatility, exit_rel_volume, raw_size)
total_slippage_cost = (entry_slippage + exit_slippage) * abs(position_size) * exit_price

# Net
net_pnl = gross_pnl - entry_fee - exit_fee - total_slippage_cost

# Normalize by equity (capital-agnostic)
normalized_pnl = net_pnl / account_equity
```

### Component 2: Risk Adjustment (fires on close, multiplicative)

```python
# MAE = Max Adverse Excursion: worst unrealized loss during the trade
# Tracked every candle while position is open
# Normalized by ATR at entry time

mae_atr = max_adverse_excursion / atr_at_entry  # how many ATRs against you

if normalized_pnl > 0 and mae_atr > 0:
    risk_ratio = normalized_pnl / (mae_atr * 0.01)  # scale mae_atr to comparable units
    if risk_ratio < 1.0:
        # Won, but risked more than you made — half credit
        reward = normalized_pnl * 0.5
    else:
        # Good risk/reward — full credit
        reward = normalized_pnl
elif normalized_pnl <= 0:
    # Loss — no adjustment. The loss itself is punishment enough.
    reward = normalized_pnl
else:
    # Trade never went against you — full credit (rare, but clean)
    reward = normalized_pnl

# Scale for gradient signal
reward *= 100.0
```

### Component 3: Inactivity Cost (per-step, when flat)

```python
if position == 0:  # flat
    candles_flat += 1
else:
    candles_flat = 0

if candles_flat > 24:  # > 6 hours on 15m
    step_reward = -0.0001 * (candles_flat - 24)
    # 24h of inactivity costs ~0.024 reward units
    # A bad trade costs 1-5+ reward units
    # So this never forces trades — just prevents permanent hibernation
```

### Component 4: Circuit Breaker (episode termination)

```python
if account_equity < starting_equity * 0.95:  # -5%
    reward = -5.0
    terminated = True
    # Painful but not so dominant it overwhelms all other signal
```

### What Is NOT In The Reward

| Excluded | Why |
|----------|-----|
| Sharpe ratio bonus | Portfolio metric. Evaluated in validation, not per-step. |
| Win rate bonus | Creates early-close bias — agent takes small wins, avoids learning to hold winners. |
| Equity smoothness | Same as Sharpe — wrong granularity for step-level reward. |
| Unrealized PnL signal | Teaches agent to open positions for green numbers, not for edge. |
| Consecutive-win bonus | Creates correlation-seeking — agent avoids trading after a loss. |
| Trade frequency bonus | Inactivity cost handles the floor. No ceiling needed. |

---

## SYSTEM 4: SAFETY ARCHITECTURE

Everything here sits OUTSIDE the RL agent. The agent cannot see, learn about, or circumvent these systems. They override the agent's decisions unconditionally.

### Layer 1: Hard Position Limits

```python
HARD_LIMITS = {
    "max_position_pct": 1.0,              # of equity (before leverage scaling)
    "max_open_time_candles": 96,           # 24h on 15m → force close at market
    "max_daily_loss": -0.02,              # -2% equity → force flat, lock 4h
    "max_weekly_loss": -0.04,             # -4% equity → force flat, lock 24h
    "min_order_notional": "exchange_min",  # Binance minimum (currently $5 USDⓈ-M)
}

# Emergency stop-loss (per-trade, in safety layer not agent):
# Adapts to volatility but has hard cap
emergency_stop = min(3.0 * atr_at_entry, 0.05 * account_equity)
# Whichever is smaller: 3 ATRs or 5% of equity
# This catches the case where agent holds a loser to the circuit breaker
```

**Implementation**: These are `if` checks that run BEFORE any order hits the exchange. Not part of the reward. Not part of the observation. Code-level walls.

### Layer 2: Trust Ladder (Earned Leverage)

```python
TRUST_LEVELS = {
    0: {
        "leverage": 0.5,
        "requirements": None,  # Starting level
    },
    1: {
        "leverage": 1.0,
        "requirements": {
            "min_trades": 50,
            "min_win_rate": 0.40,
            "max_drawdown": 0.03,
            "min_sharpe": 0.5,
        },
    },
    2: {
        "leverage": 1.5,
        "requirements": {
            "min_trades": 200,
            "min_win_rate": 0.45,
            "max_drawdown": 0.04,
            "min_sharpe": 1.0,
        },
    },
    3: {
        "leverage": 2.0,
        "requirements": {
            "min_trades": 500,
            "min_win_rate": 0.45,
            "max_drawdown": 0.05,
            "min_sharpe": 1.5,
        },
    },
}

# Evaluation: rolling window of recent trades
# Promotion: check after every 25 trades
# Demotion: IMMEDIATE if max DD exceeds current level's threshold
# Visibility: agent NEVER knows its level — leverage silently scales effective_size
```

### Layer 3: Distribution Drift Detector

```python
class DriftDetector:
    def __init__(self, training_feature_stats):
        """
        training_feature_stats: dict of {feature_idx: {"mean": float, "std": float}}
        Computed from the training data distribution.
        """
        self.baseline = training_feature_stats
    
    def check(self, recent_features, window=100):
        """
        Compare recent feature distribution against training baseline.
        Returns: "NORMAL", "HOLD_ONLY", or "HALT"
        """
        drift_scores = []
        for feat_idx in range(recent_features.shape[1]):
            recent = recent_features[-window:, feat_idx]
            mean_drift = abs(np.mean(recent) - self.baseline[feat_idx]["mean"]) \
                         / (self.baseline[feat_idx]["std"] + 1e-8)
            std_drift = abs(np.std(recent) - self.baseline[feat_idx]["std"]) \
                        / (self.baseline[feat_idx]["std"] + 1e-8)
            drift_scores.append(mean_drift + std_drift)
        
        composite = np.mean(drift_scores)
        
        if composite > 3.0:
            return "HALT"        # Unprecedented regime. Stop everything.
        elif composite > 2.0:
            return "HOLD_ONLY"   # Unusual. No new positions.
        else:
            return "NORMAL"      # Within known distribution. Trade freely.

# If HALT persists for 2 hours → alert human, remain halted until manual override
```

### Layer 4: Execution Sanity Checks (Pre-Trade Gate)

```python
def pre_trade_gate(market_state, agent_action, recent_actions):
    """
    Runs BEFORE every order. Can block, modify, or approve.
    """
    # 1. Spread too wide → market is illiquid
    if market_state.bid_ask_spread_pct > 0.005:  # 0.5%
        return {"action": "BLOCK", "reason": "spread"}
    
    # 2. Current candle is extreme outlier → likely wick/glitch
    if abs(market_state.current_return_zscore) > 3.0:
        return {"action": "BLOCK", "reason": "outlier_candle"}
    
    # 3. Agent flipped direction 3+ times in last 2h → it's confused
    flips = count_direction_flips(recent_actions, window_candles=8)
    if flips >= 3:
        return {"action": "COOLDOWN", "duration_candles": 16, "reason": "churn"}
    
    # 4. Extreme funding rate → reduce size
    size_modifier = 1.0
    if abs(market_state.funding_rate) > 0.001:  # 0.1%
        size_modifier = 0.5
    
    return {"action": "EXECUTE", "size_modifier": size_modifier}
```

### Layer 5: Paper-to-Live Pipeline

```
STAGE 1: BACKTEST
├── Data: Historical Binance candles
├── Fills: Simulated with dynamic slippage model
├── Duration: As long as needed
├── Exit criteria: Pass ALL 8 validation checks (see System 5)
└── Gate: Automated — no human needed

STAGE 2: PAPER TRADING
├── Data: Live Binance WebSocket stream
├── Fills: Simulated (no orders placed)
├── Duration: Minimum 1 week
├── Exit criteria:
│   ├── Net P&L > 0
│   ├── Max drawdown < 3%
│   ├── Total trades ≥ 20
│   └── No HALT triggers from drift detector
└── Gate: MANUAL human review required

STAGE 3: SHADOW MODE
├── Data: Live Binance stream
├── Fills: REAL orders at MINIMUM position size
├── Duration: Minimum 1 week
├── Exit criteria:
│   ├── Net P&L > 0
│   ├── Max drawdown < 2%
│   ├── Total trades ≥ 20
│   ├── Simulated vs actual fill comparison (slippage model accuracy)
│   └── No API errors or missed fills
└── Gate: MANUAL human review required

STAGE 4: LIVE
├── Trust Ladder activates (starts at Level 0: 0.5x)
├── All safety layers active
├── Daily performance report auto-generated
└── Demotion back to Shadow if weekly loss > 4%
```

---

## SYSTEM 5: TRAINING PIPELINE

### Data Requirements

```python
DATA_CONFIG = {
    "source": "Binance REST API",        # existing fetch_data.py handles this
    "pairs": ["BTCUSDT", "ETHUSDT"],
    "base_timeframe": "15m",              # step clock
    "derived_timeframes": ["1h", "4h"],   # resampled from 15m
    "period": "2023-01-01 to present",    # 2.5+ years
    "additional_data": {
        "funding_rate": {
            "endpoint": "GET /fapi/v1/fundingRate",
            "interval": "8h native, interpolated to 15m alignment",
        },
    },
    "storage": "Parquet",                 # existing format
    "approx_size": "~175,000 candles per pair for 2.5 years of 15m data",
}
```

**Migration note**: Your existing `fetch_data.py` already handles OHLCV download. You need to ADD a funding rate fetcher that pulls from Binance's funding rate endpoint and aligns timestamps to your 15m grid. Funding rate is reported every 8h — use forward-fill to align to 15m candles (the rate doesn't change between settlements).

### Feature Engineering Migration

```python
# In features.py — KEEP these from existing system:
# - RSI calculation
# - MACD histogram calculation  
# - Candle body/wick ratio calculation
# - Time-of-day sin/cos encoding

# MODIFY:
# - Change EMA from 8/21/55 alignment score → EMA(9)-EMA(21) spread, z-scored
# - Change RSI normalization from [0,100] → [-1,1] via (rsi-50)/50
# - Change MACD normalization to z-scored
# - Apply all indicators on BOTH 15m and 1h timeframes

# ADD NEW:
# - Z-scored OHLCV (100-period rolling) for 15m, 1h, 4h
# - ATR(14)/ATR(50) ratio on 1h
# - Relative volume: volume / SMA(volume, 20) on 15m
# - Funding rate normalization
# - Cross-pair momentum (other pair's 15m log-return, z-scored)

# REMOVE (redundant):
# - Bollinger %B (redundant with z-scored close)
# - ATR% (redundant with ATR ratio)
# - EMA alignment score (replaced by EMA spread)
# - Volume z-score from original (now in z-scored OHLCV block)
```

### Walk-Forward Validation

```python
WALK_FORWARD_CONFIG = {
    "type": "expanding_window",
    "train_start": "2023-01-01",              # always train from beginning
    "initial_train_months": 6,                 # first window: 6 months train
    "test_window_months": 2,                   # each test window: 2 months
    "step_months": 2,                          # slide test forward by 2 months
    
    # This produces windows like:
    # W1: Train Jan-Jun 2023    → Test Jul-Aug 2023
    # W2: Train Jan-Aug 2023    → Test Sep-Oct 2023
    # W3: Train Jan-Oct 2023    → Test Nov-Dec 2023
    # W4: Train Jan-Dec 2023    → Test Jan-Feb 2024
    # W5: Train Jan-Feb 2024    → Test Mar-Apr 2024
    # ... continuing to present
    
    # Agent must pass validation across MAJORITY of these windows
}
```

### PPO Hyperparameters

```python
PPO_CONFIG = {
    # Network architecture
    "obs_dim": 38,
    "hidden_sizes": [256, 256],
    "activation": "tanh",
    
    # Dual action heads
    "direction_head": "categorical",           # Categorical(4)
    "sizing_head": "beta",                     # Beta(alpha, beta)
    "sizing_min_threshold": 0.1,               # below this → treated as Hold
    
    # PPO core
    "rollout_steps": 4096,                     # steps per collection phase
    "n_epochs": 10,                            # PPO update epochs per rollout
    "mini_batch_size": 256,
    "clip_epsilon": 0.2,
    "gamma": 0.995,                            # high — trades can last hours/days
    "gae_lambda": 0.95,
    
    # Learning rate
    "lr_start": 3e-4,
    "lr_end": 0.0,
    "lr_schedule": "linear_decay",
    
    # Entropy (exploration → exploitation)
    "entropy_coef_start": 0.01,
    "entropy_coef_end": 0.001,
    "entropy_decay_fraction": 0.7,             # decay over 70% of training
    
    # Critic
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    
    # Training scale
    "total_timesteps": 2_000_000,              # per walk-forward window
    "eval_frequency": 10_000,                  # evaluate every N steps
    "checkpoint_frequency": 50_000,
}
```

### The "Is This Agent Real?" Validation Suite

ALL 8 checks must pass before an agent exits training. No exceptions.

```python
VALIDATION_CHECKS = {
    "1_consistent_across_time": {
        "metric": "profitable walk-forward windows / total windows",
        "threshold": ">= 0.70",
        "rationale": "70%+ of test windows must be profitable. Single lucky window is not enough.",
    },
    "2_sufficient_samples": {
        "metric": "total trades per test window",
        "threshold": ">= 100",
        "rationale": "Statistical significance. 10 lucky trades prove nothing.",
    },
    "3_capital_preservation": {
        "metric": "max drawdown in ANY single test window",
        "threshold": "< 5%",
        "rationale": "Even the worst window must be survivable.",
    },
    "4_risk_adjusted_quality": {
        "metric": "Sharpe ratio across all test periods combined",
        "threshold": ">= 0.5",
        "rationale": "Returns must be meaningful relative to risk taken.",
    },
    "5_not_degenerate": {
        "metric": "win rate",
        "threshold": "between 35% and 65%",
        "rationale": "Below 35% = losing too much. Above 65% = likely scalping tiny wins with huge tail risk.",
    },
    "6_positive_expectancy": {
        "metric": "average win / average loss",
        "threshold": ">= 1.2",
        "rationale": "Winning trades must be meaningfully larger than losing trades.",
    },
    "7_no_single_trade_dependency": {
        "metric": "largest single trade P&L / total profit",
        "threshold": "< 20%",
        "rationale": "If one trade made 20%+ of all profit, the agent got lucky, not good.",
    },
    "8_execution_robust": {
        "metric": "still profitable when slippage is DOUBLED in simulation",
        "threshold": "net positive P&L",
        "rationale": "If doubling execution costs kills profitability, the edge was never real. It was thinner than the uncertainty in your fill model.",
    },
}
```

---

## SYSTEM 6: FEE & SLIPPAGE MODEL

### Fees

```python
# Binance USDⓈ-M Futures — default tier (0-250K monthly volume)
MAKER_FEE = 0.0002    # 0.02%
TAKER_FEE = 0.0005    # 0.05%

# Agent always pays TAKER (market orders — RL doesn't do limit orders)
# Both entry AND exit incur fee
# Fee is deducted from equity, not from position size
```

### Dynamic Slippage Model

```python
def calc_slippage(volatility_regime, relative_volume, position_size_pct):
    """
    Realistic slippage that adapts to market conditions.
    
    Replaces fixed 0.05% from original system.
    
    Args:
        volatility_regime: ATR(14)/ATR(50) on 1h (from observations)
        relative_volume: volume/SMA(vol,20) on 15m (from observations)
        position_size_pct: raw_size from agent (0 to 1)
    
    Returns:
        slippage as a fraction (e.g., 0.0003 = 0.03%)
    """
    base_slippage = 0.0003  # 0.03% baseline — BTC/ETH are liquid pairs
    
    # High volatility = worse fills
    vol_mult = max(1.0, volatility_regime)
    
    # Low liquidity = worse fills
    liq_mult = max(1.0, 1.0 / max(relative_volume, 0.1))
    
    # Larger position = more market impact
    size_mult = 1.0 + position_size_pct * 0.5
    
    slippage = base_slippage * vol_mult * liq_mult * size_mult
    
    # Hard cap — if slippage would exceed this, the pre-trade gate
    # should have blocked the trade anyway
    return min(slippage, 0.01)  # max 1%

# For stress testing (validation check #8):
# stress_slippage = calc_slippage(...) * 2.0
```

---

## PROJECT FILE STRUCTURE (Evolution of Existing)

```
apex-ai-trading/
│
├── terminal/                          # EXISTING — React trading UI
│   └── (keep as-is, connect later)
│
├── rl/                                # EVOLVED — Python RL engine
│   │
│   ├── config/
│   │   └── default.yaml              # NEW — all hyperparams in one place
│   │
│   ├── data/
│   │   ├── fetch_data.py             # EXISTING ✅ — add funding rate endpoint
│   │   ├── features.py               # EXISTING 🔄 — merge to 38 features
│   │   └── validator.py              # NEW — data quality checks
│   │
│   ├── env/
│   │   ├── trading_env.py            # REWRITE 🔄 — hybrid actions, new reward, no TP/SL
│   │   ├── fee_model.py              # NEW — dynamic slippage
│   │   └── reward.py                 # NEW — separated for easy tuning
│   │
│   ├── agent/
│   │   ├── ppo_hybrid.py             # NEW (replaces dqn.py) — CleanRL PPO + dual heads
│   │   ├── networks.py               # NEW — actor-critic with shared backbone
│   │   └── distributions.py          # NEW — Beta distribution utilities
│   │
│   ├── safety/
│   │   ├── position_limits.py        # NEW — hard limits, emergency stop
│   │   ├── trust_ladder.py           # NEW — earned leverage
│   │   ├── drift_detector.py         # NEW — distribution monitoring
│   │   ├── execution_gate.py         # NEW — pre-trade sanity checks
│   │   └── pipeline.py              # NEW — paper → shadow → live stages
│   │
│   ├── training/
│   │   ├── train.py                  # REWRITE 🔄 — walk-forward PPO training
│   │   ├── walk_forward.py           # NEW — rolling window logic
│   │   ├── validation.py             # NEW — 8-check suite
│   │   └── logger.py                 # NEW — metrics logging
│   │
│   ├── live/
│   │   ├── executor.py               # NEW — Binance order execution
│   │   ├── data_stream.py            # NEW — real-time candle + funding feed
│   │   ├── orchestrator.py           # NEW — main live loop
│   │   └── monitor.py                # NEW — daily reports, alerts
│   │
│   ├── evaluate.py                   # EXISTING 🔄 — extend with validation suite
│   ├── models/                       # EXISTING — trained model checkpoints
│   └── requirements.txt              # UPDATE — add cleanrl dependencies
│
└── README.md
```

---

## BUILD ORDER FOR CLAUDE CODE

**Do NOT build everything at once. Build phase by phase. Test each phase before moving on.**

```
PHASE 1: DATA EVOLUTION
  Files: data/fetch_data.py (modify), data/features.py (modify), data/validator.py (new)
  Tasks:
    1. Add Binance funding rate fetcher to fetch_data.py
    2. Rewrite features.py for 38-feature observation space
    3. Add multi-timeframe resampling (15m → 1h, 4h)
    4. Add cross-pair feature computation (requires both BTC + ETH data loaded)
    5. Build validator.py — check for gaps, NaN, timestamp alignment
  TEST:
    - Load BTC + ETH data
    - Compute all 38 features
    - Verify: no NaN, all bounded [-5, 5], funding rate aligned, cross-pair signal present
    - Verify: 1h/4h candles only update on close (no lookahead)

PHASE 2: ENVIRONMENT EVOLUTION
  Files: env/trading_env.py (rewrite), env/fee_model.py (new), env/reward.py (new)
  Tasks:
    1. Build fee_model.py with dynamic slippage
    2. Build reward.py with 4 components (separated file for easy tuning)
    3. Rewrite trading_env.py:
       - Gymnasium env with 38-dim obs space
       - Hybrid action space (dict: {"direction": Discrete(4), "sizing": Box(0,1)})
       - Integrate fee model and reward function
       - Track MAE per trade for risk adjustment
       - Emergency stop-loss in safety layer, NOT in env reward
  TEST:
    - Random agent: verify fees deducted correctly on each trade
    - Verify reward = 0 on non-close steps
    - Verify reward fires correctly on close (with risk adjustment)
    - Verify circuit breaker triggers at -5%
    - Verify emergency stop triggers at min(3*ATR, 5% equity)
    - Verify position flip (long→short) charges both close and open fees
    - Verify sizing < 0.1 treated as Hold
    - Print 100-step trace and manually audit

PHASE 3: AGENT (New)
  Files: agent/distributions.py, agent/networks.py, agent/ppo_hybrid.py
  Tasks:
    1. Build Beta distribution utilities
    2. Build dual-head actor-critic network (shared 256-256 backbone)
    3. Build PPO training loop based on CleanRL's ppo.py
       - Modify for hybrid action space
       - log_prob = log_prob_direction + log_prob_sizing
       - entropy = entropy_direction + entropy_sizing
  TEST:
    - Single training run on 1 month of data
    - Verify loss decreases over time
    - Verify agent takes varied actions (not stuck on Hold or one direction)
    - Verify sizing output has variance (not always 0.5)
    - Check gradient norms are stable (not exploding or vanishing)

PHASE 4: TRAINING PIPELINE
  Files: training/walk_forward.py, training/validation.py, training/train.py, training/logger.py
  Tasks:
    1. Build walk-forward window generator
    2. Build validation suite (8 checks)
    3. Build full training pipeline with logging
    4. Integrate: for each window → train → validate → log → next window
  TEST:
    - Run complete walk-forward training on BTC data
    - Review validation results — does the agent pass? Which checks fail?
    - Compare to existing DQN results — is the new system better or worse?
    - Run slippage stress test (check #8)

PHASE 5: SAFETY LAYERS
  Files: safety/*.py (all new)
  Tasks:
    1. Build each safety module independently
    2. Unit test each with adversarial scenarios:
       - position_limits: feed it 50 losing trades, verify lockout triggers
       - trust_ladder: verify promotion/demotion logic
       - drift_detector: feed it OOD data, verify HALT triggers
       - execution_gate: test each check independently
       - pipeline: verify stage transitions and human gate
  TEST:
    - Wire all safety layers into environment
    - Run agent with safety ON vs OFF — verify safety prevents worst outcomes
    - Verify agent doesn't know about safety (observations unchanged)

PHASE 6: LIVE INFRASTRUCTURE (only after Phase 4 passes validation)
  Files: live/*.py (all new)
  Tasks:
    1. Paper trading: live WebSocket → agent → simulated fills → log
    2. Shadow mode: live stream → agent → real minimum-size orders → compare
    3. Full live: orchestrator with all safety layers active
  TEST:
    - 1 week paper trading — compare to backtest expectations
    - Verify slippage model accuracy against real fills in shadow mode
    - MANUAL REVIEW before each stage promotion

PHASE 7: TERMINAL INTEGRATION (final phase)
  Tasks:
    1. API endpoint serving agent signals to React terminal
    2. Replace rule-based scorer with neural network output
    3. Display confidence (sizing head output) in UI
    4. Show trust ladder level and safety status in dashboard
```

---

## KEY PRINCIPLES (Pin This to Every Claude Code Session)

1. **The agent reads the market, not its own P&L.** No balance, no unrealized PnL, no drawdown in observations.
2. **Reward the trade, evaluate the portfolio.** Per-trade reward. Sharpe/win-rate are validation metrics, not reward components.
3. **The thing that takes risk must not manage risk.** Agent ≠ safety layer. Architecturally separated. Agent can't see or learn about safety systems.
4. **Every bonus in the reward is a potential exploit.** 4 components, no more. If you're tempted to add a 5th, you're probably papering over a flaw.
5. **If doubling slippage kills profitability, the edge was never real.** Validation check #8 is the ultimate bullshit detector.
6. **The agent earns leverage, it doesn't choose leverage.** Trust ladder is invisible and non-negotiable.
7. **No stage promotion without human review.** Paper → shadow → live requires your explicit approval.
8. **Build and test in phases.** Never build Phase N+1 until Phase N passes its tests. The bugs compound otherwise.
