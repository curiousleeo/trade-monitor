# APEX RL Framework

DQN-based reinforcement learning agent for crypto trading.
Trains on Binance historical data, evaluates on a held-out test set.

## Quick Start

```bash
cd rl/

# 1. Create a virtual environment and install deps
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Download historical data (BTC 15m since 2021, ~2 min)
python fetch_data.py --coin BTC --tf 15m --since 2021-01-01

# 3. Train the agent (500k steps ≈ 15–30 min on CPU)
python train.py --coin BTC --tf 15m --steps 500000

# 4. Evaluate on held-out test set
python evaluate.py --coin BTC --tf 15m
```

## Files

| File | Purpose |
|---|---|
| `fetch_data.py` | Pull OHLCV from Binance → Parquet |
| `features.py`   | Compute 15 normalised technical features |
| `environment.py`| Gymnasium env: actions, rewards, TP/SL logic |
| `train.py`      | DQN training with EvalCallback + checkpoints |
| `evaluate.py`   | Backtest report: Sharpe, drawdown, win rate, equity plot |

## Architecture

```
State (18 values):
  15 market features (EMAs, RSI, MACD, BB, ATR, volume, candle, time)
  + position (-1/0/1)
  + unrealised P&L %
  + steps in current trade

Actions: HOLD | LONG | SHORT | CLOSE

Reward: log-return on trade close, small hold penalty

Policy: MLP 256→256→4 (DQN with experience replay)
```

## Monitoring Training

```bash
tensorboard --logdir logs/
# open http://localhost:6006
```

## Tips

- **More data = better.** Start with `--since 2020-01-01` for more training samples.
- **Try multiple coins.** Fetch ETH and SOL too, train separate agents.
- **Tune SL/TP multipliers** in `environment.py` to match your risk profile.
- **Longer episodes** (`--episode-len 5000`) give the agent more context per rollout.
