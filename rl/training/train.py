"""
train.py
────────
Main training entry point.

Modes:
  single   — train one model on the full dataset (fastest, for iteration)
  walkforward — full walk-forward pipeline with 8-check validation

Usage:
    # Quick single-run training (no walk-forward):
    python training/train.py --coin BTC --steps 200000

    # Full walk-forward validation:
    python training/train.py --coin BTC --mode walkforward --steps 200000

    # Multi-coin (sequential):
    python training/train.py --coin BTC --coin ETH --coin SOL --steps 200000

    # Stress-test sizing:
    python training/train.py --coin BTC --stress

Run from rl/ directory.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_RL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_RL_DIR))
sys.path.insert(0, str(_RL_DIR.parent))


def run_single(
    coin:           str,
    steps:          int,
    stress:         bool  = False,
    episode_length: int   = 2000,
    initial_equity: float = 10_000.0,
    coin2:          str   = None,
    eval_freq:      int   = 20_000,
    save_dir:       str   = None,
):
    """
    Train one model on the full dataset. Splits last 20% as eval set.
    """
    from features import add_features, N_FEATURES
    from env.trading_env import TradingEnv
    from dqn import DQNAgent

    data_dir = _RL_DIR / "data"
    ohlcv    = data_dir / f"{coin}_15m.parquet"
    if not ohlcv.exists():
        print(f"  Data not found: {ohlcv}", flush=True)
        print(f"  Run: python fetch_data.py --coin {coin}", flush=True)
        return None

    df_raw = pd.read_parquet(ohlcv)

    funding_series = None
    fp = data_dir / f"{coin}_funding.parquet"
    if fp.exists():
        from fetch_data import align_funding_to_15m
        funding_series = align_funding_to_15m(df_raw, pd.read_parquet(fp))

    other_close = None
    if coin2:
        p2 = data_dir / f"{coin2}_15m.parquet"
        if p2.exists():
            df2 = pd.read_parquet(p2)
            other_close = df2.set_index("time")["close"]

    print(f"\n  Computing features for {coin}...", flush=True)
    df = add_features(df_raw, funding_series, other_close)
    print(f"  {len(df):,} candles  [{df['time'].iloc[0]} → {df['time'].iloc[-1]}]",
          flush=True)

    split    = int(len(df) * 0.80)
    train_df = df.iloc[:split].reset_index(drop=True)
    eval_df  = df.iloc[split:].reset_index(drop=True)

    print(f"  Train: {len(train_df):,}  Eval: {len(eval_df):,}", flush=True)

    train_env = TradingEnv(train_df, initial_equity=initial_equity,
                           episode_length=episode_length, stress_fees=stress)
    eval_env  = TradingEnv(eval_df,  initial_equity=initial_equity,
                           episode_length=episode_length) if len(eval_df) > 500 else None

    obs_dim = N_FEATURES + 2
    agent   = DQNAgent(
        obs_dim         = obs_dim,
        n_actions       = 4,
        hidden          = [256, 256],
        lr              = 1e-4,
        gamma           = 0.99,
        buffer_size     = 100_000,
        batch_size      = 256,
        learning_starts = 5_000,
        train_freq      = 4,
        target_update   = 1_000,
        eps_start       = 1.0,
        eps_end         = 0.02,
        eps_decay_steps = int(steps * 0.6),
    )

    if save_dir is None:
        save_dir = str(_RL_DIR / "models" / f"{coin}_single")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}", flush=True)
    print(f"  Training {coin}  |  {steps:,} steps  |  "
          f"stress={'ON' if stress else 'off'}", flush=True)
    print(f"{'═'*60}\n", flush=True)

    t0 = time.time()
    agent.learn(
        train_env,
        total_timesteps  = steps,
        eval_env         = eval_env,
        eval_freq        = eval_freq,
        n_eval_episodes  = 5,
        save_path        = save_dir,
        verbose          = 1,
    )
    elapsed = time.time() - t0

    # Final evaluation
    if eval_env is not None:
        mean_r = agent._evaluate(eval_env, 5)
        final_eq_approx = initial_equity * (1 + mean_r / 100)
        print(f"\n  Final eval reward: {mean_r:.4f}", flush=True)

    # Save final model
    final_path = str(Path(save_dir) / "final_model.pkl")
    agent.save(final_path)
    print(f"  Saved: {final_path}", flush=True)
    print(f"  Elapsed: {int(elapsed//60):02d}:{int(elapsed%60):02d}", flush=True)

    return agent


def run_walkforward(coin: str, steps: int, stress: bool = False,
                    coin2: str = None):
    from training.walk_forward import WalkForwardPipeline, WFConfig

    cfg    = WFConfig(
        coin             = coin,
        coin2            = coin2,
        total_steps      = steps,
        run_stress_test  = stress,
    )
    result = WalkForwardPipeline(cfg).run()
    return result


def main():
    parser = argparse.ArgumentParser(description="APEX RL Trader — training")
    parser.add_argument("--coin",   action="append", default=None,
                        help="Coin(s) to train (BTC, ETH, SOL). Repeatable.")
    parser.add_argument("--coin2",  default=None,
                        help="Cross-pair coin for momentum feature.")
    parser.add_argument("--steps",  type=int, default=200_000)
    parser.add_argument("--mode",   choices=["single", "walkforward"],
                        default="single")
    parser.add_argument("--stress", action="store_true",
                        help="Use 2× slippage during training.")
    parser.add_argument("--episode-length", type=int, default=2000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    args = parser.parse_args()

    coins = args.coin or ["BTC"]

    for coin in coins:
        print(f"\n{'#'*65}", flush=True)
        print(f"  APEX — {coin}  ({args.mode})", flush=True)
        print(f"{'#'*65}", flush=True)

        if args.mode == "single":
            run_single(
                coin           = coin,
                steps          = args.steps,
                stress         = args.stress,
                episode_length = args.episode_length,
                coin2          = args.coin2,
                eval_freq      = args.eval_freq,
            )
        else:
            run_walkforward(
                coin   = coin,
                steps  = args.steps,
                stress = args.stress,
                coin2  = args.coin2,
            )


if __name__ == "__main__":
    main()
