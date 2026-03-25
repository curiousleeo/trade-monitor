"""
train.py
────────
Train a DQN agent on historical candle data.
Uses the pure-numpy DQN in dqn.py (no PyTorch required).

Usage:
    python train.py                              # BTC 15m, 200k steps
    python train.py --coin ETH --tf 1h
    python train.py --coin BTC --steps 500000

Output:
    models/BTC_15m/best_model.pkl   — best model by eval reward
    models/BTC_15m/final.pkl        — model at end of training
"""

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console

from features import add_features
from environment import TradingEnv
from dqn import DQNAgent

console = Console()

DATA_DIR  = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin",        default="BTC")
    parser.add_argument("--tf",          default="15m")
    parser.add_argument("--steps",       type=int, default=200_000)
    parser.add_argument("--episode-len", type=int, default=2_000)
    args = parser.parse_args()

    tag       = f"{args.coin}_{args.tf}"
    data_path = DATA_DIR / f"{tag}.parquet"

    if not data_path.exists():
        console.print(f"[red]Data not found: {data_path}[/red]")
        console.print("Run:  python fetch_data.py --coin BTC --tf 15m")
        return

    console.print(f"[bold cyan]Loading {data_path}[/bold cyan]")
    df_raw = pd.read_parquet(data_path)
    df     = add_features(df_raw)
    console.print(f"[green]✓ {len(df):,} candles after feature engineering[/green]")

    # ── Train / eval split (80 / 20) ─────────────────────────────────────────
    split    = int(len(df) * 0.8)
    df_train = df.iloc[:split].reset_index(drop=True)
    df_eval  = df.iloc[split:].reset_index(drop=True)
    console.print(f"Train: {len(df_train):,}  |  Eval: {len(df_eval):,}")

    # ── Environments ─────────────────────────────────────────────────────────
    train_env = TradingEnv(df_train, episode_len=args.episode_len)
    eval_env  = TradingEnv(df_eval,  episode_len=args.episode_len)

    obs_dim   = train_env.observation_space.shape[0]
    n_actions = train_env.action_space.n

    # ── Agent ─────────────────────────────────────────────────────────────────
    model_path = MODEL_DIR / tag
    model_path.mkdir(parents=True, exist_ok=True)

    agent = DQNAgent(
        obs_dim         = obs_dim,
        n_actions       = n_actions,
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
        eps_decay_steps = int(args.steps * 0.20),
    )

    console.print(f"[bold]Training DQN for {args.steps:,} steps[/bold]")
    console.print(f"obs_dim={obs_dim}  n_actions={n_actions}  hidden=[256,256]")
    console.print(f"Model will be saved to [cyan]{model_path}[/cyan]\n")

    agent.learn(
        env             = train_env,
        total_timesteps = args.steps,
        eval_env        = eval_env,
        eval_freq       = 10_000,
        n_eval_episodes = 5,
        save_path       = str(model_path),
        verbose         = 1,
    )

    agent.save(str(model_path / "final.pkl"))
    console.print(f"\n[bold green]✓ Training complete. Saved to {model_path}/final.pkl[/bold green]")
    console.print("Next step: [cyan]python evaluate.py[/cyan]")


if __name__ == "__main__":
    main()
