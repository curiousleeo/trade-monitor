"""
train.py
────────
Train a DQN agent on historical candle data.

Usage:
    python train.py                              # BTC 15m, 500k steps
    python train.py --coin ETH --tf 1h
    python train.py --coin BTC --steps 1000000

Output:
    models/BTC_15m_dqn/   — saved model + replay buffer
    logs/BTC_15m/         — tensorboard logs (run: tensorboard --logdir logs/)
"""

import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from rich.console import Console

from features import add_features
from environment import TradingEnv

console = Console()

DATA_DIR   = Path(__file__).parent / "data"
MODEL_DIR  = Path(__file__).parent / "models"
LOG_DIR    = Path(__file__).parent / "logs"


def make_env(df: pd.DataFrame, episode_len: int):
    def _init():
        return Monitor(TradingEnv(df, episode_len=episode_len))
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin",        default="BTC")
    parser.add_argument("--tf",          default="15m")
    parser.add_argument("--steps",       type=int, default=500_000)
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
    train_env = DummyVecEnv([make_env(df_train, args.episode_len)])
    eval_env  = DummyVecEnv([make_env(df_eval,  args.episode_len)])

    # ── Model ─────────────────────────────────────────────────────────────────
    model_path = MODEL_DIR / tag
    model_path.mkdir(parents=True, exist_ok=True)
    log_path   = LOG_DIR / tag
    log_path.mkdir(parents=True, exist_ok=True)

    model = DQN(
        policy             = "MlpPolicy",
        env                = train_env,
        learning_rate      = 1e-4,
        buffer_size        = 100_000,
        learning_starts    = 5_000,
        batch_size         = 256,
        gamma              = 0.99,
        train_freq         = 4,
        gradient_steps     = 1,
        target_update_interval = 1_000,
        exploration_fraction   = 0.20,   # explore for first 20% of steps
        exploration_final_eps  = 0.02,
        optimize_memory_usage  = False,
        verbose            = 1,
        tensorboard_log    = str(log_path),
        policy_kwargs      = dict(net_arch=[256, 256]),   # 2-layer MLP
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path = str(model_path),
            log_path             = str(log_path),
            eval_freq            = 10_000,
            n_eval_episodes      = 5,
            deterministic        = True,
            verbose              = 1,
        ),
        CheckpointCallback(
            save_freq  = 50_000,
            save_path  = str(model_path / "checkpoints"),
            name_prefix= "dqn",
        ),
    ]

    console.print(f"[bold]Training DQN for {args.steps:,} steps[/bold]")
    console.print(f"Model will be saved to [cyan]{model_path}[/cyan]")
    console.print("To monitor: [yellow]tensorboard --logdir logs/[/yellow]\n")

    model.learn(
        total_timesteps = args.steps,
        callback        = callbacks,
        progress_bar    = True,
    )

    model.save(str(model_path / "final"))
    console.print(f"\n[bold green]✓ Training complete. Model saved to {model_path}/final[/bold green]")
    console.print("Next step: [cyan]python evaluate.py[/cyan]")


if __name__ == "__main__":
    main()
