"""
evaluate.py
───────────
Run the trained DQN agent on the held-out test set and print
a full performance report: Sharpe, max drawdown, win rate, etc.

Usage:
    python evaluate.py                        # BTC 15m best_model
    python evaluate.py --coin ETH --tf 1h
    python evaluate.py --model models/BTC_15m/final.pkl
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from features import add_features
from environment import TradingEnv, INITIAL_BALANCE
from dqn import DQNAgent

console = Console()

DATA_DIR  = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"


def run_episode(env: TradingEnv, agent: DQNAgent) -> dict:
    """Run one full deterministic episode and collect stats."""
    obs, _  = env.reset()
    done     = False
    balances = [INITIAL_BALANCE]
    trades   = []
    in_trade_since = None

    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        balances.append(info["balance"])

        # track trade entries/exits via position changes
        if env.position != 0 and in_trade_since is None:
            in_trade_since = env.cursor
        elif env.position == 0 and in_trade_since is not None:
            entry_bal = balances[in_trade_since]
            exit_bal  = info["balance"]
            trades.append({
                "pnl":  exit_bal - entry_bal,
                "win":  exit_bal > entry_bal,
            })
            in_trade_since = None

    return {"balances": balances, "trades": trades}


def sharpe(balances: list, risk_free: float = 0.0) -> float:
    arr     = np.array(balances)
    returns = np.diff(arr) / arr[:-1]
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() - risk_free) / returns.std() * np.sqrt(252 * 96))
    # 252 trading days × 96 × 15min candles/day ≈ annualised


def max_drawdown(balances: list) -> float:
    arr     = np.array(balances)
    peak    = np.maximum.accumulate(arr)
    dd      = (arr - peak) / peak
    return float(dd.min())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin",  default="BTC")
    parser.add_argument("--tf",    default="15m")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    tag        = f"{args.coin}_{args.tf}"
    data_path  = DATA_DIR / f"{tag}.parquet"
    model_path = args.model or str(MODEL_DIR / tag / "best_model.pkl")

    console.print(f"[bold cyan]Loading model: {model_path}[/bold cyan]")
    agent = DQNAgent.load(model_path)

    console.print(f"[bold cyan]Loading data: {data_path}[/bold cyan]")
    df_raw = pd.read_parquet(data_path)
    df     = add_features(df_raw)

    # Use the held-out 20% test set
    split   = int(len(df) * 0.8)
    df_test = df.iloc[split:].reset_index(drop=True)
    console.print(f"Test set: {len(df_test):,} candles")

    env    = TradingEnv(df_test, episode_len=len(df_test))
    result = run_episode(env, agent)

    bals   = result["balances"]
    trades = result["trades"]
    wins   = [t for t in trades if t["win"]]
    final  = bals[-1]

    # ── Buy-and-hold baseline ─────────────────────────────────────────────────
    bh_start  = float(df_test.iloc[0]["close"])
    bh_end    = float(df_test.iloc[-1]["close"])
    bh_return = (bh_end - bh_start) / bh_start * 100

    # ── Print report ─────────────────────────────────────────────────────────
    table = Table(title=f"APEX RL — {tag} Test Results", show_header=False, min_width=44)
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="bold white")

    table.add_row("Final balance",      f"${final:,.2f}")
    table.add_row("Return",             f"{(final/INITIAL_BALANCE - 1)*100:+.1f}%")
    table.add_row("Buy & hold",         f"{bh_return:+.1f}%")
    table.add_row("Sharpe ratio",       f"{sharpe(bals):.2f}")
    table.add_row("Max drawdown",       f"{max_drawdown(bals)*100:.1f}%")
    table.add_row("Total trades",       str(len(trades)))
    table.add_row("Win rate",           f"{len(wins)/max(len(trades),1)*100:.0f}%"
                                        f"  ({len(wins)}/{len(trades)})")
    table.add_row("Avg win",            f"${np.mean([t['pnl'] for t in wins]):.2f}"
                                        if wins else "—")
    table.add_row("Avg loss",           f"${np.mean([t['pnl'] for t in trades if not t['win']]):.2f}"
                                        if [t for t in trades if not t["win"]] else "—")
    console.print(table)

    # ── Equity curve plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    axes[0].plot(bals, color="#26a69a", linewidth=1)
    axes[0].axhline(INITIAL_BALANCE, color="#555", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"APEX RL Equity Curve — {tag}")
    axes[0].set_ylabel("Balance ($)")

    arr  = np.array(bals)
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / peak * 100
    axes[1].fill_between(range(len(dd)), dd, 0, color="#ef5350", alpha=0.6)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Steps")

    out = Path(__file__).parent / f"equity_{tag}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    console.print(f"[green]Equity curve saved → {out}[/green]")


if __name__ == "__main__":
    main()
