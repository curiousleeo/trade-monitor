"""
walk_forward.py
───────────────
Expanding-window walk-forward validation.

Window scheme (blueprint v2.1):
  - Train:  always from DATA_START (2023-01-01) to window_end
  - Test:   next 2 months after window_end (non-overlapping)
  - Slide:  advance by 2 months each iteration
  - Minimum train window: 6 months of data

Example with 2-month slide:
  Window 1:  train [2023-01 → 2023-06]  test [2023-07 → 2023-08]
  Window 2:  train [2023-01 → 2023-08]  test [2023-09 → 2023-10]
  Window 3:  train [2023-01 → 2023-10]  test [2023-11 → 2023-12]
  ...

Usage:
    from training.walk_forward import WalkForwardPipeline, WFConfig

    cfg = WFConfig(coin="BTC", total_steps=200_000)
    results = WalkForwardPipeline(cfg).run()
"""

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Root of rl/ directory
_RL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_RL_DIR))
sys.path.insert(0, str(_RL_DIR.parent))

from features_1h import add_features_1h, N_FEATURES
from env.trading_env import TradingEnv
from agent.ppo import PPOAgent
from training.validation import WindowResult, validate_windows
from predictor.win_predictor import WinPredictor


@dataclass
class WFConfig:
    """Walk-forward configuration."""
    coin:            str   = "BTC"
    coin2:           str   = None    # cross-pair (e.g. ETH for BTC agent)

    # Data
    data_start:      str   = "2021-01-01"   # training always starts here
    test_window_months: int = 1             # test window size in months
    min_train_months:   int = 6             # minimum training window

    # Training
    total_steps:     int   = 200_000        # steps per window
    eval_freq:       int   = 20_000         # evaluation frequency
    n_eval_episodes: int   = 3

    # Agent hyperparams
    hidden:          List[int] = field(default_factory=lambda: [512, 256])
    lr:              float = 3e-4
    gamma:           float = 0.99
    batch_size:      int   = 64
    seq_len:         int   = 24    # frame stacking: last 24 hours of context
    eps_decay_steps: int   = None

    # Episode config
    episode_length:  int   = 1500   # candles per training episode (~2 months at 1h)
    initial_equity:  float = 10_000.0

    # Output
    save_dir:        str   = None   # None → rl/models/{coin}_wf/
    run_stress_test: bool  = True   # also evaluate with 2× slippage

    def __post_init__(self):
        if self.save_dir is None:
            self.save_dir = str(_RL_DIR / "models" / f"{self.coin}_wf")


class WalkForwardPipeline:
    """
    Runs expanding-window walk-forward validation.
    Trains a fresh agent on each expanding window, evaluates on the
    held-out test window, and accumulates WindowResult objects.
    """

    def __init__(self, cfg: WFConfig):
        self.cfg = cfg
        self._data_dir = _RL_DIR / "data"

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_features(self) -> pd.DataFrame:
        """Load and featurise the full dataset once."""
        ohlcv_path = self._data_dir / f"{self.cfg.coin}_15m.parquet"
        if not ohlcv_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {ohlcv_path}\n"
                f"Run: python fetch_data.py --coin {self.cfg.coin}"
            )

        df_raw = pd.read_parquet(ohlcv_path)

        # Optional funding rate
        funding_series = None
        funding_path   = self._data_dir / f"{self.cfg.coin}_funding.parquet"
        if funding_path.exists():
            from fetch_data import align_funding_to_15m
            import pandas as _pd
            df_f           = _pd.read_parquet(funding_path)
            funding_series = align_funding_to_15m(df_raw, df_f)

        # Optional cross-pair (1h close for cross-pair momentum)
        other_close = None
        if self.cfg.coin2:
            path2 = self._data_dir / f"{self.cfg.coin2}_15m.parquet"
            if path2.exists():
                from features_1h import _resample_to_1h as _r1h
                df2         = pd.read_parquet(path2)
                df2_1h      = _r1h(df2)
                other_close = df2_1h.set_index("time")["close"]

        print(f"  Computing features for {self.cfg.coin}...", flush=True)
        df_feat = add_features_1h(df_raw, funding_series, other_close)
        print(f"  {len(df_feat):,} candles after warmup  "
              f"[{df_feat['time'].iloc[0]} → {df_feat['time'].iloc[-1]}]",
              flush=True)
        return df_feat

    # ── Window slicing ────────────────────────────────────────────────────────

    def _build_windows(self, df: pd.DataFrame) -> List[dict]:
        """
        Return list of dicts with keys:
          window_id, train_df, test_df, start_date, test_start, test_end
        """
        times       = pd.to_datetime(df["time"])
        data_start  = pd.Timestamp(self.cfg.data_start, tz="UTC")
        min_train   = self.cfg.min_train_months
        slide_mo    = self.cfg.test_window_months

        # First test window starts after min_train_months
        first_test_start = data_start + pd.DateOffset(months=min_train)
        last_possible    = times.max() - pd.DateOffset(months=slide_mo)

        windows = []
        wid     = 1
        test_start = first_test_start

        while test_start <= last_possible:
            test_end = test_start + pd.DateOffset(months=slide_mo)

            train_mask = (times >= data_start) & (times < test_start)
            test_mask  = (times >= test_start) & (times < test_end)

            train_df = df[train_mask].reset_index(drop=True)
            test_df  = df[test_mask].reset_index(drop=True)

            if len(train_df) < 1000 or len(test_df) < 500:
                test_start += pd.DateOffset(months=slide_mo)
                wid += 1
                continue

            windows.append({
                "window_id":   wid,
                "train_df":    train_df,
                "test_df":     test_df,
                "start_date":  str(data_start.date()),
                "test_start":  str(test_start.date()),
                "test_end":    str(test_end.date()),
            })
            test_start += pd.DateOffset(months=slide_mo)
            wid += 1

        return windows

    # ── Train one window ──────────────────────────────────────────────────────

    def _train_window(self, train_df: pd.DataFrame, window_id: int) -> PPOAgent:
        cfg = self.cfg
        obs_dim = N_FEATURES + 2   # 46 market features + direction + entry_distance

        agent = PPOAgent(
            obs_dim    = obs_dim,
            n_actions  = 4,
            hidden     = tuple(cfg.hidden),
            lr         = cfg.lr,
            gamma      = cfg.gamma,
            batch_size = cfg.batch_size,
            seq_len    = cfg.seq_len,
        )

        train_env = TradingEnv(
            train_df,
            initial_equity  = cfg.initial_equity,
            episode_length  = cfg.episode_length,
        )

        # Use a random slice of train data as eval env (last 20%)
        split      = int(len(train_df) * 0.8)
        eval_slice = train_df.iloc[split:].reset_index(drop=True)
        eval_env   = TradingEnv(
            eval_slice,
            initial_equity = cfg.initial_equity,
            episode_length = cfg.episode_length,
        ) if len(eval_slice) > 500 else None

        save_path = str(Path(cfg.save_dir) / f"window_{window_id:02d}")
        Path(save_path).mkdir(parents=True, exist_ok=True)

        agent.learn(
            train_env,
            total_timesteps  = cfg.total_steps,
            eval_env         = eval_env,
            eval_freq        = cfg.eval_freq,
            n_eval_episodes  = cfg.n_eval_episodes,
            save_path        = save_path,
            verbose          = 1,
        )
        return agent

    # ── Evaluate one window ───────────────────────────────────────────────────

    def _evaluate_window(
        self,
        agent:     PPOAgent,
        test_df:   pd.DataFrame,
        window_id: int,
        start_date: str,
        test_end:  str,
        stress:    bool = False,
    ) -> WindowResult:
        """
        Run agent deterministically over the full test window.
        Returns a populated WindowResult.
        """
        env = TradingEnv(
            test_df,
            initial_equity = self.cfg.initial_equity,
            stress_fees    = stress,
        )

        obs, _ = env.reset()
        agent.reset_obs_buffer()   # clear frame stack for new episode
        done   = False
        equity_curve   = [self.cfg.initial_equity]
        trades         = 0
        wins           = 0
        losses         = 0
        gross_profit   = 0.0
        gross_loss     = 0.0
        max_single_tp  = 0.0
        prev_equity    = self.cfg.initial_equity
        in_position    = False

        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done   = terminated or truncated

            eq     = info["equity"]
            equity_curve.append(eq)

            # Detect trade close from equity change when NOT in position
            pos = info.get("position", 0)
            if in_position and pos == 0:
                # Trade just closed
                delta = eq - prev_equity
                trades += 1
                if delta > 0:
                    wins         += 1
                    gross_profit += delta
                    max_single_tp = max(max_single_tp, delta)
                else:
                    losses    += 1
                    gross_loss += delta

            in_position = (pos != 0)
            prev_equity = eq

        net_pnl = equity_curve[-1] - equity_curve[0]

        return WindowResult(
            window_id       = window_id,
            start_date      = start_date,
            end_date        = test_end,
            equity_curve    = equity_curve,
            trades          = trades,
            wins            = wins,
            losses          = losses,
            gross_profit    = gross_profit,
            gross_loss      = gross_loss,
            max_single_trade_profit = max_single_tp,
            net_pnl         = net_pnl,
        )

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run(self) -> dict:
        cfg     = self.cfg
        t_start = time.time()

        print(f"\n{'═'*65}", flush=True)
        print(f"  Walk-Forward Pipeline  —  {cfg.coin}", flush=True)
        print(f"  Steps/window: {cfg.total_steps:,}  |  "
              f"Test window: {cfg.test_window_months}mo", flush=True)
        print(f"{'═'*65}\n", flush=True)

        df      = self._load_features()
        windows = self._build_windows(df)

        print(f"  {len(windows)} walk-forward windows built.\n", flush=True)

        window_results: List[WindowResult] = []

        for i, w in enumerate(windows, 1):
            print(f"\n{'─'*65}", flush=True)
            print(f"  Window {w['window_id']}  "
                  f"train=[{w['start_date']} → {w['test_start']}]  "
                  f"test=[{w['test_start']} → {w['test_end']}]", flush=True)
            print(f"  Train candles: {len(w['train_df']):,}  "
                  f"Test candles: {len(w['test_df']):,}", flush=True)
            print(f"{'─'*65}\n", flush=True)

            # Train win predictor on training data, inject p_win into both splits
            predictor = WinPredictor(sl_pct=0.02, tp_pct=0.04, max_bars=336)
            predictor.fit(w["train_df"])
            w["train_df"] = w["train_df"].copy()
            w["test_df"]  = w["test_df"].copy()
            w["train_df"]["p_win"] = predictor.predict(w["train_df"])
            w["test_df"]["p_win"]  = predictor.predict(w["test_df"])
            fitted = "yes" if predictor._fitted else "no (too little data)"
            print(f"  WinPredictor fitted={fitted}", flush=True)

            # Train
            agent = self._train_window(w["train_df"], w["window_id"])

            # Evaluate on test window
            result = self._evaluate_window(
                agent,
                w["test_df"],
                window_id  = w["window_id"],
                start_date = w["start_date"],
                test_end   = w["test_end"],
            )

            # Stress test
            if cfg.run_stress_test:
                stress_result = self._evaluate_window(
                    agent,
                    w["test_df"],
                    window_id  = w["window_id"],
                    start_date = w["start_date"],
                    test_end   = w["test_end"],
                    stress     = True,
                )
                result.stress_net_pnl = stress_result.net_pnl

            window_results.append(result)

            # Quick window summary
            start_eq = result.equity_curve[0] if result.equity_curve else self.cfg.initial_equity
            pnl_pct  = result.net_pnl / max(start_eq, 1e-8) * 100
            print(f"\n  Window {w['window_id']} result:  "
                  f"trades={result.trades}  wr={result.win_rate*100:.0f}%  "
                  f"pnl={pnl_pct:+.2f}%  "
                  f"sharpe={result.sharpe:.3f}  "
                  f"maxDD={result.max_drawdown*100:.2f}%",
                  flush=True)

        # Full validation report
        validation = validate_windows(window_results)

        elapsed = time.time() - t_start
        print(f"  Total time: {int(elapsed//60):02d}:{int(elapsed%60):02d}",
              flush=True)

        return {
            "validation": validation,
            "windows":    window_results,
            "config":     cfg,
        }
