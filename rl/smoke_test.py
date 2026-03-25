"""
smoke_test.py
─────────────
Quick integration check: env + agent run without errors.
Does NOT need real market data — generates synthetic candles.

Run from rl/ directory:
    python smoke_test.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS = "✓"
FAIL = "✗"

def _check(name, condition, detail=""):
    status = f"{PASS} PASS" if condition else f"{FAIL} FAIL"
    line   = f"  {status}  {name}"
    if detail:
        line += f"  ({detail})"
    print(line, flush=True)
    return condition


def _make_synthetic_df(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV candles that pass basic sanity checks.
    Price: geometric random walk. Volume: log-normal.
    """
    rng   = np.random.default_rng(seed)
    price = 50_000.0
    rows  = []
    t     = pd.Timestamp("2023-01-01", tz="UTC")

    for _ in range(n):
        ret   = rng.normal(0, 0.002)
        close = max(price * (1 + ret), 100.0)
        high  = close * (1 + abs(rng.normal(0, 0.001)))
        low   = close * (1 - abs(rng.normal(0, 0.001)))
        open_ = price
        vol   = max(rng.lognormal(10, 1), 1.0)
        rows.append({"time": t, "open": open_, "high": high,
                     "low": low, "close": close, "volume": vol})
        price = close
        t    += pd.Timedelta("15min")

    return pd.DataFrame(rows)


def run():
    all_pass = True
    print(f"\n{'═'*60}", flush=True)
    print("  Smoke Test — Phase 2+3 integration", flush=True)
    print(f"{'═'*60}\n", flush=True)

    # ── 1. Feature computation ─────────────────────────────────────────────────
    print("── Feature pipeline ──", flush=True)
    try:
        from features import add_features, N_FEATURES, FEATURE_COLS
        df_raw  = _make_synthetic_df(2000)
        df_feat = add_features(df_raw)
        ok = _check("add_features() runs without error",
                    len(df_feat) > 0, f"{len(df_feat)} rows")
        all_pass &= ok

        ok = _check(f"Feature matrix has {N_FEATURES} columns",
                    all(c in df_feat.columns for c in FEATURE_COLS))
        all_pass &= ok

        ok = _check("No NaN in feature matrix",
                    not df_feat[FEATURE_COLS].isna().any().any())
        all_pass &= ok

        ok = _check("ATR column present",
                    "atr" in df_feat.columns)
        all_pass &= ok
    except Exception as e:
        _check("Feature pipeline", False, str(e))
        all_pass = False
        df_feat  = None

    # ── 2. Reward module ───────────────────────────────────────────────────────
    print("\n── Reward module ──", flush=True)
    try:
        from env.reward import (reward_on_close, reward_on_hold,
                                check_circuit_breaker, TradeResult)

        r = reward_on_close(TradeResult(
            net_pnl=100.0, account_equity=10_000.0,
            max_adverse_excursion=0.0, atr_at_entry=150.0
        ))
        ok = _check("reward_on_close (win, no MAE) > 0", r > 0, f"r={r:.4f}")
        all_pass &= ok

        r2 = reward_on_close(TradeResult(
            net_pnl=-50.0, account_equity=10_000.0,
            max_adverse_excursion=0.0, atr_at_entry=150.0
        ))
        ok = _check("reward_on_close (loss) < 0", r2 < 0, f"r={r2:.4f}")
        all_pass &= ok

        ok = _check("reward_on_hold within grace = 0.0",
                    reward_on_hold(10) == 0.0)
        all_pass &= ok

        ok = _check("reward_on_hold beyond grace < 0",
                    reward_on_hold(30) < 0)
        all_pass &= ok

        trig, pen = check_circuit_breaker(9_400.0, 10_000.0)
        ok = _check("Circuit breaker triggers at -6%", trig and pen < 0)
        all_pass &= ok

        no_trig, _ = check_circuit_breaker(9_700.0, 10_000.0)
        ok = _check("Circuit breaker silent at -3%", not no_trig)
        all_pass &= ok
    except Exception as e:
        _check("Reward module", False, str(e))
        all_pass = False

    # ── 3. Fee model ───────────────────────────────────────────────────────────
    print("\n── Fee model ──", flush=True)
    try:
        from env.fee_model import calc_trade_cost, calc_slippage

        slip = calc_slippage(0.5, 0.5, 0.3)
        ok = _check("calc_slippage returns positive float",
                    0 < slip < 0.01, f"slip={slip:.5f}")
        all_pass &= ok

        slip_stress = calc_slippage(0.5, 0.5, 0.3, stress=True)
        ok = _check("Stress slippage = 2× normal",
                    abs(slip_stress - 2 * slip) < 1e-8)
        all_pass &= ok

        cost = calc_trade_cost(50_000, 50_500, 0.01,
                               0.5, 0.5, 0.5, 0.5, 0.3)
        ok = _check("calc_trade_cost total > 0",
                    cost["total"] > 0, f"total=${cost['total']:.4f}")
        all_pass &= ok
    except Exception as e:
        _check("Fee model", False, str(e))
        all_pass = False

    # ── 4. TradingEnv ──────────────────────────────────────────────────────────
    print("\n── TradingEnv ──", flush=True)
    if df_feat is not None:
        try:
            from env.trading_env import TradingEnv, HOLD, LONG, SHORT, CLOSE

            env = TradingEnv(df_feat, initial_equity=10_000.0,
                             episode_length=500)
            obs, info = env.reset()
            from features import N_FEATURES
            ok = _check(f"Reset → obs shape ({N_FEATURES+2},)",
                        obs.shape == (N_FEATURES + 2,), str(obs.shape))
            all_pass &= ok

            ok = _check("Obs has no NaN", not np.isnan(obs).any())
            all_pass &= ok

            # Step with Hold
            action = {"direction": HOLD, "sizing": np.array([0.02], dtype=np.float32)}
            obs2, rew, term, trunc, info2 = env.step(action)
            ok = _check("Hold step returns valid obs", not np.isnan(obs2).any())
            all_pass &= ok

            # Step with Long
            action = {"direction": LONG, "sizing": np.array([0.15], dtype=np.float32)}
            obs3, rew, term, trunc, info3 = env.step(action)
            ok = _check("Long step opens position",
                        info3.get("position", 0) == 1,
                        f"pos={info3.get('position')}")
            all_pass &= ok

            # Step with Close
            action = {"direction": CLOSE, "sizing": np.array([0.0], dtype=np.float32)}
            obs4, rew, term, trunc, info4 = env.step(action)
            ok = _check("Close step closes position",
                        info4.get("position", 1) == 0,
                        f"pos={info4.get('position')}")
            all_pass &= ok

            # Run a full episode with random actions
            obs, _ = env.reset()
            steps  = 0
            done   = False
            while not done and steps < 2000:
                d = np.random.randint(4)
                s = float(np.random.uniform(0.05, 0.3))
                a = {"direction": d, "sizing": np.array([s], dtype=np.float32)}
                obs, rew, term, trunc, info = env.step(a)
                done  = term or trunc
                steps += 1

            ok = _check(f"Full episode ran {steps} steps without crash",
                        steps > 0, f"eq={info['equity']:,.2f}")
            all_pass &= ok

        except Exception as e:
            import traceback
            _check("TradingEnv", False, str(e))
            traceback.print_exc()
            all_pass = False

    # ── 5. DQN Agent ──────────────────────────────────────────────────────────
    print("\n── DQN Agent ──", flush=True)
    if df_feat is not None:
        try:
            from env.trading_env import TradingEnv
            from dqn import DQNAgent
            from features import N_FEATURES

            obs_dim = N_FEATURES + 2
            agent   = DQNAgent(
                obs_dim         = obs_dim,
                n_actions       = 4,
                hidden          = [64, 64],   # small for speed
                learning_starts = 200,
                buffer_size     = 10_000,
                batch_size      = 64,
                eps_decay_steps = 500,
            )

            env = TradingEnv(df_feat, initial_equity=10_000.0,
                             episode_length=300)

            action = agent.predict(np.zeros(obs_dim, dtype=np.float32))
            ok = _check("predict() returns Dict with direction + sizing",
                        "direction" in action and "sizing" in action)
            all_pass &= ok

            ok = _check("sizing in [0, 1]",
                        0.0 <= float(action["sizing"][0]) <= 1.0,
                        f"sz={float(action['sizing'][0]):.4f}")
            all_pass &= ok

            # Short training run (500 steps)
            agent.learn(env, total_timesteps=500, verbose=0)
            ok = _check("500-step training run completed", True,
                        f"buffer={len(agent.buffer)}")
            all_pass &= ok

        except Exception as e:
            import traceback
            _check("DQN Agent", False, str(e))
            traceback.print_exc()
            all_pass = False

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}", flush=True)
    if all_pass:
        print(f"  {PASS}  ALL SMOKE TESTS PASSED — ready for training", flush=True)
    else:
        print(f"  {FAIL}  SOME TESTS FAILED — fix before training", flush=True)
    print(f"{'═'*60}\n", flush=True)

    return all_pass


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
