"""
pipeline.py
───────────
SafetyPipeline — the single entry point that ties all safety layers together.

This is what sits between the trained agent and the exchange.

Architecture:
  Agent obs → Agent.predict() → SafetyPipeline.evaluate() → order / blocked

evaluate() does:
  1. Feed obs to DriftDetector
  2. Compute signal_score from Q-values
  3. Estimate trade cost via fee_model
  4. Run ExecutionGate (all 5 checks)
  5. If allowed → clamp sizing, record in PositionLimits
  6. Return SafetyDecision

Also exposes:
  on_trade_closed(coin, pnl, equity) — update all state after a close
  status() — full snapshot of all safety layer states

Usage (paper/live trading loop):
    pipeline = SafetyPipeline.from_model_dir("models/BTC_single/")
    ...
    obs  = get_live_obs()
    action = agent.predict(obs, deterministic=True)
    decision = pipeline.evaluate(obs, action, coin="BTC",
                                 entry_price=price, equity=equity,
                                 vol_regime=vr, rel_volume=rv)
    if decision.allowed:
        # place order with decision.final_sizing
    else:
        print("Blocked:", decision.reason)
"""

import sys
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_RL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_RL_DIR))

from safety.position_limits import PositionLimits
from safety.trust_ladder    import TrustLadder
from safety.drift_detector  import DriftDetector
from safety.execution_gate  import ExecutionGate, TradeProposal, compute_signal_score
from env.fee_model          import calc_slippage, TAKER_FEE
from features               import N_FEATURES


@dataclass
class SafetyDecision:
    allowed:       bool
    reason:        str
    gate:          str
    final_sizing:  float        # clamped sizing (0 if blocked)
    signal_score:  float
    est_cost_pct:  float
    drift_level:   str          # "ok" / "warn" / "block"


class SafetyPipeline:
    """
    Combines PositionLimits + TrustLadder + DriftDetector + ExecutionGate
    into a single evaluate() call.

    Parameters
    ----------
    position_limits : PositionLimits
    trust_ladder    : TrustLadder
    drift_detector  : DriftDetector
    starting_equity : float
    """

    def __init__(
        self,
        position_limits: PositionLimits,
        trust_ladder:    TrustLadder,
        drift_detector:  DriftDetector,
    ):
        self._limits  = position_limits
        self._trust   = trust_ladder
        self._drift   = drift_detector
        self._gate    = ExecutionGate(position_limits, trust_ladder, drift_detector)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_model_dir(
        cls,
        model_dir:      str,
        starting_equity: float = 10_000.0,
        starting_rung:   int   = 1,
    ) -> "SafetyPipeline":
        """
        Load drift reference from model_dir/drift_ref.npz (if exists).
        Creates a fresh DriftDetector if file not found.
        """
        from features import FEATURE_COLS

        drift_path = Path(model_dir) / "drift_ref.npz"
        if drift_path.exists():
            drift = DriftDetector.load(str(drift_path))
        else:
            # No reference yet — create an uninitialised detector
            # (will be "ok" until MIN_LIVE_OBS observations accumulate)
            dummy_ref = np.zeros((10, N_FEATURES))
            drift = DriftDetector.from_training_data(
                dummy_ref, feature_names=FEATURE_COLS
            )

        limits  = PositionLimits(starting_equity=starting_equity)
        trust   = TrustLadder(starting_rung=starting_rung)
        return cls(limits, trust, drift)

    @classmethod
    def build_drift_reference(
        cls,
        model_dir:     str,
        feature_matrix: np.ndarray,
        feature_names:  list,
    ):
        """
        Call once after training to save the drift reference distribution.
        feature_matrix: (n_samples, N_FEATURES) from training data.
        """
        detector = DriftDetector.from_training_data(feature_matrix, feature_names)
        out_path = str(Path(model_dir) / "drift_ref.npz")
        detector.save(out_path)
        print(f"  Drift reference saved → {out_path}", flush=True)
        return detector

    # ── Main evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        obs:         np.ndarray,    # full 40-dim obs (or 38-dim market only)
        action:      dict,          # {"direction": int, "sizing": array}
        coin:        str,
        entry_price: float,
        equity:      float,
        vol_regime:  float,
        rel_volume:  float,
    ) -> SafetyDecision:
        """
        Evaluate whether to execute the agent's proposed trade.

        Call this every step — even Hold steps (for drift monitoring).
        Only entry actions (Long=1, Short=2) go through the full gate.
        Hold and Close always pass through.
        """
        direction_act = int(action["direction"])
        sizing        = float(action["sizing"].flat[0])

        # Always feed obs to drift detector (market features only)
        market_obs = obs[:N_FEATURES]
        self._drift.add_observation(market_obs)

        # Compute Q-value signal score (needs online network forward pass)
        # The agent should pass q_values; if not available, use sizing as proxy
        q_values     = action.get("q_values", np.zeros(4, dtype=np.float32))
        signal_score = compute_signal_score(q_values, direction_act)

        # Hold / Close always allowed — no gate needed
        if direction_act in (0, 3):   # Hold or Close
            return SafetyDecision(
                allowed      = True,
                reason       = "",
                gate         = "",
                final_sizing = sizing,
                signal_score = signal_score,
                est_cost_pct = 0.0,
                drift_level  = self._drift.check().level,
            )

        # Entry action — run the full gate
        direction_sign = 1 if direction_act == 1 else -1
        clamped_sizing = self._limits.clamped_sizing(sizing)

        # Estimate round-trip cost
        slip_pct     = calc_slippage(vol_regime, rel_volume, clamped_sizing)
        est_cost_pct = 2 * TAKER_FEE + 2 * slip_pct   # entry + exit

        proposal = TradeProposal(
            coin         = coin,
            direction    = direction_sign,
            sizing       = clamped_sizing,
            entry_price  = entry_price,
            equity       = equity,
            q_values     = q_values,
            signal_score = signal_score,
            vol_regime   = vol_regime,
            rel_volume   = rel_volume,
            est_cost_pct = est_cost_pct,
        )

        result = self._gate.check(proposal)

        if result.allowed:
            notional = equity * clamped_sizing
            self._limits.record_open(
                coin        = coin,
                direction   = direction_sign,
                entry_price = entry_price,
                size        = notional / max(entry_price, 1e-8),
                notional    = notional,
            )

        return SafetyDecision(
            allowed      = result.allowed,
            reason       = result.reason,
            gate         = result.gate,
            final_sizing = clamped_sizing if result.allowed else 0.0,
            signal_score = signal_score,
            est_cost_pct = est_cost_pct,
            drift_level  = self._drift.check().level,
        )

    # ── Post-trade update ─────────────────────────────────────────────────────

    def on_trade_closed(self, coin: str, pnl: float, equity: float):
        """Call when a position closes. Updates limits + trust ladder."""
        self._limits.record_close(coin, pnl)
        self._trust.record_trade(pnl, equity)

    def on_day_end(self, day_pnl: float, day_start_equity: float,
                   week_pnl: float = 0.0):
        """Call at end of each trading day."""
        self._trust.end_of_day(day_pnl, day_start_equity, week_pnl)

    # ── Status snapshot ───────────────────────────────────────────────────────

    def status(self) -> dict:
        drift = self._drift.check()
        return {
            "limits":  self._limits.status(),
            "trust":   self._trust.status(),
            "drift": {
                "level":            drift.level,
                "max_psi":          round(drift.max_psi, 4),
                "mean_psi":         round(drift.mean_psi, 4),
                "drifted_features": drift.drifted_features,
                "n_live_obs":       drift.n_live_obs,
            },
        }

    def print_status(self):
        s = self.status()
        print(f"\n── Safety Pipeline Status ──────────────────────", flush=True)
        t = s["trust"]
        print(f"  Trust:   Rung {t['rung']} ({t['rung_name']})  "
              f"lev={t['leverage_cap']}×  "
              f"max_pos={t['max_positions']}  "
              f"paper={'yes' if t['paper_only'] else 'no'}", flush=True)
        l = s["limits"]
        print(f"  Limits:  open={l['n_open']}/{l['max_positions'] if 'max_positions' in l else '?'}  "
              f"daily_loss={l['daily_loss_pct']*100:+.2f}%  "
              f"halted={l['halted']}", flush=True)
        d = s["drift"]
        print(f"  Drift:   [{d['level'].upper()}]  "
              f"max_PSI={d['max_psi']:.3f}  "
              f"n={d['n_live_obs']}", flush=True)
        print(f"────────────────────────────────────────────────\n", flush=True)
