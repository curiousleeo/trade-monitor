"""
execution_gate.py
─────────────────
Pre-trade execution gate — the final checkpoint before any order is sent.

Every proposed trade must pass ALL gates:
  1. Position limits     — hard caps (max positions, daily loss limit)
  2. Trust ladder        — current rung allows live trading
  3. Drift detector      — market distribution hasn't drifted too far
  4. Signal strength     — composite score ≥ minimum threshold
  5. Slippage estimate   — projected cost doesn't eat the edge

If any gate fails, the trade is blocked and a reason is returned.
The gate is stateless itself — it delegates to the individual safety modules.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


# ── Thresholds ────────────────────────────────────────────────────────────────

MIN_SIGNAL_SCORE   = 0.55    # agent's Q-value confidence (normalised 0–1)
MAX_COST_PCT       = 0.005   # projected fee+slip must be < 0.5% of notional


@dataclass
class TradeProposal:
    """All info about a proposed trade, assembled by the pipeline."""
    coin:          str
    direction:     int      # 1=long, -1=short
    sizing:        float    # fraction of equity (already clamped by position_limits)
    entry_price:   float
    equity:        float

    # Agent outputs
    q_values:      np.ndarray   # raw Q-values for all 4 actions
    signal_score:  float        # normalised confidence [0, 1]

    # Market context (from obs vector)
    vol_regime:    float        # [0, 1]
    rel_volume:    float        # [0, 1]

    # Estimated cost (from fee_model)
    est_cost_pct:  float        # estimated (fee + slip) / notional


@dataclass
class GateResult:
    allowed:  bool
    reason:   str   # empty if allowed
    gate:     str   # which gate blocked it ("position_limits", "trust", etc.)

    @classmethod
    def ok(cls) -> "GateResult":
        return cls(allowed=True, reason="", gate="")

    @classmethod
    def block(cls, gate: str, reason: str) -> "GateResult":
        return cls(allowed=False, reason=reason, gate=gate)


class ExecutionGate:
    """
    Wraps all safety layers into a single check() call.

    Usage:
        gate = ExecutionGate(position_limits, trust_ladder, drift_detector)
        result = gate.check(proposal)
        if result.allowed:
            # send order
    """

    def __init__(self, position_limits, trust_ladder, drift_detector,
                 min_signal_score: float = MIN_SIGNAL_SCORE,
                 max_cost_pct:     float = MAX_COST_PCT):
        self._limits  = position_limits
        self._trust   = trust_ladder
        self._drift   = drift_detector
        self.min_signal_score = min_signal_score
        self.max_cost_pct     = max_cost_pct

    def check(self, proposal: TradeProposal) -> GateResult:
        """
        Run all gates in order. Returns on first failure.
        Gates are ordered cheapest-to-most-expensive.
        """

        # ── Gate 1: Trust ladder ──────────────────────────────────────────────
        if self._trust.is_paper_only:
            return GateResult.block(
                "trust_ladder",
                f"rung={self._trust.current_rung.name} is paper-only"
            )

        if self._trust.max_positions < 1:
            return GateResult.block(
                "trust_ladder",
                "trust ladder max_positions < 1"
            )

        # ── Gate 2: Position limits ───────────────────────────────────────────
        allowed, reason = self._limits.check_entry(
            coin      = proposal.coin,
            direction = proposal.direction,
            sizing    = proposal.sizing,
            equity    = proposal.equity,
        )
        if not allowed:
            return GateResult.block("position_limits", reason)

        # Also check trust ladder's position cap
        if self._limits.n_open >= self._trust.max_positions:
            return GateResult.block(
                "trust_ladder",
                f"trust rung limits to {self._trust.max_positions} positions"
            )

        # ── Gate 3: Drift detector ────────────────────────────────────────────
        drift_status = self._drift.check()
        if drift_status.level == "block":
            return GateResult.block(
                "drift_detector",
                f"market drift detected (max_PSI={drift_status.max_psi:.3f}): "
                f"{', '.join(drift_status.drifted_features[:3])}"
            )

        # ── Gate 4: Signal strength ───────────────────────────────────────────
        if proposal.signal_score < self.min_signal_score:
            return GateResult.block(
                "signal_strength",
                f"signal_score={proposal.signal_score:.3f} < {self.min_signal_score}"
            )

        # ── Gate 5: Cost check ────────────────────────────────────────────────
        if proposal.est_cost_pct > self.max_cost_pct:
            return GateResult.block(
                "cost_check",
                f"est_cost={proposal.est_cost_pct*100:.3f}% > max {self.max_cost_pct*100:.1f}%"
            )

        return GateResult.ok()


# ── Signal score helper ───────────────────────────────────────────────────────

def compute_signal_score(q_values: np.ndarray, chosen_action: int) -> float:
    """
    Compute a normalised confidence score [0, 1] from Q-values.

    Method: softmax(Q)[chosen_action]
    A score near 1.0 = agent is very confident this is the best action.
    A score near 0.25 = agent is roughly indifferent between 4 actions.
    """
    q      = q_values.astype(np.float64)
    q      = q - q.max()           # subtract max for numerical stability
    exp_q  = np.exp(q)
    probs  = exp_q / exp_q.sum()
    return float(probs[chosen_action])
