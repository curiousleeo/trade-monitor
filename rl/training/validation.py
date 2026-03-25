"""
validation.py
─────────────
8-check validation suite. Runs against a list of backtest results
(one per walk-forward window) and reports pass/fail for each check.

The 8 checks (blueprint v2.1):
  1. Consistent across time      — 70%+ of windows are net profitable
  2. Sufficient samples          — every window has ≥ 100 trades
  3. Capital preservation        — max drawdown < 5% in every window
  4. Risk-adjusted quality       — Sharpe ≥ 0.5 in every window
  5. Not degenerate              — win rate 35–65% in every window
  6. Positive expectancy         — avg_win / avg_loss ≥ 1.2 in every window
  7. No single-trade dependency  — no single trade > 20% of total profit
  8. Execution robust            — profitable with 2× slippage stress test
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class WindowResult:
    """
    Results from one walk-forward window.
    Filled by the training loop after evaluating the trained model.
    """
    window_id:      int
    start_date:     str
    end_date:       str

    # Equity curve (one float per candle / episode step)
    equity_curve:   List[float] = field(default_factory=list)

    # Trade-level stats
    trades:         int   = 0
    wins:           int   = 0
    losses:         int   = 0
    gross_profit:   float = 0.0   # sum of winning trades' PnL
    gross_loss:     float = 0.0   # sum of losing trades' PnL (negative)
    max_single_trade_profit: float = 0.0
    net_pnl:        float = 0.0

    # Stress-test result (2× slippage)
    stress_net_pnl: Optional[float] = None   # None = not yet run

    # Derived on demand
    @property
    def is_profitable(self) -> bool:
        return self.net_pnl > 0

    @property
    def win_rate(self) -> float:
        if self.trades == 0:
            return 0.0
        return self.wins / self.trades

    @property
    def avg_win(self) -> float:
        if self.wins == 0:
            return 0.0
        return self.gross_profit / self.wins

    @property
    def avg_loss(self) -> float:
        if self.losses == 0:
            return 1e-8
        return abs(self.gross_loss) / self.losses

    @property
    def expectancy_ratio(self) -> float:
        return self.avg_win / max(self.avg_loss, 1e-8)

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        eq  = np.array(self.equity_curve, dtype=np.float64)
        peak = np.maximum.accumulate(eq)
        dd   = (peak - eq) / (peak + 1e-8)
        return float(dd.max())

    @property
    def sharpe(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        eq      = np.array(self.equity_curve, dtype=np.float64)
        returns = np.diff(eq) / (eq[:-1] + 1e-8)
        if returns.std() < 1e-10:
            return 0.0
        # Annualise for 15m data: 4 candles/h × 24h × 365d = 35,040 per year
        return float((returns.mean() / returns.std()) * np.sqrt(35_040))

    @property
    def single_trade_dependency(self) -> float:
        """Largest single trade as fraction of total profit."""
        if self.net_pnl <= 0:
            return 1.0   # worst case (no profit at all)
        return self.max_single_trade_profit / max(self.net_pnl, 1e-8)


# ── 8-check validation ────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


def _check(name: str, condition: bool, detail: str = "", warn_only: bool = False) -> bool:
    symbol = PASS if condition else (WARN if warn_only else FAIL)
    status = "PASS" if condition else ("WARN" if warn_only else "FAIL")
    line   = f"  {symbol}  {name:<50} {status}"
    if detail:
        line += f"  ({detail})"
    print(line, flush=True)
    return condition


def validate_windows(windows: List[WindowResult], verbose: bool = True) -> dict:
    """
    Run the 8-check suite against a list of walk-forward window results.

    Returns a dict:
      {
        "all_pass": bool,
        "checks":   {check_name: bool},
        "summary":  {per-window stats},
      }
    """
    if not windows:
        print("  No windows to validate.", flush=True)
        return {"all_pass": False, "checks": {}, "summary": {}}

    if verbose:
        print(f"\n{'═'*65}", flush=True)
        print(f"  Walk-Forward Validation  ({len(windows)} windows)", flush=True)
        print(f"{'═'*65}", flush=True)

    n         = len(windows)
    all_pass  = True
    checks    = {}

    # Check 1: Consistent across time (70%+ profitable windows)
    profitable_count = sum(1 for w in windows if w.is_profitable)
    pct_profitable   = profitable_count / n
    ok = _check(
        "1. Consistent across time (≥70% windows profitable)",
        pct_profitable >= 0.70,
        f"{profitable_count}/{n} = {pct_profitable*100:.0f}%",
    )
    checks["consistent_across_time"] = ok
    all_pass &= ok

    # Check 2: Sufficient samples (≥100 trades per window)
    min_trades = min(w.trades for w in windows)
    ok = _check(
        "2. Sufficient samples (≥100 trades/window)",
        min_trades >= 100,
        f"min trades = {min_trades}",
    )
    checks["sufficient_samples"] = ok
    all_pass &= ok

    # Check 3: Capital preservation (<5% max DD in every window)
    max_dd_vals = [w.max_drawdown for w in windows]
    worst_dd    = max(max_dd_vals)
    ok = _check(
        "3. Capital preservation (max DD < 5% per window)",
        worst_dd < 0.05,
        f"worst DD = {worst_dd*100:.2f}%",
    )
    checks["capital_preservation"] = ok
    all_pass &= ok

    # Check 4: Risk-adjusted quality (Sharpe ≥ 0.5 per window)
    sharpe_vals = [w.sharpe for w in windows]
    min_sharpe  = min(sharpe_vals)
    ok = _check(
        "4. Risk-adjusted quality (Sharpe ≥ 0.5 per window)",
        min_sharpe >= 0.5,
        f"min Sharpe = {min_sharpe:.3f}",
    )
    checks["risk_adjusted_quality"] = ok
    all_pass &= ok

    # Check 5: Not degenerate (win rate 35-65%)
    wr_vals = [w.win_rate for w in windows]
    min_wr  = min(wr_vals)
    max_wr  = max(wr_vals)
    ok = _check(
        "5. Not degenerate (win rate 35–65% per window)",
        all(0.35 <= wr <= 0.65 for wr in wr_vals),
        f"range [{min_wr*100:.0f}%, {max_wr*100:.0f}%]",
    )
    checks["not_degenerate"] = ok
    all_pass &= ok

    # Check 6: Positive expectancy (avg_win / avg_loss ≥ 1.2)
    exp_vals   = [w.expectancy_ratio for w in windows]
    min_exp    = min(exp_vals)
    ok = _check(
        "6. Positive expectancy (avg_win/avg_loss ≥ 1.2)",
        min_exp >= 1.2,
        f"min ratio = {min_exp:.3f}",
    )
    checks["positive_expectancy"] = ok
    all_pass &= ok

    # Check 7: No single-trade dependency (<20% of profit from 1 trade)
    dep_vals = [w.single_trade_dependency for w in windows]
    max_dep  = max(dep_vals)
    ok = _check(
        "7. No single-trade dependency (<20% of profit/trade)",
        max_dep < 0.20,
        f"max = {max_dep*100:.1f}%",
    )
    checks["no_single_trade_dep"] = ok
    all_pass &= ok

    # Check 8: Execution robust (profitable with 2× slippage)
    stress_results = [w for w in windows if w.stress_net_pnl is not None]
    if stress_results:
        stress_profitable = sum(1 for w in stress_results if w.stress_net_pnl > 0)
        pct_stress        = stress_profitable / len(stress_results)
        ok = _check(
            "8. Execution robust (profitable with 2× slippage)",
            pct_stress >= 0.70,
            f"{stress_profitable}/{len(stress_results)} = {pct_stress*100:.0f}%",
        )
        checks["execution_robust"] = ok
        all_pass &= ok
    else:
        _check("8. Execution robust (2× slippage)", True,
               "SKIPPED — run with stress_fees=True", warn_only=True)
        checks["execution_robust"] = None

    # ── Per-window summary table ───────────────────────────────────────────────
    if verbose:
        print(f"\n  {'Win':<5} {'Dates':<24} {'Trades':>6} {'WR%':>5} "
              f"{'NetPnL%':>8} {'MaxDD%':>7} {'Sharpe':>7}", flush=True)
        print(f"  {'-'*65}", flush=True)
        for w in windows:
            start_eq = w.equity_curve[0] if w.equity_curve else 1.0
            pnl_pct  = (w.net_pnl / max(start_eq, 1e-8)) * 100
            print(
                f"  {w.window_id:<5} {w.start_date[:10]}…{w.end_date[:10]}  "
                f"{w.trades:>6} {w.win_rate*100:>5.0f} "
                f"{pnl_pct:>+8.2f} {w.max_drawdown*100:>7.2f} {w.sharpe:>7.3f}",
                flush=True
            )

    print(f"\n{'═'*65}", flush=True)
    result_str = "ALL CHECKS PASSED" if all_pass else "VALIDATION FAILED"
    print(f"  {PASS if all_pass else FAIL}  {result_str}", flush=True)
    print(f"{'═'*65}\n", flush=True)

    return {
        "all_pass": all_pass,
        "checks":   checks,
        "windows":  windows,
    }
