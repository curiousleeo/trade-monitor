/**
 * BacktestPanel — Realistic model evaluation UI
 *
 * Runs the APEX TA engine against the current coin's historical candles
 * and surfaces win rate, profit factor, drawdown, signal attribution,
 * confidence calibration, and walk-forward results.
 */

import { useMemo, useState } from 'react';
import { Candle, Coin, Timeframe } from '../types';
import { runBacktest, BacktestResult } from '../ai/backtest';

interface Props {
  candles:   Candle[];
  coin:      Coin;
  timeframe: Timeframe;
}

// ─── Equity Curve SVG ────────────────────────────────────────────────────────

function EquityCurve({ curve }: { curve: number[] }) {
  if (curve.length < 2) return <div className="bt-eq-empty">Not enough trades</div>;

  const W = 320;
  const H = 80;
  const min = Math.min(...curve);
  const max = Math.max(...curve);
  const range = max - min || 1;

  const pts = curve.map((v, i) => {
    const x = (i / (curve.length - 1)) * W;
    const y = H - ((v - min) / range) * H;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const finalBalance = curve[curve.length - 1];
  const startBalance = curve[0];
  const isUp = finalBalance >= startBalance;
  const color = isUp ? '#22c55e' : '#ef4444';

  // Fill path
  const fillPath = `M0,${H} L${pts.join(' L')} L${W},${H} Z`;
  const linePath = `M${pts.join(' L')}`;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="bt-equity-svg" preserveAspectRatio="none">
      <defs>
        <linearGradient id="eq-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor={color} stopOpacity="0.25" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <path d={fillPath} fill="url(#eq-fill)" />
      <path d={linePath} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

// ─── Stat card ────────────────────────────────────────────────────────────────

function Stat({ label, value, sub, good, bad }: {
  label: string;
  value: string;
  sub?: string;
  good?: boolean;
  bad?: boolean;
}) {
  const cls = good ? 'bt-stat--good' : bad ? 'bt-stat--bad' : '';
  return (
    <div className={`bt-stat ${cls}`}>
      <div className="bt-stat-value">{value}</div>
      <div className="bt-stat-label">{label}</div>
      {sub && <div className="bt-stat-sub">{sub}</div>}
    </div>
  );
}

// ─── Main Panel ───────────────────────────────────────────────────────────────

export function BacktestPanel({ candles, coin, timeframe }: Props) {
  const [showTrades, setShowTrades] = useState(false);

  const result: BacktestResult | null = useMemo(() => {
    if (candles.length < 120) return null;
    try {
      return runBacktest(candles, coin, timeframe);
    } catch {
      return null;
    }
  }, [candles, coin, timeframe]);

  if (!result) {
    return (
      <div className="bt-loading">
        <span className="bt-loading-dot" />
        Waiting for candle data…
      </div>
    );
  }

  if (result.trades.length === 0) {
    return (
      <div className="bt-loading">
        No trades generated on {candles.length} candles — model is too conservative or data is too short.
      </div>
    );
  }

  const winPct    = (result.winRate * 100).toFixed(1);
  const totalPct  = (result.totalReturnPct * 100).toFixed(1);
  const maxDDPct  = (result.maxDrawdownPct * 100).toFixed(1);
  const oosPct    = (result.walkForward.oosWinRate * 100).toFixed(1);
  const degPct    = (result.walkForward.degradation * 100).toFixed(1);
  const oosRet    = (result.walkForward.oosReturnPct * 100).toFixed(1);

  const isGoodWR   = result.winRate >= 0.60;
  const isGoodPF   = result.profitFactor >= 1.5;
  const isGoodDD   = result.maxDrawdownPct <= 0.15;
  const isGoodSharpe = result.sharpeRatio >= 1.0;

  return (
    <div className="bt-panel">

      {/* Header */}
      <div className="bt-header">
        <span className="bt-title">Backtest — {coin} {timeframe}</span>
        <span className="bt-meta">{candles.length} candles · {result.trades.length} trades</span>
      </div>

      {/* Equity curve */}
      <div className="bt-eq-wrap">
        <EquityCurve curve={result.equityCurve} />
        <div className="bt-eq-labels">
          <span>$1,000</span>
          <span className={result.finalBalance >= 1000 ? 'bt-eq-up' : 'bt-eq-down'}>
            ${result.finalBalance.toFixed(0)}
          </span>
        </div>
      </div>

      {/* Core stats grid */}
      <div className="bt-stats-grid">
        <Stat
          label="Win Rate"
          value={`${winPct}%`}
          sub={`${result.trades.filter(t => t.result === 'WIN').length}W / ${result.trades.filter(t => t.result === 'LOSS').length}L`}
          good={isGoodWR}
          bad={!isGoodWR}
        />
        <Stat
          label="Profit Factor"
          value={result.profitFactor >= 99 ? '∞' : result.profitFactor.toFixed(2)}
          sub="gross profit / gross loss"
          good={isGoodPF}
          bad={!isGoodPF}
        />
        <Stat
          label="Total Return"
          value={`${Number(totalPct) >= 0 ? '+' : ''}${totalPct}%`}
          sub={`$1k → $${result.finalBalance.toFixed(0)}`}
          good={result.totalReturnPct > 0}
          bad={result.totalReturnPct < 0}
        />
        <Stat
          label="Max Drawdown"
          value={`${maxDDPct}%`}
          sub="peak to trough"
          good={isGoodDD}
          bad={!isGoodDD}
        />
        <Stat
          label="Sharpe Ratio"
          value={result.sharpeRatio.toFixed(2)}
          sub="annualised"
          good={isGoodSharpe}
          bad={!isGoodSharpe}
        />
        <Stat
          label="Calmar Ratio"
          value={result.calmarRatio >= 99 ? '∞' : result.calmarRatio.toFixed(2)}
          sub="return / drawdown"
          good={result.calmarRatio >= 1}
        />
        <Stat
          label="Avg Win"
          value={`+${(result.avgWinPct * 100).toFixed(2)}%`}
          sub="net of fees"
        />
        <Stat
          label="Avg Loss"
          value={`${(result.avgLossPct * 100).toFixed(2)}%`}
          sub="net of fees"
        />
        <Stat
          label="Avg R:R"
          value={result.avgRR.toFixed(2)}
          sub="actual achieved"
          good={result.avgRR >= 1.5}
        />
        <Stat
          label="Trade Freq"
          value={`${result.tradesPerHundredCandles.toFixed(1)}/100`}
          sub="per 100 candles"
        />
        <Stat
          label="Avg Hold"
          value={`${result.avgHoldingCandles.toFixed(1)}`}
          sub="candles per trade"
        />
        <Stat
          label="OOS Win Rate"
          value={`${oosPct}%`}
          sub={`${result.walkForward.oosTrades} trades (last 40%)`}
          good={result.walkForward.oosWinRate >= 0.55}
          bad={result.walkForward.oosWinRate < 0.50}
        />
      </div>

      {/* Walk-forward result */}
      <div className="bt-section">
        <div className="bt-section-title">Walk-Forward Analysis</div>
        <div className="bt-wf-row">
          <div className="bt-wf-block">
            <div className="bt-wf-label">In-Sample (60%)</div>
            <div className="bt-wf-value">{(result.walkForward.inSampleWinRate * 100).toFixed(1)}% win</div>
            <div className="bt-wf-sub">{result.walkForward.inSampleTrades} trades</div>
          </div>
          <div className="bt-wf-arrow">→</div>
          <div className="bt-wf-block">
            <div className="bt-wf-label">Out-of-Sample (40%)</div>
            <div className={`bt-wf-value ${result.walkForward.oosWinRate >= 0.55 ? 'bt-eq-up' : 'bt-eq-down'}`}>
              {oosPct}% win
            </div>
            <div className="bt-wf-sub">{result.walkForward.oosTrades} trades · {Number(oosRet) >= 0 ? '+' : ''}{oosRet}% return</div>
          </div>
          <div className="bt-wf-block bt-wf-block--right">
            <div className="bt-wf-label">Degradation</div>
            <div className={`bt-wf-value ${Math.abs(result.walkForward.degradation) < 0.08 ? 'bt-eq-up' : 'bt-eq-down'}`}>
              {Number(degPct) > 0 ? '-' : '+'}{Math.abs(Number(degPct)).toFixed(1)}pp
            </div>
            <div className="bt-wf-sub">{Math.abs(result.walkForward.degradation) < 0.08 ? 'Low overfitting' : 'Watch overfitting'}</div>
          </div>
        </div>
      </div>

      {/* Signal attribution */}
      <div className="bt-section">
        <div className="bt-section-title">Signal Attribution</div>
        <div className="bt-sig-table">
          <div className="bt-sig-row bt-sig-header">
            <span>Signal</span>
            <span>Wt</span>
            <span>Win when aligned</span>
            <span>Win when opposed</span>
          </div>
          {[...result.signalAttribution]
            .sort((a, b) => b.winRateWhenAligned - a.winRateWhenAligned)
            .map(s => {
              const aligned  = (s.winRateWhenAligned * 100).toFixed(0);
              const opposed  = (s.winRateWhenOpposed * 100).toFixed(0);
              const diff     = s.winRateWhenAligned - s.winRateWhenOpposed;
              const isStrong = diff > 0.10;
              return (
                <div key={s.name} className={`bt-sig-row ${isStrong ? 'bt-sig-row--strong' : ''}`}>
                  <span className="bt-sig-name">{s.name}</span>
                  <span className="bt-sig-wt">{(s.weight * 100).toFixed(0)}%</span>
                  <span className={`bt-sig-wr ${Number(aligned) >= 55 ? 'bt-eq-up' : 'bt-eq-down'}`}>
                    {aligned}%
                    <small> ({s.alignedCount})</small>
                  </span>
                  <span className={`bt-sig-wr ${Number(opposed) >= 55 ? 'bt-eq-up' : 'bt-eq-down'}`}>
                    {opposed}%
                  </span>
                </div>
              );
            })}
        </div>
        <div className="bt-sig-hint">
          High "aligned" win rate + low "opposed" win rate = strong predictive signal
        </div>
      </div>

      {/* Confidence calibration */}
      <div className="bt-section">
        <div className="bt-section-title">Confidence Calibration</div>
        <div className="bt-conf-table">
          <div className="bt-conf-row bt-conf-header">
            <span>Confidence</span>
            <span>Trades</span>
            <span>Win Rate</span>
            <span>Avg Return</span>
          </div>
          {result.confidenceBuckets.filter(b => b.count > 0).map(b => (
            <div key={b.label} className="bt-conf-row">
              <span>{b.label}</span>
              <span>{b.count}</span>
              <span className={`${b.winRate >= 0.55 ? 'bt-eq-up' : 'bt-eq-down'}`}>
                {(b.winRate * 100).toFixed(0)}%
              </span>
              <span className={b.avgReturn >= 0 ? 'bt-eq-up' : 'bt-eq-down'}>
                {b.avgReturn >= 0 ? '+' : ''}{(b.avgReturn * 100).toFixed(2)}%
              </span>
            </div>
          ))}
        </div>
        <div className="bt-sig-hint">
          Win rate should increase with confidence — if not, confidence scoring needs re-calibration
        </div>
      </div>

      {/* Recent trades toggle */}
      <button className="bt-trades-toggle" onClick={() => setShowTrades(v => !v)}>
        {showTrades ? 'Hide' : 'Show'} Recent Trades ({result.trades.length})
      </button>

      {showTrades && (
        <div className="bt-trades">
          {[...result.trades].reverse().slice(0, 30).map((t, i) => (
            <div key={i} className={`bt-trade-row ${t.result === 'WIN' ? 'bt-trade--win' : 'bt-trade--loss'}`}>
              <span className={`bt-trade-dir ${t.direction === 'LONG' ? 'bt-eq-up' : 'bt-eq-down'}`}>
                {t.direction}
              </span>
              <span className="bt-trade-conf">{t.confidence.toFixed(0)}%</span>
              <span className="bt-trade-entry">${t.entryPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
              <span className={`bt-trade-pnl ${t.pnlPct >= 0 ? 'bt-eq-up' : 'bt-eq-down'}`}>
                {t.pnlPct >= 0 ? '+' : ''}{(t.pnlPct * 100).toFixed(2)}%
              </span>
              <span className="bt-trade-exit">{t.exitReason}</span>
              <span className="bt-trade-hold">{t.holdingCandles}c</span>
            </div>
          ))}
        </div>
      )}

    </div>
  );
}
