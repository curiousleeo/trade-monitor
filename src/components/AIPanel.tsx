import { useState } from 'react';
import { Coin, Prediction, TFBias, Trade } from '../types';
import { StoredPortfolio } from '../lib/db';

interface Props {
  portfolio: StoredPortfolio;
  closedTrades: Trade[];
  predictions: Record<Coin, Prediction | null>;
  tfMatrix: TFBias[];
  activeCoin: Coin;
  livePrice: number | null;
  onReset: () => void;
  onForceEntry: (coin: Coin) => void;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmtUsd(v: number, decimals = 2): string {
  return `$${v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
}

function fmtPct(v: number): string {
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
}

function directionIcon(d: string): string {
  if (d === 'LONG') return '▲';
  if (d === 'SHORT') return '▼';
  return '→';
}

function directionClass(d: string): string {
  if (d === 'LONG') return 'up';
  if (d === 'SHORT') return 'down';
  return 'neutral';
}

function livePnl(trade: Trade, currentPrice: number | null): number | null {
  if (!currentPrice || trade.status !== 'OPEN') return trade.pnl ?? null;
  const units = trade.size / trade.entryPrice;
  return trade.direction === 'LONG'
    ? units * (currentPrice - trade.entryPrice)
    : units * (trade.entryPrice - currentPrice);
}

function winRate(closedTrades: Trade[]): string {
  if (closedTrades.length === 0) return '—';
  const wins = closedTrades.filter(t => t.status === 'CLOSED_TP').length;
  return `${wins}/${closedTrades.length} (${((wins / closedTrades.length) * 100).toFixed(0)}%)`;
}

function predAccuracy(closed: Trade[]): string {
  if (closed.length === 0) return '—';
  const wins = closed.filter(t => t.status === 'CLOSED_TP').length;
  return `${wins}/${closed.length} (${((wins / closed.length) * 100).toFixed(0)}%)`;
}

// ─── Signal Bar ───────────────────────────────────────────────────────────────

function SignalBar({ name, value, weight, description }: { name: string; value: number; weight: number; description: string }) {
  const pct = ((value + 100) / 2); // map -100..100 → 0..100
  const isPos = value >= 0;
  return (
    <div className="signal-row" title={description}>
      <div className="signal-row-header">
        <span className="signal-name">{name}</span>
        <span className={`signal-value ${isPos ? 'up' : 'down'}`}>{isPos ? '+' : ''}{value.toFixed(0)}</span>
        <span className="signal-weight">{(weight * 100).toFixed(0)}%</span>
      </div>
      <div className="signal-bar-track">
        <div className="signal-bar-center" />
        <div
          className={`signal-bar-fill ${isPos ? 'signal-bar-fill--up' : 'signal-bar-fill--down'}`}
          style={{
            width:  `${Math.abs(pct - 50)}%`,
            left:   isPos ? '50%' : `${pct}%`,
          }}
        />
      </div>
      <div className="signal-desc">{description}</div>
    </div>
  );
}

// ─── Prediction Card ──────────────────────────────────────────────────────────

function PredictionCard({ pred, coin }: { pred: Prediction | null; coin: Coin }) {
  if (!pred) {
    return (
      <div className="prediction-card prediction-card--loading">
        <div className="pred-loading-text">Analyzing {coin} market…</div>
        <div className="pred-loading-sub">Waiting for candle data</div>
      </div>
    );
  }

  const dc = directionClass(pred.direction);

  return (
    <div className={`prediction-card prediction-card--${dc}`}>
      {/* Header row */}
      <div className="pred-header">
        <div className="pred-dir-badge" data-dir={pred.direction}>
          {directionIcon(pred.direction)} {pred.direction}
        </div>
        <div className="pred-coin">{coin} / USDT · {pred.timeframe}</div>
        <div className="pred-confidence">
          <span className="pred-conf-val">{pred.confidence.toFixed(0)}</span>
          <span className="pred-conf-label">% conf</span>
        </div>
      </div>

      {/* Levels */}
      <div className="pred-levels">
        <div className="pred-level">
          <span className="pred-level-label">Entry</span>
          <span className="pred-level-val">
            {fmtUsd(pred.entryZone[0], 0)}–{fmtUsd(pred.entryZone[1], 0)}
          </span>
        </div>
        <div className="pred-level">
          <span className="pred-level-label">Target</span>
          <span className="pred-level-val up">{fmtUsd(pred.targetPrice, 0)}</span>
        </div>
        <div className="pred-level">
          <span className="pred-level-label">Stop</span>
          <span className="pred-level-val down">{fmtUsd(pred.stopPrice, 0)}</span>
        </div>
        <div className="pred-level">
          <span className="pred-level-label">R:R</span>
          <span className="pred-level-val">
            1:{(Math.abs(pred.targetPrice - pred.entryZone[0]) / Math.abs(pred.stopPrice - pred.entryZone[0])).toFixed(1)}
          </span>
        </div>
      </div>

      {/* Signal bars */}
      <div className="pred-signals">
        {pred.signals.map(s => (
          <SignalBar key={s.name} {...s} />
        ))}
      </div>

      {/* Reasoning */}
      <div className="pred-reasoning">{pred.reasoning}</div>
    </div>
  );
}

// ─── TF Matrix ────────────────────────────────────────────────────────────────

function TFMatrix({ matrix }: { matrix: TFBias[] }) {
  return (
    <div className="tf-matrix">
      <div className="tf-matrix-title">TIMEFRAME CONFLUENCE</div>
      <div className="tf-matrix-rows">
        {matrix.map(({ timeframe, direction, score }) => (
          <div key={timeframe} className="tf-row">
            <span className="tf-label">{timeframe.toUpperCase()}</span>
            <div className="tf-bar-track">
              <div
                className={`tf-bar-fill tf-bar-fill--${directionClass(direction)}`}
                style={{ width: `${Math.abs(score - 50) * 2}%`, marginLeft: score >= 50 ? '50%' : `${score}%` }}
              />
              <div className="tf-bar-mid" />
            </div>
            <span className={`tf-dir ${directionClass(direction)}`}>
              {directionIcon(direction)} {direction}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Trade Card ───────────────────────────────────────────────────────────────

function TradeCard({ trade, currentPrice }: { trade: Trade; currentPrice: number | null }) {
  const pnl = livePnl(trade, trade.status === 'OPEN' ? currentPrice : null);
  const isOpen = trade.status === 'OPEN';

  return (
    <div className={`trade-card trade-card--${isOpen ? 'open' : trade.status === 'CLOSED_TP' ? 'tp' : 'sl'}`}>
      <div className="trade-card-header">
        <span className={`trade-dir-badge ${trade.direction === 'LONG' ? 'up' : 'down'}`}>
          {directionIcon(trade.direction)} {trade.direction}
        </span>
        <span className="trade-coin">{trade.coin}</span>
        <span className={`trade-status trade-status--${isOpen ? 'open' : trade.status === 'CLOSED_TP' ? 'tp' : 'sl'}`}>
          {isOpen ? 'OPEN' : trade.status === 'CLOSED_TP' ? '✓ TP HIT' : '✗ SL HIT'}
        </span>
      </div>
      <div className="trade-levels">
        <span className="trade-level-item">
          <span className="trade-level-label">Entry</span>
          <span>{fmtUsd(trade.entryPrice, 0)}</span>
        </span>
        <span className="trade-level-item">
          <span className="trade-level-label">SL</span>
          <span className="down">{fmtUsd(trade.stopLoss, 0)}</span>
        </span>
        <span className="trade-level-item">
          <span className="trade-level-label">TP</span>
          <span className="up">{fmtUsd(trade.takeProfit, 0)}</span>
        </span>
        <span className="trade-level-item">
          <span className="trade-level-label">Size</span>
          <span>{fmtUsd(trade.size, 0)}</span>
        </span>
      </div>
      {pnl !== null && (
        <div className={`trade-pnl ${pnl >= 0 ? 'up' : 'down'}`}>
          {fmtUsd(pnl)} ({fmtPct((pnl / trade.size) * 100)})
        </div>
      )}
    </div>
  );
}

// ─── Main AIPanel ─────────────────────────────────────────────────────────────

export function AIPanel({ portfolio, closedTrades, predictions, tfMatrix, activeCoin, livePrice, onReset, onForceEntry }: Props) {
  const [subTab, setSubTab] = useState<'signals' | 'trades'>('signals');
  const pred = predictions[activeCoin];

  const totalPnl = portfolio.balance - portfolio.startBalance;
  const totalPnlPct = (totalPnl / portfolio.startBalance) * 100;
  const progress = (portfolio.balance / 100000) * 100;

  const allCoins: Coin[] = [
    'BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'TRX',
    'SOL', 'AVAX', 'DOT', 'LINK', 'ATOM', 'NEAR', 'UNI', 'ADA',
    'DOGE', 'SUI', 'APT', 'ARB', 'OP', 'INJ',
    'PAXG',
  ];

  return (
    <div className="ai-panel">
      {/* Sub-tabs */}
      <div className="ai-sub-tabs">
        <button
          className={`ai-sub-tab ${subTab === 'signals' ? 'active' : ''}`}
          onClick={() => setSubTab('signals')}
        >
          📊 SIGNALS
        </button>
        <button
          className={`ai-sub-tab ${subTab === 'trades' ? 'active' : ''}`}
          onClick={() => setSubTab('trades')}
        >
          📋 TRADES
        </button>
      </div>

      {subTab === 'signals' && (
        <div className="ai-signals-tab">
          {/* Coin selector row for predictions */}
          <div className="pred-coin-row">
            {allCoins.map(c => {
              const p = predictions[c];
              return (
                <div key={c} className={`pred-coin-pill ${directionClass(p?.direction ?? 'NEUTRAL')} ${c === activeCoin ? 'active' : ''}`}>
                  <span>{c}</span>
                  <span>{p ? `${directionIcon(p.direction)} ${p.confidence.toFixed(0)}%` : '…'}</span>
                </div>
              );
            })}
          </div>

          <PredictionCard pred={pred} coin={activeCoin} />

          {pred && pred.direction !== 'NEUTRAL' && !portfolio.openTrades.some(t => t.coin === activeCoin) && (
            <button
              className="force-entry-btn"
              onClick={() => onForceEntry(activeCoin)}
            >
              ⚡ FORCE ENTRY — {pred.direction} {activeCoin}
            </button>
          )}

          {tfMatrix.length > 0 && <TFMatrix matrix={tfMatrix} />}
        </div>
      )}

      {subTab === 'trades' && (
        <div className="ai-trades-tab">
          {/* Portfolio summary */}
          <div className="portfolio-summary">
            <div className="portfolio-row">
              <span className="portfolio-label">Balance</span>
              <span className="portfolio-val">{fmtUsd(portfolio.balance)}</span>
            </div>
            <div className="portfolio-row">
              <span className="portfolio-label">Total P&L</span>
              <span className={`portfolio-val ${totalPnl >= 0 ? 'up' : 'down'}`}>
                {fmtUsd(totalPnl)} ({fmtPct(totalPnlPct)})
              </span>
            </div>
            <div className="portfolio-row">
              <span className="portfolio-label">Win Rate</span>
              <span className="portfolio-val">{winRate(closedTrades)}</span>
            </div>
            <div className="portfolio-row">
              <span className="portfolio-label">Pred Accuracy</span>
              <span className="portfolio-val">{predAccuracy(closedTrades)}</span>
            </div>
            {/* $1k → $100k progress */}
            <div className="portfolio-progress">
              <div className="portfolio-progress-label">
                <span>$1K → $100K Challenge</span>
                <span>{progress.toFixed(2)}%</span>
              </div>
              <div className="portfolio-progress-track">
                <div className="portfolio-progress-fill" style={{ width: `${Math.min(100, progress)}%` }} />
              </div>
            </div>
            <button className="reset-btn" onClick={() => { if (confirm('Reset APEX portfolio to $1,000?')) onReset(); }}>
              Reset Portfolio
            </button>
          </div>

          {/* Open positions */}
          {portfolio.openTrades.length > 0 && (
            <div className="trades-section">
              <div className="trades-section-title">OPEN POSITIONS ({portfolio.openTrades.length})</div>
              {portfolio.openTrades.map(t => (
                <TradeCard key={t.id} trade={t} currentPrice={t.coin === activeCoin ? livePrice : null} />
              ))}
            </div>
          )}

          {/* Closed trades */}
          {closedTrades.length > 0 && (
            <div className="trades-section">
              <div className="trades-section-title">HISTORY ({closedTrades.length})</div>
              {closedTrades.slice(0, 20).map(t => (
                <TradeCard key={t.id} trade={t} currentPrice={null} />
              ))}
            </div>
          )}

          {portfolio.openTrades.length === 0 && closedTrades.length === 0 && (
            <div className="trades-empty">
              <div>Waiting for high-confidence signal…</div>
              <div className="trades-empty-sub">APEX needs ≥65% confidence to enter a trade</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
