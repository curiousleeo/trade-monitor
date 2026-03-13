import { useState } from 'react';
import { Coin, LogEntry, Prediction, TFBias, TickerData, Trade } from '../types';
import { StoredPortfolio } from '../lib/db';
import { LearnedWeights, DEFAULT_WEIGHTS } from '../ai/learner';

interface Props {
  portfolio:      StoredPortfolio;
  closedTrades:   Trade[];
  predictions:    Record<Coin, Prediction | null>;
  tfMatrix:       TFBias[];
  activeCoin:     Coin;
  tickers:        Record<Coin, TickerData | null>;
  learnedWeights: LearnedWeights;
  activityLog:    LogEntry[];
  onReset:        () => void;
  onForceEntry:   (coin: Coin) => void;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmtUsd(v: number, decimals?: number): string {
  const abs = Math.abs(v);
  const d = decimals ?? (abs >= 100 ? 2 : abs >= 1 ? 2 : abs >= 0.01 ? 4 : 6);
  return `$${v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })}`;
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
  const rr = (Math.abs(pred.targetPrice - pred.entryZone[0]) / Math.abs(pred.stopPrice - pred.entryZone[0])).toFixed(1);

  return (
    <div className={`prediction-card prediction-card--${dc}`}>
      <div className="pred-header">
        <div className="pred-dir-badge" data-dir={pred.direction}>
          {directionIcon(pred.direction)} {pred.direction}
        </div>
        <div className="pred-coin">{coin} / USDT · {pred.timeframe}</div>
      </div>

      <div className="pred-hero">
        <div className="pred-conf-ring">
          <span className="pred-conf-val">{pred.confidence.toFixed(0)}</span>
          <span className="pred-conf-label">% CONF</span>
        </div>
        <div className="pred-levels-col">
          <div className="pred-level-row">
            <span className="pred-level-label">Entry</span>
            <span className="pred-level-val">{fmtUsd(pred.entryZone[0])}</span>
          </div>
          <div className="pred-level-row">
            <span className="pred-level-label">Target</span>
            <span className="pred-level-val up">{fmtUsd(pred.targetPrice)}</span>
          </div>
          <div className="pred-level-row">
            <span className="pred-level-label">Stop</span>
            <span className="pred-level-val down">{fmtUsd(pred.stopPrice)}</span>
          </div>
          <div className="pred-level-row">
            <span className="pred-level-label">R:R</span>
            <span className="pred-level-val">1:{rr}</span>
          </div>
        </div>
      </div>

      <div className="pred-signals">
        {pred.signals.map(s => (
          <SignalBar key={s.name} {...s} />
        ))}
      </div>

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
          <span>{fmtUsd(trade.entryPrice)}</span>
        </span>
        <span className="trade-level-item">
          <span className="trade-level-label">SL</span>
          <span className="down">{fmtUsd(trade.stopLoss)}</span>
        </span>
        <span className="trade-level-item">
          <span className="trade-level-label">TP</span>
          <span className="up">{fmtUsd(trade.takeProfit)}</span>
        </span>
        <span className="trade-level-item">
          <span className="trade-level-label">Size</span>
          <span>{fmtUsd(trade.size)}</span>
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

export function AIPanel({ portfolio, closedTrades, predictions, tfMatrix, activeCoin, tickers, learnedWeights, activityLog, onReset, onForceEntry }: Props) {
  const [subTab, setSubTab] = useState<'signals' | 'trades' | 'log'>('signals');
  const pred = predictions[activeCoin];

  // Unrealized P&L across all open positions using live prices
  const unrealizedPnl = portfolio.openTrades.reduce((sum, t) => {
    const price = tickers[t.coin]?.price ?? null;
    const pnl = livePnl(t, price);
    return sum + (pnl ?? 0);
  }, 0);

  const realizedPnl  = portfolio.balance - portfolio.startBalance;
  const totalPnl     = realizedPnl + unrealizedPnl;
  const totalPnlPct  = (totalPnl / portfolio.startBalance) * 100;
  const progress     = (portfolio.balance / 100000) * 100;

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
        <button
          className={`ai-sub-tab ${subTab === 'log' ? 'active' : ''}`}
          onClick={() => setSubTab('log')}
        >
          🧠 LOG
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
            <div className="portfolio-hero">
              <div className="portfolio-hero-label">Apex Portfolio</div>
              <div className="portfolio-hero-balance">{fmtUsd(portfolio.balance)}</div>
              <div className={`portfolio-hero-pnl ${totalPnl >= 0 ? 'up' : 'down'}`}>
                {fmtUsd(totalPnl)} · {fmtPct(totalPnlPct)}
              </div>
            </div>
            <div className="portfolio-stats">
              <div className="portfolio-stat">
                <span className="portfolio-label">Win Rate</span>
                <span className="portfolio-val">{winRate(closedTrades)}</span>
              </div>
              <div className="portfolio-stat">
                <span className="portfolio-label">Open Trades</span>
                <span className="portfolio-val">{portfolio.openTrades.length}</span>
              </div>
              <div className="portfolio-stat">
                <span className="portfolio-label">Realized P&L</span>
                <span className={`portfolio-val ${realizedPnl >= 0 ? 'up' : 'down'}`}>{fmtUsd(realizedPnl)}</span>
              </div>
              <div className="portfolio-stat">
                <span className="portfolio-label">Unrealized</span>
                <span className={`portfolio-val ${unrealizedPnl >= 0 ? 'up' : 'down'}`}>{fmtUsd(unrealizedPnl)}</span>
              </div>
            </div>
            <div className="portfolio-progress-wrap">
              <div className="portfolio-progress-label">
                <span>$1K → $100K Challenge</span>
                <span>{progress.toFixed(3)}%</span>
              </div>
              <div className="portfolio-progress-track">
                <div className="portfolio-progress-fill" style={{ width: `${Math.min(100, progress)}%` }} />
              </div>
              <div className="portfolio-progress-milestones">
                <span>$1K</span><span>$10K</span><span>$50K</span><span>$100K</span>
              </div>
            </div>
            <button className="reset-btn" onClick={() => { if (confirm('Reset APEX portfolio to $1,000?')) onReset(); }}>
              ↺ Reset Portfolio
            </button>
          </div>

          {/* Open positions */}
          {portfolio.openTrades.length > 0 && (
            <div className="trades-section">
              <div className="trades-section-title">OPEN POSITIONS ({portfolio.openTrades.length})</div>
              {portfolio.openTrades.map(t => (
                <TradeCard key={t.id} trade={t} currentPrice={tickers[t.coin]?.price ?? null} />
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

      {subTab === 'log' && (
        <div className="ai-log-tab">

          {/* ── Activity feed ───────────────────────────────────── */}
          <div className="log-section-title">ACTIVITY LOG</div>
          {activityLog.length === 0 ? (
            <div className="log-empty">No activity yet — waiting for first trade…</div>
          ) : (
            <div className="log-feed">
              {activityLog.map(entry => (
                <div key={entry.id} className={`log-entry log-entry--${entry.level}`}>
                  <span className="log-time">
                    {new Date(entry.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })}
                  </span>
                  <div className="log-body">
                    <div className="log-message">{entry.message}</div>
                    {entry.detail && <div className="log-detail">{entry.detail}</div>}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* ── What Apex learned ───────────────────────────────── */}
          <div className="log-section-title" style={{ marginTop: 18 }}>WHAT APEX LEARNED</div>

          {/* Current weight table */}
          <div className="learn-weights-table">
            <div className="learn-weights-header">
              <span>Signal</span>
              <span>Default</span>
              <span>Current</span>
              <span>Accuracy</span>
            </div>
            {Object.keys(DEFAULT_WEIGHTS).map(name => {
              const def  = DEFAULT_WEIGHTS[name];
              const cur  = learnedWeights.weights[name] ?? def;
              const acc  = learnedWeights.accuracy[name];
              const diff = cur - def;
              return (
                <div key={name} className="learn-weights-row">
                  <span className="learn-signal-name">{name}</span>
                  <span className="learn-weight-def">{(def * 100).toFixed(0)}%</span>
                  <span className={`learn-weight-cur ${diff > 0.005 ? 'up' : diff < -0.005 ? 'down' : ''}`}>
                    {(cur * 100).toFixed(1)}%
                    {Math.abs(diff) > 0.005 && (
                      <span className="learn-weight-delta">
                        {diff > 0 ? ' ▲' : ' ▼'}{Math.abs(diff * 100).toFixed(1)}
                      </span>
                    )}
                  </span>
                  <span className="learn-accuracy">
                    {acc != null ? `${(acc * 100).toFixed(0)}%` : '—'}
                  </span>
                </div>
              );
            })}
          </div>

          {learnedWeights.version === 0 && (
            <div className="learn-status">
              Model v0 — using defaults. Needs {15 - closedTrades.length > 0 ? `${15 - closedTrades.length} more` : 'enough'} closed trades before first update.
            </div>
          )}

          {/* Learning history */}
          {learnedWeights.history.length > 0 && (
            <div className="learn-history">
              {learnedWeights.history.map((event, i) => (
                <div key={i} className="learn-event">
                  <div className="learn-event-header">
                    <span className="learn-event-title">Model v{event.version}</span>
                    <span className="learn-event-meta">
                      {new Date(event.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                      {' · '}{event.trainedOn} trades
                    </span>
                  </div>
                  <div className="learn-event-summary">{event.summary}</div>
                  {event.changes.filter(c => Math.abs(c.delta) > 0.003).map(c => (
                    <div key={c.signal} className={`learn-change learn-change--${c.delta >= 0 ? 'up' : 'down'}`}>
                      <span>{c.delta >= 0 ? '▲' : '▼'} {c.signal}</span>
                      <span>{(c.oldWeight * 100).toFixed(1)}% → {(c.newWeight * 100).toFixed(1)}%</span>
                      <span className="learn-change-acc">{(c.accuracy * 100).toFixed(0)}% acc</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}

        </div>
      )}
    </div>
  );
}
