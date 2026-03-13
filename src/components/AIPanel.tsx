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

// ─── Section Header ───────────────────────────────────────────────────────────

function SectionHeader({ label, meta }: { label: string; meta?: string }) {
  return (
    <div className="apex-section-header">
      <span className="apex-section-label">{label}</span>
      {meta && <span className="apex-section-meta">{meta}</span>}
    </div>
  );
}

// ─── Portfolio Status Bar ─────────────────────────────────────────────────────

function PortfolioBar({
  portfolio, closedTrades, unrealizedPnl, totalPnl, totalPnlPct, onReset,
}: {
  portfolio: StoredPortfolio;
  closedTrades: Trade[];
  unrealizedPnl: number;
  totalPnl: number;
  totalPnlPct: number;
  onReset: () => void;
}) {
  const progress = (portfolio.balance / 100000) * 100;
  const realizedPnl = portfolio.balance - portfolio.startBalance;

  return (
    <div className="apex-portfolio-bar">
      <div className="apex-portfolio-top">
        <div className="apex-bal-group">
          <span className="apex-bal-label">Balance</span>
          <span className="apex-bal-val">{fmtUsd(portfolio.balance)}</span>
        </div>
        <div className={`apex-pnl-group ${totalPnl >= 0 ? 'up' : 'down'}`}>
          <span className="apex-pnl-val">{fmtUsd(totalPnl)}</span>
          <span className="apex-pnl-pct">{fmtPct(totalPnlPct)}</span>
        </div>
        <button className="apex-reset-btn" onClick={() => { if (confirm('Reset APEX portfolio to $1,000?')) onReset(); }}>↺</button>
      </div>

      <div className="apex-portfolio-stats">
        <div className="apex-stat-pill">
          <span className="apex-stat-label">Realized</span>
          <span className={`apex-stat-val ${realizedPnl >= 0 ? 'up' : 'down'}`}>{fmtUsd(realizedPnl)}</span>
        </div>
        <div className="apex-stat-pill">
          <span className="apex-stat-label">Unrealized</span>
          <span className={`apex-stat-val ${unrealizedPnl >= 0 ? 'up' : 'down'}`}>{fmtUsd(unrealizedPnl)}</span>
        </div>
        <div className="apex-stat-pill">
          <span className="apex-stat-label">Win Rate</span>
          <span className="apex-stat-val">{winRate(closedTrades)}</span>
        </div>
        <div className="apex-stat-pill">
          <span className="apex-stat-label">Open</span>
          <span className="apex-stat-val">{portfolio.openTrades.length}</span>
        </div>
      </div>

      <div className="apex-progress-wrap">
        <div className="apex-progress-labels">
          <span>$1K → $100K</span>
          <span>{progress.toFixed(3)}%</span>
        </div>
        <div className="apex-progress-track">
          <div className="apex-progress-fill" style={{ width: `${Math.min(100, progress)}%` }} />
        </div>
      </div>
    </div>
  );
}

// ─── Signal Bar ───────────────────────────────────────────────────────────────

function SignalBar({ name, value, weight, description }: { name: string; value: number; weight: number; description: string }) {
  const pct = ((value + 100) / 2);
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
          style={{ width: `${Math.abs(pct - 50)}%`, left: isPos ? '50%' : `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ─── Prediction Card ──────────────────────────────────────────────────────────

function PredictionCard({ pred, coin }: { pred: Prediction | null; coin: Coin }) {
  if (!pred) {
    return (
      <div className="prediction-card prediction-card--loading">
        <div className="pred-loading-text">Analyzing {coin}…</div>
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
        <div className="pred-coin">{coin} · {pred.timeframe}</div>
        <div className="pred-conf-badge">{pred.confidence.toFixed(0)}%</div>
      </div>

      <div className="pred-levels">
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
          {isOpen ? 'OPEN' : trade.status === 'CLOSED_TP' ? '✓ TP' : '✗ SL'}
        </span>
        {pnl !== null && (
          <span className={`trade-pnl ${pnl >= 0 ? 'up' : 'down'}`}>
            {fmtUsd(pnl)}
          </span>
        )}
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
    </div>
  );
}

// ─── Coin Selector Pills ──────────────────────────────────────────────────────

const ALL_COINS: Coin[] = [
  'BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'TRX',
  'SOL', 'AVAX', 'DOT', 'LINK', 'ATOM', 'NEAR', 'UNI', 'ADA',
  'DOGE', 'SUI', 'APT', 'ARB', 'OP', 'INJ', 'PAXG',
];

// ─── Main AIPanel ─────────────────────────────────────────────────────────────

export function AIPanel({
  portfolio, closedTrades, predictions, tfMatrix, activeCoin,
  tickers, learnedWeights, activityLog, onReset, onForceEntry,
}: Props) {
  const pred = predictions[activeCoin];

  const unrealizedPnl = portfolio.openTrades.reduce((sum, t) => {
    const price = tickers[t.coin]?.price ?? null;
    const pnl = livePnl(t, price);
    return sum + (pnl ?? 0);
  }, 0);

  const realizedPnl = portfolio.balance - portfolio.startBalance;
  const totalPnl    = realizedPnl + unrealizedPnl;
  const totalPnlPct = (totalPnl / portfolio.startBalance) * 100;

  return (
    <div className="ai-panel">
      <div className="ai-dashboard">

        {/* ── 1. Portfolio Status ─────────────────────── */}
        <PortfolioBar
          portfolio={portfolio}
          closedTrades={closedTrades}
          unrealizedPnl={unrealizedPnl}
          totalPnl={totalPnl}
          totalPnlPct={totalPnlPct}
          onReset={onReset}
        />

        {/* ── 2. Coin Scanner ────────────────────────── */}
        <SectionHeader label="COIN SCANNER" />
        <div className="pred-coin-row">
          {ALL_COINS.map(c => {
            const p = predictions[c];
            return (
              <div key={c} className={`pred-coin-pill ${directionClass(p?.direction ?? 'NEUTRAL')} ${c === activeCoin ? 'active' : ''}`}>
                <span>{c}</span>
                <span>{p ? `${directionIcon(p.direction)} ${p.confidence.toFixed(0)}%` : '…'}</span>
              </div>
            );
          })}
        </div>

        {/* ── 3. Active Prediction ───────────────────── */}
        <SectionHeader label="PREDICTION" meta={activeCoin} />
        <PredictionCard pred={pred} coin={activeCoin} />

        {pred && pred.direction !== 'NEUTRAL' && !portfolio.openTrades.some(t => t.coin === activeCoin) && (
          <button className="force-entry-btn" onClick={() => onForceEntry(activeCoin)}>
            ⚡ FORCE ENTRY — {pred.direction} {activeCoin}
          </button>
        )}

        {/* ── 4. TF Confluence ───────────────────────── */}
        {tfMatrix.length > 0 && (
          <>
            <SectionHeader label="TIMEFRAME CONFLUENCE" />
            <TFMatrix matrix={tfMatrix} />
          </>
        )}

        {/* ── 5. Open Positions ──────────────────────── */}
        {portfolio.openTrades.length > 0 && (
          <>
            <SectionHeader label="OPEN POSITIONS" meta={String(portfolio.openTrades.length)} />
            {portfolio.openTrades.map(t => (
              <TradeCard key={t.id} trade={t} currentPrice={tickers[t.coin]?.price ?? null} />
            ))}
          </>
        )}

        {/* ── 6. Activity Log ────────────────────────── */}
        <SectionHeader label="ACTIVITY LOG" meta={activityLog.length > 0 ? `${activityLog.length} events` : undefined} />
        {activityLog.length === 0 ? (
          <div className="log-empty">Waiting for first trade…</div>
        ) : (
          <div className="log-feed">
            {activityLog.slice(0, 30).map(entry => (
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

        {/* ── 7. Model Weights ───────────────────────── */}
        <SectionHeader
          label="MODEL WEIGHTS"
          meta={learnedWeights.version > 0 ? `v${learnedWeights.version} · ${learnedWeights.trainedOn} trades` : 'v0 default'}
        />
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
                    <span className="learn-weight-delta">{diff > 0 ? ' ▲' : ' ▼'}{Math.abs(diff * 100).toFixed(1)}</span>
                  )}
                </span>
                <span className="learn-accuracy">{acc != null ? `${(acc * 100).toFixed(0)}%` : '—'}</span>
              </div>
            );
          })}
        </div>

        {/* ── 8. Closed Trade History ─────────────────── */}
        {closedTrades.length > 0 && (
          <>
            <SectionHeader label="TRADE HISTORY" meta={`${closedTrades.length} trades`} />
            {closedTrades.slice(0, 20).map(t => (
              <TradeCard key={t.id} trade={t} currentPrice={null} />
            ))}
          </>
        )}

        <div style={{ height: 24 }} />
      </div>
    </div>
  );
}
