import { FearGreed, FundingRate, TickerData } from '../types';

interface Props {
  fearGreed: FearGreed | null;
  fundingRate: FundingRate | null;
  ticker: TickerData | null;
}

function fgColor(v: number): string {
  if (v <= 25) return '#f43f5e';
  if (v <= 45) return '#f97316';
  if (v <= 55) return '#eab308';
  if (v <= 75) return '#84cc16';
  return '#10b981';
}

function fgShortLabel(label: string): string {
  return label.replace('Extreme ', 'EX ');
}

function formatVolume(v: number): string {
  if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (v >= 1e9)  return `$${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6)  return `$${(v / 1e6).toFixed(0)}M`;
  return `$${v.toFixed(0)}`;
}

function formatPrice(v: number): string {
  if (v >= 1000) return `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
  return `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function StatsStrip({ fearGreed, fundingRate, ticker }: Props) {
  const fgVal   = fearGreed?.value ?? null;
  const fgLabel = fearGreed ? fgShortLabel(fearGreed.label) : null;
  const fgClr   = fgVal !== null ? fgColor(fgVal) : '#64748b';

  const fundRate = fundingRate?.rate ?? null;
  const fundIsUp = (fundRate ?? 0) >= 0;

  return (
    <div className="stats-strip">

      {/* Fear & Greed */}
      <div className="stat-item">
        <span className="stat-label">F&G</span>
        <div className="fg-bar-track">
          <div
            className="fg-bar-fill"
            style={{ width: `${fgVal ?? 50}%`, background: fgClr }}
          />
        </div>
        <span className="stat-value" style={{ color: fgClr }}>
          {fgVal ?? '—'}
        </span>
        <span className="stat-sub" style={{ color: fgClr }}>
          {fgLabel ?? ''}
        </span>
      </div>

      {/* Funding Rate */}
      <div className="stat-item">
        <span className="stat-label">FUNDING</span>
        <span className={`stat-value ${fundIsUp ? 'up' : 'down'}`}>
          {fundRate !== null ? `${fundIsUp ? '+' : ''}${(fundRate * 100).toFixed(4)}%` : '—'}
        </span>
        <span className={`stat-sub ${fundIsUp ? 'up' : 'down'}`}>
          {fundRate !== null ? (fundIsUp ? 'longs pay' : 'shorts pay') : ''}
        </span>
      </div>

      {/* 24H Volume */}
      <div className="stat-item">
        <span className="stat-label">24H VOL</span>
        <span className="stat-value">
          {ticker ? formatVolume(ticker.volume24h) : '—'}
        </span>
      </div>

      {/* 24H Range */}
      <div className="stat-item">
        <span className="stat-label">24H RANGE</span>
        <span className="stat-value" style={{ color: '#f87171' }}>
          {ticker ? formatPrice(ticker.low24h) : '—'}
        </span>
        <span className="stat-sub">—</span>
        <span className="stat-value" style={{ color: '#4ade80' }}>
          {ticker ? formatPrice(ticker.high24h) : '—'}
        </span>
      </div>

    </div>
  );
}
