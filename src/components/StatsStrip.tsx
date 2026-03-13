import { useState, useEffect, useRef } from 'react';
import { FearGreed, FundingRate, TickerData } from '../types';

// ─── Market Session ────────────────────────────────────────────────────────────

// US market session boundaries in ET (minutes since midnight)
const SESSIONS = [
  { name: 'Pre-Market',   start:  4 * 60,       end:  9 * 60 + 30, color: '#f59f00' },
  { name: 'Market Open',  start:  9 * 60 + 30,  end: 16 * 60,      color: '#26a69a' },
  { name: 'After-Hours',  start: 16 * 60,        end: 20 * 60,      color: '#4c8df5' },
  { name: 'Closed',       start: 20 * 60,        end: 28 * 60,      color: '#787b86' }, // 28*60 = 4am next day
];

// Timeline segments for the hover tooltip bar
const TIMELINE = [
  { label: '12 AM', pct: 0 },
  { label: '4 AM',  pct: (4 / 24) * 100 },
  { label: '9:30',  pct: (9.5 / 24) * 100 },
  { label: '4 PM',  pct: (16 / 24) * 100 },
  { label: '8 PM',  pct: (20 / 24) * 100 },
];

const SESSION_SEGMENTS = [
  { start: 0,          end: (4/24)*100,    color: 'var(--text4)' },   // closed 12am–4am
  { start: (4/24)*100, end: (9.5/24)*100,  color: '#f59f0066' },      // pre-market
  { start: (9.5/24)*100, end: (16/24)*100, color: '#26a69a66' },      // market open
  { start: (16/24)*100, end: (20/24)*100,  color: '#4c8df566' },      // after-hours
  { start: (20/24)*100, end: 100,          color: 'var(--text4)' },   // closed 8pm–12am
];

function getETMinutes(): { minutes: number; isWeekend: boolean } {
  const now = new Date();
  // Convert to ET (UTC-5 standard, UTC-4 daylight)
  const utc = now.getTime() + now.getTimezoneOffset() * 60000;
  const etOffset = isDST(now) ? -4 : -5;
  const et = new Date(utc + etOffset * 3600000);
  const minutes = et.getHours() * 60 + et.getMinutes();
  const day = et.getDay(); // 0=Sun, 6=Sat
  return { minutes, isWeekend: day === 0 || day === 6 };
}

function isDST(date: Date): boolean {
  const jan = new Date(date.getFullYear(), 0, 1).getTimezoneOffset();
  const jul = new Date(date.getFullYear(), 6, 1).getTimezoneOffset();
  return date.getTimezoneOffset() < Math.max(jan, jul);
}

function getSession() {
  const { minutes, isWeekend } = getETMinutes();
  if (isWeekend) return { name: 'Closed', color: '#787b86', minutesLeft: null };
  // Wrap minutes past midnight into the overnight "closed" band
  const m = minutes < 4 * 60 ? minutes + 24 * 60 : minutes;
  for (const s of SESSIONS) {
    if (m >= s.start && m < s.end) {
      return { name: s.name, color: s.color, minutesLeft: s.end - m };
    }
  }
  return { name: 'Closed', color: '#787b86', minutesLeft: null };
}

function MarketSession() {
  const [session, setSession]     = useState(getSession);
  const [visible, setVisible]     = useState(false);
  const tooltipRef                = useRef<HTMLDivElement>(null);
  const wrapRef                   = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const t = setInterval(() => setSession(getSession()), 30000);
    return () => clearInterval(t);
  }, []);

  const { minutes } = getETMinutes();
  const dayPct = (Math.min(minutes, 24 * 60) / (24 * 60)) * 100;

  const timeLeft = session.minutesLeft !== null
    ? `${Math.floor(session.minutesLeft / 60)}h ${session.minutesLeft % 60}m left`
    : null;

  return (
    <div
      ref={wrapRef}
      className="market-session-wrap"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      <span className="stat-label">MARKET</span>
      <span className="market-session-dot" style={{ background: session.color }} />
      <span className="stat-value" style={{ color: session.color }}>{session.name}</span>
      {timeLeft && <span className="stat-sub">{timeLeft}</span>}

      {visible && (
        <div ref={tooltipRef} className="market-tooltip">
          <div className="market-tooltip-title">
            <span className="market-tooltip-dot" style={{ background: session.color }} />
            US Market · <span style={{ color: session.color }}>{session.name}</span>
          </div>
          {session.minutesLeft !== null ? (
            <div className="market-tooltip-sub">
              Session ends in {Math.floor(session.minutesLeft / 60)}h {session.minutesLeft % 60}m
            </div>
          ) : (
            <div className="market-tooltip-sub">Pre-market opens at 4:00 AM ET</div>
          )}
          {/* Timeline bar */}
          <div className="market-timeline">
            <div className="market-timeline-track">
              {SESSION_SEGMENTS.map((seg, i) => (
                <div key={i} className="market-timeline-seg" style={{
                  left: `${seg.start}%`, width: `${seg.end - seg.start}%`, background: seg.color,
                }} />
              ))}
              <div className="market-timeline-now" style={{ left: `${dayPct}%` }} />
            </div>
            <div className="market-timeline-labels">
              {TIMELINE.map(t => (
                <div key={t.label} className="market-timeline-label" style={{ left: `${t.pct}%` }}>
                  {t.label}
                </div>
              ))}
            </div>
          </div>
          <div className="market-tooltip-note">Times in US Eastern (ET)</div>
        </div>
      )}
    </div>
  );
}

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

      {/* Market Session */}
      <div className="stat-item" style={{ position: 'relative' }}>
        <MarketSession />
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
