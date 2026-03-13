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

function getETDateTime(): string {
  const now = new Date();
  const utc = now.getTime() + now.getTimezoneOffset() * 60000;
  const et  = new Date(utc + (isDST(now) ? -4 : -5) * 3600000);
  return et.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })
    + ' · '
    + et.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })
    + ' ET';
}

function getSessionDesc(session: ReturnType<typeof getSession>): string {
  const h = session.minutesLeft !== null ? Math.floor(session.minutesLeft / 60) : 0;
  const m = session.minutesLeft !== null ? session.minutesLeft % 60 : 0;
  const t = `${h > 0 ? `${h}h ` : ''}${m}m`;
  switch (session.name) {
    case 'Pre-Market':  return `Pre-market trading is active. The regular market opens in ${t}.`;
    case 'Market Open': return `The market is open for regular trading and will close in ${t}.`;
    case 'After-Hours': return `After-hours trading is active. The session closes in ${t}.`;
    default:            return 'The market is currently closed. Pre-market opens at 4:00 AM ET on weekdays.';
  }
}

function MarketSession() {
  const [session,  setSession]  = useState(getSession);
  const [etTime,   setEtTime]   = useState(getETDateTime);
  const [tooltipPos, setTooltipPos] = useState<{ top: number; left: number } | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const t = setInterval(() => {
      setSession(getSession());
      setEtTime(getETDateTime());
    }, 1000);
    return () => clearInterval(t);
  }, []);

  const showTooltip = () => {
    if (!wrapRef.current) return;
    const r = wrapRef.current.getBoundingClientRect();
    const cardW = 296;
    const margin = 8;
    // Calculate the left EDGE of the card (centered under trigger, clamped to viewport)
    const rawLeft = r.left + r.width / 2 - cardW / 2;
    const left = Math.min(Math.max(rawLeft, margin), window.innerWidth - cardW - margin);
    setTooltipPos({ top: r.bottom + 6, left });
  };
  const hideTooltip = () => setTooltipPos(null);

  const { minutes } = getETMinutes();
  const dayPct   = (Math.min(minutes, 24 * 60) / (24 * 60)) * 100;
  const timeLeft = session.minutesLeft !== null
    ? `${Math.floor(session.minutesLeft / 60)}h ${session.minutesLeft % 60}m`
    : null;
  const isLive   = session.name !== 'Closed';

  return (
    <div
      ref={wrapRef}
      className="market-session-wrap"
      onMouseEnter={showTooltip}
      onMouseLeave={hideTooltip}
    >
      <span className="stat-label">MARKET</span>
      <span className={`market-session-dot${isLive ? ' market-session-dot--live' : ''}`} style={{ background: session.color }} />
      <span className="stat-value" style={{ color: session.color }}>{session.name}</span>
      {timeLeft && <span className="stat-sub">{timeLeft} left</span>}

      {tooltipPos && (
        <div
          className="market-tooltip"
          style={{ position: 'fixed', top: tooltipPos.top, left: tooltipPos.left }}
        >
          {/* Card header */}
          <div className="market-tooltip-header">
            <div className="market-tooltip-header-left">
              <span className={`market-tooltip-icon${isLive ? ' market-tooltip-icon--live' : ''}`} style={{ color: session.color }}>◉</span>
              <span className="market-tooltip-session" style={{ color: session.color }}>{session.name}</span>
            </div>
            <span className="market-tooltip-time">{etTime}</span>
          </div>

          {/* Description */}
          <div className="market-tooltip-desc">{getSessionDesc(session)}</div>

          {/* Session legend */}
          <div className="market-tooltip-legend">
            {SESSIONS.map(s => (
              <div key={s.name} className={`market-legend-item${session.name === s.name ? ' active' : ''}`}>
                <span className="market-legend-dot" style={{ background: s.color }} />
                <span>{s.name}</span>
              </div>
            ))}
          </div>

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
