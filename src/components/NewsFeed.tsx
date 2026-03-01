import { useEffect, useRef } from 'react';
import { NewsItem } from '../types';
import { scoreSentiment } from '../utils/sentiment';

interface Props {
  news: NewsItem[];
  highlightedId: string | null;
  onItemClick: (publishedAt: number, id: string) => void;
}

function timeAgo(ts: number): string {
  const diff = Math.floor(Date.now() / 1000) - ts;
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

// Coins we track
const TRACKED_COINS = ['BTC', 'ETH', 'SOL'];
const COIN_COLORS: Record<string, string> = {
  BTC: '#f7931a', ETH: '#627eea', SOL: '#9945ff',
};

// Macro categories that move crypto prices — with display labels + colors
const MACRO_TAGS: Record<string, { label: string; color: string }> = {
  // Global macro / geopolitical (stream 3)
  MacroEcon:   { label: 'MACRO',   color: '#ea580c' },
  GeoPolitic:  { label: 'GEO',     color: '#b91c1c' },
  // Crypto-specific macro (stream 2)
  Regulation:  { label: 'REG',     color: '#dc2626' },
  Exchange:    { label: 'EXCH',    color: '#2563eb' },
  Hack:        { label: 'HACK',    color: '#dc2626' },
  Stablecoin:  { label: 'STABLE',  color: '#16a34a' },
  Market:      { label: 'MARKET',  color: '#6b7280' },
  Blockchain:  { label: 'CHAIN',   color: '#7c3aed' },
  Mining:      { label: 'MINING',  color: '#d97706' },
  Trading:     { label: 'TRADE',   color: '#0891b2' },
};

// Left-border color by sentiment
const SENTIMENT_BORDER: Record<string, string> = {
  bullish: 'var(--green)',
  bearish: 'var(--red)',
  neutral: 'var(--border)',
};

function getCoins(categories: string): string[] {
  return TRACKED_COINS.filter(c => categories.includes(c));
}

function getMacroTags(categories: string): { label: string; color: string }[] {
  return Object.entries(MACRO_TAGS)
    .filter(([key]) => categories.includes(key))
    .map(([, val]) => val);
}

export function NewsFeed({ news, highlightedId, onItemClick }: Props) {
  const listRef    = useRef<HTMLDivElement>(null);
  const prevLen    = useRef(0);
  const itemRefs   = useRef<Record<string, HTMLDivElement | null>>({});

  // Auto-scroll to top when new items arrive
  useEffect(() => {
    if (news.length > prevLen.current && listRef.current) {
      listRef.current.scrollTop = 0;
    }
    prevLen.current = news.length;
  }, [news.length]);

  // Scroll highlighted item into view
  useEffect(() => {
    if (highlightedId && itemRefs.current[highlightedId]) {
      itemRefs.current[highlightedId]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [highlightedId]);

  return (
    <div className="news-panel">
      <div className="panel-header">
        <span className="live-dot" />
        <span>Market News</span>
        {news.length > 0 && <span className="news-count">{news.length}</span>}
      </div>

      <div className="news-list" ref={listRef}>
        {news.length === 0 ? (
          <div className="news-empty">Fetching market news…</div>
        ) : (
          news.map(item => {
            const sentiment    = scoreSentiment(item.title);
            const coins        = getCoins(item.categories);
            const macroTags    = getMacroTags(item.categories);
            const isHighlighted = item.id === highlightedId;
            const borderColor  = isHighlighted
              ? 'var(--accent)'
              : SENTIMENT_BORDER[sentiment];

            return (
              <div
                key={item.id}
                ref={el => { itemRefs.current[item.id] = el; }}
                className={`news-item ${isHighlighted ? 'news-item--highlighted' : ''}`}
                style={{ borderLeftColor: borderColor }}
                onClick={() => onItemClick(item.publishedAt, item.id)}
              >
                <div className="news-meta-row">
                  <span className="news-source">{item.source}</span>
                  <span className="news-time">{timeAgo(item.publishedAt)}</span>
                </div>

                <div className="news-title">{item.title}</div>

                <div className="news-footer">
                  <div className="news-coins">
                    {/* Coin chips */}
                    {coins.map(c => (
                      <span
                        key={c}
                        className="news-coin"
                        style={{ color: COIN_COLORS[c], borderColor: `${COIN_COLORS[c]}44` }}
                      >
                        {c}
                      </span>
                    ))}
                    {/* Macro category chips */}
                    {macroTags.map(tag => (
                      <span
                        key={tag.label}
                        className="news-coin news-tag"
                        style={{ color: tag.color, borderColor: `${tag.color}44` }}
                      >
                        {tag.label}
                      </span>
                    ))}
                  </div>
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="news-link"
                    onClick={e => e.stopPropagation()}
                  >
                    ↗
                  </a>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
