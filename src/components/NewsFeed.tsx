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

const COIN_COLORS: Record<string, string> = {
  BTC: '#f7931a', ETH: '#627eea', SOL: '#9945ff',
};

const SENTIMENT_BORDER: Record<string, string> = {
  bullish: '#10b981',
  bearish: '#f43f5e',
  neutral: '#1f2333',
};

function getCoins(categories: string): string[] {
  return ['BTC', 'ETH', 'SOL'].filter(c => categories.includes(c));
}

export function NewsFeed({ news, highlightedId, onItemClick }: Props) {
  const listRef = useRef<HTMLDivElement>(null);
  const prevLengthRef = useRef(0);
  const itemRefs = useRef<Record<string, HTMLDivElement | null>>({});

  useEffect(() => {
    if (news.length > prevLengthRef.current && listRef.current) {
      listRef.current.scrollTop = 0;
    }
    prevLengthRef.current = news.length;
  }, [news.length]);

  useEffect(() => {
    if (highlightedId && itemRefs.current[highlightedId]) {
      itemRefs.current[highlightedId]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [highlightedId]);

  return (
    <div className="news-panel">
      <div className="panel-header">
        <span className="live-dot" />
        <span>Live News</span>
        {news.length > 0 && <span className="news-count">{news.length}</span>}
      </div>

      <div className="news-list" ref={listRef}>
        {news.length === 0 ? (
          <div className="news-empty">Fetching latest news...</div>
        ) : (
          news.map(item => {
            const sentiment  = scoreSentiment(item.title);
            const coins      = getCoins(item.categories);
            const isHighlighted = item.id === highlightedId;
            const borderColor   = isHighlighted ? '#f59e0b' : SENTIMENT_BORDER[sentiment];

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
                    {coins.map(c => (
                      <span key={c} className="news-coin" style={{ color: COIN_COLORS[c] }}>
                        {c}
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
