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
  if (diff < 60) return `${diff}s`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
  return `${Math.floor(diff / 86400)}d`;
}

const COIN_COLORS: Record<string, string> = {
  BTC: '#f7931a', ETH: '#627eea', SOL: '#9945ff',
};

const SENTIMENT_LABELS: Record<string, string> = {
  bullish: '▲ BULL', bearish: '▼ BEAR', neutral: '— NEUT',
};

function getCoins(categories: string): string[] {
  return ['BTC', 'ETH', 'SOL'].filter(c => categories.includes(c));
}

export function NewsFeed({ news, highlightedId, onItemClick }: Props) {
  const listRef = useRef<HTMLDivElement>(null);
  const prevLengthRef = useRef(0);
  const itemRefs = useRef<Record<string, HTMLDivElement | null>>({});

  // Scroll to top on new items
  useEffect(() => {
    if (news.length > prevLengthRef.current && listRef.current) {
      listRef.current.scrollTop = 0;
    }
    prevLengthRef.current = news.length;
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
        <span>LIVE NEWS</span>
        <span className="news-count">{news.length}</span>
      </div>

      <div className="news-list" ref={listRef}>
        {news.length === 0 ? (
          <div className="news-empty">FETCHING FEED...</div>
        ) : (
          news.map(item => {
            const sentiment = scoreSentiment(item.title);
            const coins = getCoins(item.categories);
            const isHighlighted = item.id === highlightedId;

            return (
              <div
                key={item.id}
                ref={el => { itemRefs.current[item.id] = el; }}
                className={`news-item ${isHighlighted ? 'news-item--highlighted' : ''}`}
                onClick={() => onItemClick(item.publishedAt, item.id)}
              >
                <div className="news-top-row">
                  <span className="news-source">{item.source}</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span className={`sentiment-badge sentiment-${sentiment}`}>
                      {SENTIMENT_LABELS[sentiment]}
                    </span>
                    <span className="news-time">{timeAgo(item.publishedAt)}</span>
                  </div>
                </div>
                <div className="news-title">{item.title}</div>
                <div className="news-bottom-row">
                  {coins.length > 0 && (
                    <div className="news-tags">
                      {coins.map(c => (
                        <span key={c} className="news-tag" style={{ color: COIN_COLORS[c], borderColor: COIN_COLORS[c] }}>
                          {c}
                        </span>
                      ))}
                    </div>
                  )}
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
